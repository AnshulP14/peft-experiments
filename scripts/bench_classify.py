"""Kev vs Laya: zero-shot label prediction on MHDialog and IndieMH, at 4 / 8 / 16-bit weights, with per-request speed.

One invocation = one (model, bits) config, run over every example of the chosen datasets/splits. Each example is one
request asking all of that dataset's label questions at once (TypeSafe System One `choice` questions, which both models
take unchanged). Rows are appended to <out>/preds/<model>-<bits>bit.jsonl as they finish, so a killed job resumes where
it stopped. Configs can run concurrently for accuracy; their latencies then include contention, so speed is measured
in a separate solo pass (`--subdir speed --limit N`, one config at a time). `--summarize` turns every preds file in <out> into summary.json + summary.md (quality and speed tables).

Quantization: 16 = the weights as shipped (Kev: bf16 backbone; Laya: fp16 encoder). 4 / 8 = MLX affine group
quantization (group 64) of every nn.Linear in the backbone / encoder; embeddings and the decision heads stay as they are.

  .venv-bench/bin/python scripts/bench_classify.py --model kev-0.8b --bits 4 --limit 20     # smoke
  .venv-bench/bin/python scripts/bench_classify.py --summarize --out output/bench_classify/<run>
  ./run_bench_classify.sh                                                                     # full grid, detached
"""
import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MODELS = {"kev-0.8b": "jaredpalmer/kev-0.8b", "kev-4b": "jaredpalmer/kev-4b", "laya": "aac6fef/laya-mlx"}
BITS = (4, 8, 16)
GROUP_SIZE = 64

# ----------------------------------------------------------------------------- tasks (label sets from the dataset cards)

MHDIALOG = {
    "intent": ("Which dialog intent best describes the user's behaviour across this conversation?", {
        "Recovery": "user shows signs of improvement or willingness to seek help",
        "Adversarial": "user resists support or shows hostility",
        "Crisis Escalation": "the situation intensifies during the conversation",
        "Avoidance": "user avoids engaging with the support or topic",
        "Explicit Help-seeking": "user directly asks for help or resources",
        "Emotional Venting": "user expresses emotions without seeking solutions",
        "Implicit Help-seeking": "user indirectly signals a need for support",
        "Validation-Seeking": "user seeks acknowledgment or affirmation",
    }),
    "concern": ("What type of suicide-related concern does the user express?", {
        "Behavior": "self-harm behaviours or concerning actions",
        "Ideation": "suicidal thoughts or ideation",
        "Attempt": "history of or current suicide attempts",
        "Indicator": "warning signs or risk indicators",
        "Supportive": "seeking support for others",
        "Not Related": "not related to suicide or self-harm concerns",
        "Unsure": "cannot be determined",
    }),
    "level": ("What is the user's suicide risk level?", {
        "No": "no apparent risk indicators",
        "Minor": "minimal risk indicators",
        "Moderate": "some concerning elements present",
        "Severe": "immediate concern warranted",
        "Not Related": "content not related to mental health risk",
        "Unsure": "risk level cannot be determined",
    }),
}
MHDIALOG_COLS = {"intent": "Dialog Intent", "concern": "Concern Type", "level": "Level"}

EMOTIONS = ["Neutral", "Sadness", "Scared", "Annoyance", "Confusion", "Anticipation", "Contempt", "Joy", "Trust",
            "Serenity", "Disgust", "Anger", "Surprise"]
DIALOGUE_ACTS = {
    "id": "information delivery", "irq": "information request", "gc": "general chat", "crq": "clarification request",
    "cd": "clarification delivery", "ack": "acknowledgement", "yq": "yes/no question", "gt": "greeting",
    "on": "opinion or narrative", "od": "opinion delivery", "op": "opinion or perspective", "orq": "opinion request",
    "cr": "confirmation request", "cv": "confirmation", "ci": "closing",
}
SUBTOPICS = {
    "routine": "day-to-day activities and coping strategies",
    "inactive": "opening/closing exchanges, off-topic turns",
    "symp/reasoning": "symptom discussion and clinical reasoning",
    "story": "personal narrative and life events",
}
INDIEMH = {
    "emotion": ("Which emotion does the speaker express in the LAST utterance of this counseling transcript?", dict.fromkeys(EMOTIONS)),
    "dialogue_act": ("Which dialogue act is the LAST utterance of this counseling transcript?", DIALOGUE_ACTS),
    "subtopic": ("Which conversation sub-topic does the LAST utterance of this counseling transcript belong to?", SUBTOPICS),
}
SPEAKER = {"T": "Therapist", "P": "Patient"}


def questions(spec):
    return {qid: {"type": "choice", "instructions": instr, "criteria": crit} for qid, (instr, crit) in spec.items()}


def parse_dialogue(x):
    """The JSON turn list; some rows carry prose before it or are truncated (same recovery as prepare_data.conv_mhdialog)."""
    s = x.strip() if isinstance(x, str) else ""
    for candidate in (s, s[s.find("["):] if "[" in s else ""):
        try:
            v, _ = json.JSONDecoder().raw_decode(candidate)
            if isinstance(v, list) and v: return [t for t in v if isinstance(t, dict)]
        except ValueError:
            pass
    return None


def load_mhdialog(splits):
    """One example per dialogue; state = the turns as a list (Laya truncates a list state from the left). Unparseable
    dialogues are skipped and counted."""
    for split in splits:
        df = pd.read_csv(DATA / "MHDialog" / f"{split}.csv")
        skipped = 0
        for i, row in df.iterrows():
            turns = parse_dialogue(row["Dialogue"])
            state = [f"{who.capitalize()}: {t[who]}" for t in turns or [] for who in ("user", "supporter") if str(t.get(who, "")).strip()]
            if not state:
                skipped += 1; continue
            gold = {qid: row[col] for qid, col in MHDIALOG_COLS.items()}
            yield {"id": f"{split}_{i}", "split": split, "state": state, "gold": gold}
        if skipped: print(f"mhdialog {split}: skipped {skipped} unparseable dialogues", flush=True)


def load_indiemh(splits, context, text_col):
    """One example per utterance; state = up to `context` previous turns of the same session plus the utterance (last)."""
    names = {"val": "validation"}
    for split in splits:
        df = pd.read_parquet(DATA / "IndieMH" / "data" / f"{names.get(split, split)}-00000-of-00001.parquet")
        df = df[df["Speaker"].isin(SPEAKER)].reset_index(drop=True)
        df["conv"] = df["ID"].str.rsplit("_", n=1).str[0]
        for _, g in df.groupby("conv", sort=False):
            lines = [f"{SPEAKER[s]}: {u}" for s, u in zip(g["Speaker"], g[text_col].fillna(g["Utterance"]))]
            for k, (_, row) in enumerate(g.iterrows()):
                emo = [e.strip() for e in str(row["Emotion"]).split(",") if e.strip()]
                gold = {"emotion": emo, "dialogue_act": str(row["Dialogue_Act"]).strip(), "subtopic": str(row["Sub topic"]).strip()}
                yield {"id": row["ID"], "split": split, "speaker": row["Speaker"], "state": lines[max(0, k - context):k + 1], "gold": gold}


# ----------------------------------------------------------------------------- models


def linear_predicate(_path, module):
    import mlx.nn as nn
    return isinstance(module, nn.Linear) and module.weight.shape[-1] % GROUP_SIZE == 0


def mlx_quantize(module, bits):
    import mlx.core as mx
    import mlx.nn as nn
    if bits != 16:
        nn.quantize(module, group_size=GROUP_SIZE, bits=bits, class_predicate=linear_predicate)
    mx.eval(module.parameters())


def param_bytes(module):
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


class KevRunner:
    def __init__(self, repo, bits):
        from kev.checkpoint import Checkpoint, LoadOptions
        self.tok, self.model = Checkpoint(repo).load("mps", LoadOptions(backend="mlx"))
        mlx_quantize(self.model.text, bits)
        self.weight_bytes = param_bytes(self.model.text)

    def __call__(self, state, qs):
        from kev.api import SystemOneRequest, to_record
        from kev.model import admit
        rec, meta = to_record(SystemOneRequest.model_validate({"state": state, "questions": qs}))
        enc = admit(self.model, self.tok, rec, truncate=True)
        ps = self.model.probs(enc)
        probs = {m["id"]: dict(zip(m["keys"], [float(x) for x in (p.tolist() if hasattr(p, "tolist") else p)])) for m, p in zip(meta, ps)}
        return probs, {"input_tokens": len(enc["ids"]), "truncated": bool(enc.get("state_truncated", False))}


class LayaRunner:
    def __init__(self, repo, bits):
        import laya_mlx
        self.agent = laya_mlx.load(repo, dtype="float16", cache_prompts=False)
        mlx_quantize(self.agent.model.encoder, bits)
        self.agent._inference = self.agent.model   # Agent compiles nothing by default; keep it pointing at the quantized model
        self.weight_bytes = param_bytes(self.agent.model.encoder)

    def __call__(self, state, qs):
        out = self.agent.predict(state, qs)
        probs = {qid: a["probabilities"] for qid, a in out["answers"].items()}
        u = out["usage"]
        return probs, {"input_tokens": u["input_tokens"], "truncated": u["truncated"], "state_tokens_dropped": u["state_tokens_dropped"]}


def make_runner(name, bits):
    return (LayaRunner if name == "laya" else KevRunner)(MODELS[name], bits)


# ----------------------------------------------------------------------------- run


def _parses(line):
    try: json.loads(line); return True
    except ValueError: return False


def run(args):
    import mlx.core as mx
    out = Path(args.out); (out / args.subdir).mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}-{args.bits}bit"
    path = out / args.subdir / f"{tag}.jsonl"
    done = set()
    if path.exists():   # a kill mid-write can leave a half line: drop it, the example reruns
        lines = path.read_text().splitlines()
        good = [l for l in lines if l.strip() and _parses(l)]
        if len(good) != len([l for l in lines if l.strip()]): path.write_text("".join(l + "\n" for l in good))
        done = {(r["dataset"], r["id"]) for r in map(json.loads, good)}
    jobs = []
    if "mhdialog" in args.datasets:
        jobs.append(("mhdialog", questions(MHDIALOG), load_mhdialog(args.splits)))
    if "indiemh" in args.datasets:
        jobs.append(("indiemh", questions(INDIEMH), load_indiemh(args.splits, args.context, args.indiemh_text)))

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    runner = make_runner(args.model, args.bits)
    load_s = time.perf_counter() - t0
    runner(["warmup"], questions(MHDIALOG))   # first-call graph build and allocator growth, excluded
    meta = {"model": args.model, "repo": MODELS[args.model], "bits": args.bits, "group_size": GROUP_SIZE, "load_s": round(load_s, 2),
            "weight_bytes": runner.weight_bytes, "machine": platform.machine(), "mlx": mx.__version__, "argv": sys.argv[1:],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / args.subdir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)

    with path.open("a") as f:
        for dataset, qs, examples in jobs:
            n, t_ds = 0, time.perf_counter()
            for ex in examples:
                if (dataset, ex["id"]) in done: continue
                if args.limit and n >= args.limit: break
                start = time.perf_counter()
                try:
                    probs, usage = runner(ex["state"], qs)
                    err = None
                except Exception as e:   # one bad example never kills a long job
                    probs, usage, err = {}, {}, f"{type(e).__name__}: {e}"
                ms = (time.perf_counter() - start) * 1000
                pred = {qid: max(p, key=p.get) for qid, p in probs.items()}
                row = {"model": args.model, "bits": args.bits, "dataset": dataset, "split": ex["split"], "id": ex["id"],
                       "speaker": ex.get("speaker"), "gold": ex["gold"], "pred": pred,
                       "probs": {qid: {k: round(v, 5) for k, v in p.items()} for qid, p in probs.items()},
                       "latency_ms": round(ms, 2), **usage, **({"error": err} if err else {})}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                n += 1
                if n % args.log_every == 0:
                    rate = n / (time.perf_counter() - t_ds)
                    print(f"[{tag}] {dataset} {n} done  {rate:.2f} ex/s  last {ms:.0f} ms  peak {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
            print(f"[{tag}] {dataset} finished: {n} new examples in {time.perf_counter() - t_ds:.1f}s", flush=True)
    meta.update(peak_memory_gib=round(mx.get_peak_memory() / 2**30, 3), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / args.subdir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------- summary


def macro_f1(gold, pred):
    from sklearn.metrics import f1_score
    return f1_score(gold, pred, average="macro", zero_division=0) if gold else float("nan")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def read_rows(d):
    rows = defaultdict(list)
    for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for line in p.read_text().splitlines():
            if line.strip() and _parses(line):
                r = json.loads(line); rows[(r["model"], r["bits"], r["dataset"])].append(r)
    return rows


def speed_stats(rs, meta_path):
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    lat = [r["latency_ms"] for r in rs if "error" not in r]
    toks = sum(r.get("input_tokens", 0) for r in rs if "error" not in r)
    return {"speed_n": len(lat), "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
            "latency_ms_p50": round(pct(lat, .5), 1), "latency_ms_p90": round(pct(lat, .9), 1), "latency_ms_p99": round(pct(lat, .99), 1),
            "examples_per_s": round(len(lat) / (sum(lat) / 1000), 3) if lat else None,
            "input_tokens_per_s": round(toks / (sum(lat) / 1000), 1) if lat else None,
            "load_s": meta.get("load_s"), "weight_mib": round(meta["weight_bytes"] / 2**20, 1) if "weight_bytes" in meta else None,
            "peak_memory_gib": meta.get("peak_memory_gib")}


def summarize(out):
    out = Path(out)
    rows, speed_rows = read_rows(out / "preds"), read_rows(out / "speed")
    results = []
    for (model, bits, dataset), rs in sorted(rows.items()):
        ok = [r for r in rs if "error" not in r]
        solo = (model, bits, dataset) in speed_rows   # timings from the solo pass when it ran; else the (maybe concurrent) accuracy pass
        res = {"model": model, "bits": bits, "dataset": dataset, "n": len(rs), "errors": len(rs) - len(ok),
               "truncated_frac": round(sum(bool(r.get("truncated")) for r in ok) / max(1, len(ok)), 4),
               "speed_source": "solo" if solo else "accuracy pass",
               **speed_stats(speed_rows[(model, bits, dataset)] if solo else ok, out / ("speed" if solo else "preds") / f"{model}-{bits}bit.meta.json"),
               "tasks": {}}
        tasks = MHDIALOG if dataset == "mhdialog" else INDIEMH
        for qid, (_, crit) in tasks.items():
            pairs = [(r["gold"][qid], r["pred"].get(qid)) for r in ok if qid in r["pred"]]
            if qid == "emotion":   # multi-label gold: hit = prediction is one of the annotated emotions; F1 on the first-listed label
                hit = [p in g for g, p in pairs if g]
                first = [(g[0], p) for g, p in pairs if g]
                patient = [p in r["gold"][qid] for r in ok if r.get("speaker") == "P" and r["gold"][qid] and qid in r["pred"] for p in [r["pred"][qid]]]
                res["tasks"][qid] = {"n": len(hit), "acc": round(sum(hit) / max(1, len(hit)), 4),
                                     "macro_f1": round(macro_f1(*zip(*first)) if first else float("nan"), 4),
                                     "acc_patient_turns": round(sum(patient) / max(1, len(patient)), 4),
                                     "majority_acc": round(Counter(g[0] for g, _ in first).most_common(1)[0][1] / max(1, len(first)), 4) if first else None}
            else:                  # gold outside the documented label set is not scored
                pairs = [(g, p) for g, p in pairs if g in crit]
                acc = sum(g == p for g, p in pairs) / max(1, len(pairs))
                res["tasks"][qid] = {"n": len(pairs), "acc": round(acc, 4),
                                     "macro_f1": round(macro_f1(*zip(*pairs)) if pairs else float("nan"), 4),
                                     "majority_acc": round(Counter(g for g, _ in pairs).most_common(1)[0][1] / max(1, len(pairs)), 4) if pairs else None}
        results.append(res)
    (out / "summary.json").write_text(json.dumps(results, indent=2))

    md = ["# Kev vs Laya: label prediction on MHDialog / IndieMH", "",
          "Zero-shot `choice` questions, one request per example with all of the dataset's questions. "
          "Accuracy / macro-F1; emotion acc = prediction is one of the annotated emotions. majority = always-most-frequent baseline. "
          "Speed: 'solo' rows were timed with one config on the machine; 'accuracy pass' timings may include other configs running concurrently.", ""]
    for dataset in ("mhdialog", "indiemh"):
        rs = [r for r in results if r["dataset"] == dataset]
        if not rs: continue
        qids = list(rs[0]["tasks"])
        md += [f"## {dataset}", "", "| model | bits | n | " + " | ".join(f"{q} acc / F1" for q in qids) + " | trunc |",
               "|---|---|---|" + "---|" * len(qids) + "---|"]
        for r in rs:
            md.append(f"| {r['model']} | {r['bits']} | {r['n']} | " + " | ".join(f"{r['tasks'][q]['acc']:.3f} / {r['tasks'][q]['macro_f1']:.3f}" for q in qids) + f" | {r['truncated_frac']:.2f} |")
        md.append("| majority | – | – | " + " | ".join(f"{rs[0]['tasks'][q]['majority_acc']:.3f} / –" for q in qids) + " | – |")
        md += ["", "| model | bits | timed on | n timed | ex/s | tok/s | p50 ms | p90 ms | p99 ms | weights MiB | peak GiB | load s | errors |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            md.append(f"| {r['model']} | {r['bits']} | {r['speed_source']} | {r['speed_n']} | {r['examples_per_s']} | {r['input_tokens_per_s']} | {r['latency_ms_p50']} | "
                      f"{r['latency_ms_p90']} | {r['latency_ms_p99']} | {r['weight_mib']} | {r['peak_memory_gib']} | {r['load_s']} | {r['errors']} |")
        md.append("")
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--bits", type=int, choices=BITS, default=16)
    ap.add_argument("--datasets", nargs="+", default=["mhdialog", "indiemh"], choices=["mhdialog", "indiemh"])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--context", type=int, default=6, help="IndieMH: previous turns given with each utterance")
    ap.add_argument("--indiemh_text", default="Code-Mix", choices=["Code-Mix", "Utterance"], help="code-mixed Hinglish or the original English")
    ap.add_argument("--limit", type=int, default=0, help="max new examples per dataset (smoke tests)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "output" / "bench_classify" / "default"))
    ap.add_argument("--subdir", default="preds", help="preds = the accuracy pass (configs may run concurrently); speed = the solo timing pass")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize: return summarize(args.out)
    if not args.model: ap.error("--model is required unless --summarize")
    run(args)


if __name__ == "__main__":
    main()
