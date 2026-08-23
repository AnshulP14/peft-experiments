#!/usr/bin/env python3
"""Convert the raw mental-health datasets in data/ into the canonical
HF-chat format used by the PEFT finetuning pipeline (methods/common.py).

Canonical output, one folder per dataset:

    data/<name>/train.parquet   rows: {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
    data/<name>/val.parquet     same schema (used for eval_loss + samples)
    data/<name>/meta.json       provenance: source, license, counts, converter version

    data/manifest.json          dataset name -> folder

Design notes:
  * Plain text only — NO special tokens. The model-specific chat-template
    delimiters are applied at train time by ChatCollator (tokenizer-dependent),
    so one copy of the data serves every model.
  * Multi-turn is lossless: single-turn sources become 2-message conversations.
  * Idempotent: re-running rewrites the same outputs; raw files are never touched.

Usage (from the repo root, project venv):

    ../.venv/bin/python scripts/prepare_data.py            # all datasets
    ../.venv/bin/python scripts/prepare_data.py psydial    # one dataset
    ../.venv/bin/python scripts/prepare_data.py --list
"""

import argparse
import datetime as dt
import glob
import json
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SEED = 42
CONVERTER = "scripts/prepare_data.py v1"

ROLE_OK = {"user", "assistant", "system"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def u(content: str) -> dict:
    return {"role": "user", "content": str(content).strip()}


def a(content: str) -> dict:
    return {"role": "assistant", "content": str(content).strip()}


def drop_no_assistant(rows):
    """Rows without any assistant turn carry zero training signal — drop them."""
    kept = [r for r in rows if any(m["role"] == "assistant" for m in r["messages"])]
    return kept, len(rows) - len(kept)


def split_rng(n: int, val_frac: float):
    """Return (train_idx, val_idx) for a seeded 95/5-ish split of range(n)."""
    idx = list(range(n))
    # deterministic LCG — pandas's sample needs the series at call site; keep it simple
    import random

    rnd = random.Random(SEED)
    rnd.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    val = set(idx[:n_val])
    return [i for i in range(n) if i not in val], sorted(val)


def validate(rows, name):
    bad = 0
    for r in rows:
        msgs = r["messages"]
        if not msgs or any(m["role"] not in ROLE_OK or not m["content"] for m in msgs):
            bad += 1
    if bad:
        raise ValueError(f"{name}: {bad} rows with empty/invalid messages")
    return rows


def write_split(name: str, train_rows, val_rows, meta_extra: dict):
    base = os.path.join(DATA_DIR, name)
    os.makedirs(base, exist_ok=True)
    for split, rows in (("train", train_rows), ("val", val_rows)):
        if rows:
            pd.DataFrame(rows).to_parquet(os.path.join(base, f"{split}.parquet"), index=False)
    meta = {
        "name": name,
        "converter": CONVERTER,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        **meta_extra,
    }
    with open(os.path.join(base, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[prepare] {name}: train={len(train_rows)} val={len(val_rows)} -> {base}")
    return meta


# --------------------------------------------------------------------------- #
# converters — each returns (train_rows, val_rows, meta_extra)
# --------------------------------------------------------------------------- #
def conv_mentalchat():
    """MentalChat16K: alpaca-style CSV, real interviews only (synthetic excluded)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "MentalChat16K", "Interview_Data_6K.csv"))
    rows = []
    for r in df.itertuples(index=False):
        prompt = r.instruction + (f"\n{r.input}" if isinstance(r.input, str) and r.input else "")
        rows.append({"messages": [u(prompt), a(r.output)]})
    train_idx, val_idx = split_rng(len(rows), 0.05)
    return (
        [rows[i] for i in train_idx],
        [rows[i] for i in val_idx],
        {
            "source": "HF MentalChat16K / Interview_Data_6K.csv (real anonymized health-coach interviews)",
            "license": "MIT",
            "note": "Synthetic_Data_10K.csv excluded per request (GPT-3.5 generated)",
        },
    )


def conv_psydial():
    """PsyDial-D101: already OpenAI chat format; pass through (Chinese, kept)."""
    with open(os.path.join(DATA_DIR, "PsyDial-D101", "PsyDial-D101.json")) as fh:
        data = json.load(fh)
    rows = []
    for item in data:
        msgs = [
            {"role": m["role"], "content": str(m["content"]).strip()}
            for m in item["messages"]
            if m.get("role") in ROLE_OK and str(m.get("content", "")).strip()
        ]
        if msgs:
            rows.append({"messages": msgs})
    kept, no_asst = drop_no_assistant(rows)
    train_idx, val_idx = split_rng(len(kept), 0.05)
    return (
        [kept[i] for i in train_idx],
        [kept[i] for i in val_idx],
        {
            "source": "HF PsyDial-D101 (real counseling sessions, privacy-pipeline reconstructed)",
            "license": "CC0",
            "note": "multi-turn (median 13 messages); 'golden' last-turn response is included "
            f"in messages; dropped {no_asst} items with no assistant turn",
        },
    )


def conv_empathlm():
    """EmpathLM v2: paired single-turn; kept splits (train/test->val).
    Extra columns kept for a future DPO run (chosen=empathetic, rejected=unsafe)."""
    extra_cols = ["unsafe_response", "psychology_principle", "category"]

    def conv(df: pd.DataFrame):
        out = []
        for r in df.itertuples(index=False):
            row = {"messages": [u(r.situation), a(r.empathetic_response)]}
            for c in extra_cols:
                v = getattr(r, c, None)
                row[c] = v if isinstance(v, str) else None
            out.append(row)
        return out

    train_df = pd.read_parquet(glob.glob(os.path.join(DATA_DIR, "empathLM-dataset-v2", "data", "train-*.parquet"))[0])
    val_df = pd.read_parquet(glob.glob(os.path.join(DATA_DIR, "empathLM-dataset-v2", "data", "test-*.parquet"))[0])
    return (
        conv(train_df),
        conv(val_df),
        {
            "source": "HF empathLM-dataset-v2 (real counselor sessions, MI/CBT-aligned)",
            "license": "MIT",
            "note": "test-* split used as val; unsafe_response/principle/category cols kept for DPO",
        },
    )


def conv_mhdialog():
    """MHDialog: CSV with a JSON 'Dialogue' column of {round, user, supporter} turns.

    ~24 rows in the wild file are malformed: LLM refusal strings, prose
    preamble before the JSON, or genuinely truncated JSON. Recover what we
    can (strict parse, then raw_decode from first '['), drop the rest and
    report the count in meta."""
    from json import JSONDecoder

    def recover(x):
        if not isinstance(x, str):
            return None
        s = x.strip()
        try:
            v = json.loads(s)
            if isinstance(v, list) and v:
                return v
        except Exception:
            pass
        i = s.find("[")
        if i >= 0:
            try:
                v, _ = JSONDecoder().raw_decode(s[i:])
                if isinstance(v, list) and v:
                    return v
            except Exception:
                pass
        return None

    def conv(csv_path: str):
        df = pd.read_csv(csv_path)
        rows, dropped = [], 0
        for dlg in df["Dialogue"]:
            turns = recover(dlg)
            if turns is None:
                dropped += 1
                continue
            msgs = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                if str(turn.get("user", "")).strip():
                    msgs.append(u(turn["user"]))
                if str(turn.get("supporter", "")).strip():
                    msgs.append(a(turn["supporter"]))
            if msgs:
                rows.append({"messages": msgs})
            else:
                dropped += 1
        return rows, dropped

    train_rows, tr_drop = conv(os.path.join(DATA_DIR, "MHDialog", "train.csv"))
    val_rows, va_drop = conv(os.path.join(DATA_DIR, "MHDialog", "val.csv"))
    train_rows, tr_no_asst = drop_no_assistant(train_rows)
    val_rows, va_no_asst = drop_no_assistant(val_rows)
    return (
        train_rows,
        val_rows,
        {
            "source": "HF MHDialog (human-authored emotional-support dialogues)",
            "license": "see source repo",
            "note": "test.csv kept unconverted in raw folder; dropped rows "
            f"(unparseable/no content): train={tr_drop}, val={va_drop}; "
            f"dropped (no assistant turn): train={tr_no_asst}, val={va_no_asst}",
        },
    )


def conv_indiemh():
    """IndieMH: turn-level rows grouped into dialogues by ID prefix.
    ID looks like '102_0' -> dialogue 102, turn 0. Speaker T (therapist) ->
    assistant, P (patient) -> user. Consecutive same-speaker turns are merged."""

    ROLE = {"T": a, "P": u}

    def conv(parquet_path: str):
        df = pd.read_parquet(parquet_path)
        df = df[df["Speaker"].isin(ROLE)]
        df = df.copy()
        df["dlg"] = df["ID"].astype(str).str.rsplit("_", n=1).str[0]
        df["turn"] = df["ID"].astype(str).str.rsplit("_", n=1).str[1].astype(float)
        df = df.sort_values(["dlg", "turn"])
        rows = []
        for _, grp in df.groupby("dlg", sort=False):
            msgs = []
            for r in grp.itertuples(index=False):
                msg = ROLE[r.Speaker](r.Utterance)
                if msgs and msgs[-1]["role"] == msg["role"]:
                    msgs[-1]["content"] += "\n" + msg["content"]
                else:
                    msgs.append(msg)
            if msgs:
                rows.append({"messages": msgs})
        kept, _no_asst = drop_no_assistant(rows)
        return kept

    glob_train = glob.glob(os.path.join(DATA_DIR, "IndieMH", "data", "train-*.parquet"))
    glob_val = glob.glob(os.path.join(DATA_DIR, "IndieMH", "data", "validation-*.parquet"))
    if not glob_train or not glob_val:
        raise FileNotFoundError("IndieMH train/validation parquet not found")
    train_rows, val_rows = conv(glob_train[0]), conv(glob_val[0])
    return (
        train_rows,
        val_rows,
        {
            "source": "HF IndieMH (code-mixed Hindi-English real counseling, HOPE-based)",
            "license": "restricted — non-commercial academic use only",
            "note": "grouped by ID prefix; T->assistant, P->user; rows with Speaker='Type' "
            "dropped; dialogues without assistant turns dropped",
        },
    )


CONVERTERS = {
    "mentalchat16k": conv_mentalchat,
    "psydial": conv_psydial,
    "empathlm": conv_empathlm,
    "mhdialog": conv_mhdialog,
    "indiemh": conv_indiemh,
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("names", nargs="*", help="dataset name(s); default: all")
    p.add_argument("--list", action="store_true", help="list available datasets and exit")
    args = p.parse_args()

    if args.list or not args.names:
        names = list(CONVERTERS)
    else:
        unknown = [n for n in args.names if n not in CONVERTERS]
        if unknown:
            raise SystemExit(f"unknown dataset(s): {unknown}; available: {list(CONVERTERS)}")
        names = args.names
    if args.list:
        print("\n".join(names))
        return

    metas = {}
    for name in names:
        train_rows, val_rows, extra = CONVERTERS[name]()
        metas[name] = write_split(name, validate(train_rows, name), validate(val_rows, name), extra)

    manifest = {name: {"folder": name, "license": m["license"], "n_train": m["n_train"], "n_val": m["n_val"]} for name, m in metas.items()}
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[prepare] manifest written to {os.path.join(DATA_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()
