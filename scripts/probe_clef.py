"""Latency probe for Clef Flash (OpenAI-compatible oMLX server) on prompts shaped like the bench_classify requests.

Reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment or from peft_experiments/.env. For each prompt size it
streams REPEATS chat completions and reports time to first token, total latency, and decode tokens/s. Clef Flash is a
reasoning chat model: for labels, send chat_template_kwargs={"enable_thinking": false} and a json_schema response_format
with the label enums (~3.5-4.6 s/example measured; with thinking on it ran past 1,024 tokens, >23 s, without answering).

  .venv-bench/bin/python scripts/probe_clef.py
"""
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_classify import INDIEMH, MHDIALOG, load_indiemh, load_mhdialog  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("CLEF_MODEL", "clef-flash-oQ4e")
REPEATS = 3


def env():
    vals = {}
    if (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1); vals[k.strip().removeprefix("export ").strip()] = v.strip().strip('"\'')
    base = os.environ.get("OPENAI_BASE_URL") or vals.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY") or vals.get("OPENAI_API_KEY")
    if not (base and key): sys.exit("OPENAI_BASE_URL and OPENAI_API_KEY must be set (environment or peft_experiments/.env)")
    return base.rstrip("/"), key


def prompt(state, spec):
    """All of a dataset's label questions in one prompt, answered as one JSON object (one request per example)."""
    qs = "\n".join(f'- "{qid}": {instr} Options: ' + "; ".join(f"{k} ({v})" if v else k for k, v in crit.items())
                   for qid, (instr, crit) in spec.items())
    return ("Transcript:\n" + "\n".join(state) + "\n\nAnswer each question with exactly one option name. "
            "Reply with only a JSON object keyed by question id.\n" + qs)


def call(base, key, text, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    start = time.perf_counter(); first = None; out = []; usage = {}
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for c in chunk.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                if piece:
                    first = first or time.perf_counter(); out.append(piece)
    end = time.perf_counter()
    return {"ttft_ms": round(((first or end) - start) * 1000, 1), "total_ms": round((end - start) * 1000, 1),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "text": "".join(out)}


def main():
    base, key = env()
    req = urllib.request.Request(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        print("models:", [m["id"] for m in json.loads(r.read())["data"]])
    mh = next(load_mhdialog(["test"])); im = next(load_indiemh(["test"], 6, "Code-Mix"))
    cases = [("hello", "Hello", 16), ("indiemh-turn", prompt(im["state"], INDIEMH), 64), ("mhdialog-dialogue", prompt(mh["state"], MHDIALOG), 64)]
    call(base, key, "warmup", 4)
    for name, text, max_tokens in cases:
        runs = [call(base, key, text, max_tokens) for _ in range(REPEATS)]
        decode = [(r["completion_tokens"] or 0) / max(1e-3, (r["total_ms"] - r["ttft_ms"]) / 1000) for r in runs]
        print(json.dumps({"case": name, "prompt_tokens": runs[-1]["prompt_tokens"], "completion_tokens": runs[-1]["completion_tokens"],
                          "ttft_ms": [r["ttft_ms"] for r in runs], "total_ms": [r["total_ms"] for r in runs],
                          "total_ms_median": statistics.median(r["total_ms"] for r in runs),
                          "decode_tok_s_median": round(statistics.median(decode), 1)}))
        print("  sample output:", runs[-1]["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
