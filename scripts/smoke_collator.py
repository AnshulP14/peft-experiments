"""Collator smoke test: multi-turn masking + legacy prompt/response + batch padding.

Run:  ../.venv/bin/python scripts/smoke_collator.py
"""
import sys

sys.path.insert(0, "methods")

from transformers import AutoTokenizer  # noqa: E402

from common import ChatCollator  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
col = ChatCollator(tok, max_length=512)

# Qwen delimiter tokens, built from parts to keep this file copy-paste safe
IM_START = tok.convert_tokens_to_ids("<|" + "im_start" + "|>")
IM_END = tok.convert_tokens_to_ids("<|" + "im_end" + "|>")
assert IM_START and IM_END

# 1) multi-turn
msgs = [
    {"role": "user", "content": "I feel like I'm going through the motions."},
    {"role": "assistant", "content": "That sounds exhausting. Can you tell me more?"},
    {"role": "user", "content": "Work has been overwhelming lately."},
    {"role": "assistant", "content": "Let's break that down together."},
]
ids, lbl = col._encode(msgs)
# Qwen2.5 template auto-prepends a default system turn -> 5 boundaries for 4 messages
assert ids.count(IM_START) == 5 and ids.count(IM_END) == 5, "delimiters not injected"
labeled = [p for p, l in enumerate(lbl) if l != -100]
assert labeled, "no labeled tokens"
# every labeled token must fall after an im_start that follows an im_end or start-of-seq
first_labeled = min(labeled)
# first labeled token is the assistant's opening im_start — strictly after the
# system turn's im_end, so user/system text is never trained on
assert ids.index(IM_END) < first_labeled and ids[first_labeled] == IM_START, "training on user text"
# last labeled token of content must be im_end (render ends with im_end + newline
# separator — the newline is part of the template diff and fine to train on)
assert ids[-2] == IM_END and lbl[-2] == IM_END, "trailing im_end missing from labels"
print(f"multi-turn: {len(ids)} tokens, {len(labeled)} labeled — OK")

# 2) legacy prompt/response
ids2, lbl2 = col._encode(
    [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
)
# Qwen2.5 template auto-prepends a default system turn -> 3 boundaries for 2 messages
assert ids2.count(IM_START) == 3 and ids2.count(IM_END) == 3 and any(l != -100 for l in lbl2)
print(f"legacy: {len(ids2)} tokens — OK")

# 3) batch padding via __call__ (mixed multi-turn + prompt/response rows)
batch = col(
    [
        {"messages": msgs},
        {"prompt": "hi", "response": "yo"},
    ]
)
assert batch["input_ids"].shape == batch["labels"].shape and batch["input_ids"].shape[0] == 2
assert batch["labels"][0, 0] == -100 and batch["labels"][1, 0] == -100, "left padding must be masked"
print(f"batch: {tuple(batch['input_ids'].shape)} — OK")
print("collator smoke: ALL PASS")
