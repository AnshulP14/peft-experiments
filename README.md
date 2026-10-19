# PEFT Finetuning Experiments (Apple Silicon / MPS)

A reproducible setup for comparing PEFT finetuning methods on the **MPS**
backend, with a three-layer design:

1. **Method layer** — `methods/*.py`, one standalone script per PEFT method,
   each exposing `run_training(config, prune=None)`.
2. **Config layer** — every hyperparameter resolves as
   **CLI flag > `PEFT_*` env var > built-in default** (see `.env.example`).
   Steer single runs *or entire overnight studies* purely from the environment.
3. **Meta-tuning layer** — `tune.py`: an **Optuna** study (TPE sampler +
   MedianPruner) on top of the methods. Each trial = one full training job =
   one MLflow run. Bad trials are **pruned early** (reported every 10 steps),
   failed trials don't kill the study, and the study persists in SQLite so an
   overnight run can be killed and resumed in the morning.

**Observability: MLflow** — free, fully local, no account, no cloud.

## Layout

```
peft_experiments/
├── methods/
│   ├── common.py      # RunConfig (env layer), collator, MLflow + pruning callbacks, dataset cache
│   └── qlora.py       # QLoRA (NF4 4-bit + LoRA) — done, smoke-tested on MPS
├── scripts/
│   ├── prepare_data.py    # raw datasets -> canonical chat-format parquet (data/<name>/)
│   ├── smoke_collator.py  # collator unit checks (run after touching ChatCollator)
│   ├── bench_classify.py  # Kev vs Laya zero-shot label prediction (MHDialog, IndieMH) at 4/8/16-bit
│   └── probe_clef.py      # latency probe for Clef Flash on an OpenAI-compatible oMLX server
├── tune.py            # Optuna meta-tuning driver (search space per method)
├── data/              # raw + canonical datasets (gitignored; rebuilt via scripts/prepare_data.py)
├── run_overnight.sh   # detached study launcher (auto-resumes on re-run)
├── run_bench_classify.sh  # detached label-prediction benchmark (resumes on re-run)
├── Makefile
├── .env.example       # all PEFT_* variables documented
└── output/            # runs, mlflow store, tuning state (gitignored)
```

## Datasets

Five mental-health conversation datasets live in `data/` (see `data/README.md`
for provenance). `scripts/prepare_data.py` converts them to the canonical
format — one plain-text `messages` row per conversation
(`[{"role": "user"|"assistant", "content": ...}]`), split into
`data/<name>/train.parquet` + `val.parquet`, plus `meta.json` and a
`data/manifest.json` index:

| name | rows (train/val) | shape |
|---|---|---|
| `mentalchat16k` | 5995/315 | single-turn |
| `psydial` | 1119/58 | multi-turn (up to 149 msgs, Chinese) |
| `empathlm` | 684/76 | single-turn, DPO pairs kept in extra cols |
| `mhdialog` | 690/150 | multi-turn (10 rounds) |
| `indiemh` | 142/21 | multi-turn dialogue, code-mixed Hindi-English |

No special tokens are written: the model's chat-template delimiters are
applied at train time by `ChatCollator` (the one model-specific point), so one
copy of the data serves every model.

```bash
../.venv/bin/python scripts/prepare_data.py            # (re)build all
make train DATASET=psydial                             # finetune on one
make train DATASET=empathlm MODEL=Qwen/Qwen2.5-1.5B-Instruct
PEFT_NUM_TRAIN=64 PEFT_EPOCHS=1 make train DATASET=psydial   # smoke (~2 min)
```

## Quick start

```bash
cd peft_experiments
make install
make serve            # MLflow UI at http://127.0.0.1:5000

make qlora-smoke      # ~1 min: single QLoRA run, 64 rows, 1 epoch
```

## Env-var config

Everything is settable via `PEFT_*` (full list in `.env.example`):

```bash
PEFT_MODEL=Qwen/Qwen2.5-1.5B-Instruct PEFT_EPOCHS=2 PEFT_R=32 python methods/qlora.py
#   ↑ CLI beats env beats default; unset flags fall through to env
```

Key knobs: `PEFT_MODEL`, `PEFT_DATASET_NAME`, `PEFT_NUM_TRAIN`, `PEFT_EPOCHS`,
`PEFT_BATCH_SIZE`, `PEFT_LEARNING_RATE`, `PEFT_MAX_LENGTH`, `PEFT_R`,
`PEFT_LORA_ALPHA`, `PEFT_QUANT_TYPE`, `PEFT_SEED`, `PEFT_MLFLOW_TRACKING_URI`.

## Overnight experimentation

```bash
# 150 trials (or 12h) of QLoRA hyperparameter search, detached:
./run_overnight.sh
# or with a different budget / model:
PEFT_MODEL=Qwen/Qwen2.5-1.5B-Instruct N_TRIALS=100 TIMEOUT=10h ./run_overnight.sh

# morning: study auto-resumes (or: python tune.py --study <name> --n_trials 50 --timeout 6h)
tail -f output/tuning/<study>/run.log
```

What the tuner searches for QLoRA:

| param | space |
|---|---|
| r | {8, 16, 32, 64} (alpha = 2r) |
| learning_rate | 5e-5 … 5e-4 (log) |
| batch_size × grad_accum | {2,4,8} × {1,2,4} |
| max_length | {256, 512} |
| num_train | {500, 1000, 2000} |
| epochs | {2, 3} |
| lora_dropout | {0.0, 0.05} |

Search-space choices are env-overridable (`PEFT_TUNE_NUM_TRAIN=256,512`, ...) —
scale the study to your RAM/night without editing code.

At the end of a study `tune.py` writes `output/tuning/<study>/best_params.json`
+ `trials.csv`, retrains the best config `--repeats` times with different seeds
(error bars), and logs a `<study>-summary` MLflow run with artifacts.

**Sizing a night** (0.5B model, 4-bit, measured): one trial ≈ 5–15 min
(num_train 500–2000); the pruner kills ~half the trials early, so
`N_TRIALS=150, TIMEOUT=12h` is a reasonable overnight default on this machine.

## Label-prediction benchmark (Kev vs Laya)

`scripts/bench_classify.py` asks small decision models to predict the
datasets' annotated labels, zero-shot, through TypeSafe System One `choice`
questions (one request per example with all of that dataset's questions):

| dataset | unit | labels |
|---|---|---|
| MHDialog | dialogue (~1k) | intent (8), concern type (7), risk level (6) |
| IndieMH | utterance (11.5k, code-mixed Hinglish + 6 turns of context) | emotion (13, multi-label gold), dialogue act (15), sub-topic (4) |

Models: `kev-0.8b` / `kev-4b` (`jaredpalmer/kev-*`, MLX backend) and `laya`
(`aac6fef/laya-mlx`). Bits: 16 = weights as shipped (Kev bf16, Laya fp16);
8 / 4 = MLX group-64 quantization of every `nn.Linear` in the backbone.
Every row logs gold, prediction, probabilities, latency and token counts;
`--summarize` writes accuracy / macro-F1 (vs a majority baseline) and speed
tables to `summary.md`.

```bash
uv venv .venv-bench --python 3.13
uv pip install --python .venv-bench/bin/python -e ../kev mlx mlx-lm laya-mlx pandas pyarrow

.venv-bench/bin/python scripts/bench_classify.py --model laya --bits 4 --splits test --limit 20   # smoke
./run_bench_classify.sh                                  # Laya + Kev-0.8B x 16/8/4, all splits, detached
RUN=<run> ./run_bench_classify.sh                        # resume
MODELS="kev-4b" SPLITS="test" ./run_bench_classify.sh    # Kev-4B swaps on 16 GB (~5 s/example)
```

The launcher runs an accuracy pass over everything, then a speed pass with
each config alone on 300 examples per dataset; the latencies in `summary.md`
come from the speed pass. Configs run one at a time by default (`JOBS=1`):
on an M2 one config already keeps the GPU busy, and three concurrent Laya
configs ran at about half the combined rate of one.

Measured on an M2 16 GB (smoke tests): Laya ~0.5 s and Kev-0.8B ~0.5–0.7 s
per example at 16-bit; 4-bit saves memory but is ~15–20% slower. Laya reads
at most 512 tokens, so every MHDialog dialogue is truncated to its last turns.

**Clef Flash** is served as a generative reasoning chat model (oMLX,
OpenAI-compatible), not a classifier, and only as 4-bit (`clef-flash-oQ4e`).
`scripts/probe_clef.py` (needs `OPENAI_BASE_URL` / `OPENAI_API_KEY` in the
environment or `.env`) measures it: with thinking off and a `json_schema`
response format limited to the label names, ~3.5–4.6 s per example, almost
all prompt processing (~280 tok/s); with thinking on it ran past 1,024 tokens
without answering.

## What QLoRA uses here (verified working on MPS)

- base model quantized to **4-bit NF4** via `bitsandbytes`
  (`bnb_4bit_compute_dtype=float32` — f16/bf16 compute is unstable on MPS)
- LoRA adapters on all attention+MLP projections, f32 training on MPS
  (MPS mixed-precision is flaky; fine at ≤1B params)
- gradient checkpointing (`use_reentrant=False`)
- loss masked to the assistant response only (chat-template aware)
- default model `Qwen/Qwen2.5-0.5B-Instruct` (public, chat template, fits any
  M-series); scale to 1.5B+ via `PEFT_MODEL`
- default dataset `tatsu-lab/alpaca` (52k instruct samples; `--num_train` caps
  it). `prompt`/`response` or `instruction`/`input`/`output` columns both work;
  processed parquet cached in `data/`.

## Adding a new PEFT method

1. Copy `methods/qlora.py` → `methods/lora.py`, drop the quantization config.
2. Add a branch in `tune.py:suggest_params(trial, method)` for its search space.
3. Add a Makefile target. That's it — env layer, MLflow, pruning, resuming,
   summary, and error bars all come for free.

## Caveats

- MPS quirks: keep batch size modest (4 works for 0.5B in 4-bit); if you OOM,
  lower `PEFT_BATCH_SIZE` and raise `PEFT_GRAD_ACCUM`.
- Model downloads cache in `~/.cache/huggingface`; the default model is public.
- MLflow state: `output/mlruns.db` + `output/mlflow-artifacts/`.
- Optuna state: `output/tuning/study.db` (never delete mid-experiment).
