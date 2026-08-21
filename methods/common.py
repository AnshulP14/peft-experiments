"""Shared plumbing for PEFT finetuning experiments on MPS.

Every method file (qlora.py, lora.py, ...) calls:

    config = RunConfig.from_args(args, method="qlora")
    metrics = run_training(config, prune=optional_optuna_trial)

and gets: dataset loading/caching, chat-template collator (loss masked to the
response), QLoRA/model building is method-specific (see each method file),
MLflow run logging, optional Optuna pruning, sample generation.

Config resolution order for every hyperparameter:
    CLI flag  >  PEFT_* env var  >  built-in default
so an overnight sweep can be steered purely from the environment, e.g.:

    PEFT_MODEL=Qwen/Qwen2.5-1.5B-Instruct PEFT_EPOCHS=2 python methods/qlora.py
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time
from typing import Any, Callable, Optional

import mlflow
import torch
from datasets import Dataset, load_dataset
from transformers import TrainerCallback

MLFLOW_DEFAULT_URI = "http://127.0.0.1:5000"
ENV_PREFIX = "PEFT_"


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by every method script. All default to None so that
    unset CLI flags fall through to PEFT_* env vars, then built-ins."""
    def opt(name: str, **kw):
        kw.setdefault("default", None)
        parser.add_argument(name, **kw)

    opt("--model", help="HF model id [env PEFT_MODEL]")
    opt("--dataset_name", help="HF dataset id [env PEFT_DATASET_NAME]")
    opt("--dataset_subset", help="Optional dataset config name [env PEFT_DATASET_SUBSET]")
    opt("--output_dir", help="Base output dir [env PEFT_OUTPUT_DIR]")
    opt("--run_name", help="MLflow run name [env PEFT_RUN_NAME]")
    opt("--num_train", type=int, help="Cap on training rows [env PEFT_NUM_TRAIN]")
    opt("--num_eval", type=int, help="Cap on eval rows [env PEFT_NUM_EVAL]")
    opt("--epochs", type=int, help="[env PEFT_EPOCHS]")
    opt("--batch_size", type=int, help="[env PEFT_BATCH_SIZE]")
    opt("--grad_accum", type=int, help="[env PEFT_GRAD_ACCUM]")
    opt("--learning_rate", type=float, help="[env PEFT_LEARNING_RATE]")
    opt("--max_length", type=int, help="[env PEFT_MAX_LENGTH]")
    opt("--max_new_tokens", type=int, help="[env PEFT_MAX_NEW_TOKENS]")
    opt("--num_samples", type=int, help="Sample generations logged at the end [env PEFT_NUM_SAMPLES]")
    opt("--save_steps", type=int, help="[env PEFT_SAVE_STEPS]")
    opt("--logging_steps", type=int, help="[env PEFT_LOGGING_STEPS]")
    opt("--eval_steps", type=int, help="[env PEFT_EVAL_STEPS]")
    opt("--seed", type=int, help="[env PEFT_SEED]")
    opt("--mlflow_tracking_uri", help="MLflow tracking URI [env PEFT_MLFLOW_TRACKING_URI]")
    # LoRA / quantization knobs (used by qlora.py; ignored by other methods)
    opt("--r", type=int, help="LoRA rank [env PEFT_R]")
    opt("--lora_alpha", type=int, help="LoRA alpha [env PEFT_LORA_ALPHA]")
    opt("--lora_dropout", type=float, help="[env PEFT_LORA_DROPOUT]")
    opt("--target_modules", help="comma-separated or 'all-linear' [env PEFT_TARGET_MODULES]")
    opt("--quant_type", choices=["nf4", "fp4"], help="4-bit quant type [env PEFT_QUANT_TYPE]")
    opt("--no_double_quant", action="store_true", help="disable bnb double quant [env PEFT_DOUBLE_QUANT=false]")
    opt("--no_gc", action="store_true", help="disable gradient checkpointing [env PEFT_GRADIENT_CHECKPOINTING=false]")


# --------------------------------------------------------------------------- #
# RunConfig: CLI > PEFT_* env > default
# --------------------------------------------------------------------------- #
def _cast_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


def _pick(cli_val: Any, env_key: str, default: Any, cast: Callable = str) -> Any:
    if cli_val is not None:
        return cli_val
    if env_key in os.environ and os.environ[env_key] != "":
        raw = os.environ[env_key]
        if cast is bool:
            return _cast_bool(raw)
        try:
            return cast(raw)
        except ValueError:
            raise SystemExit(f"invalid value for {env_key}: {raw!r}")
    return default


@dataclasses.dataclass
class RunConfig:
    method: str = "qlora"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_subset: Optional[str] = None
    output_dir: str = "output"
    run_name: Optional[str] = None
    num_train: int = 2000
    num_eval: int = 128
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 2
    learning_rate: float = 2e-4
    max_length: int = 512
    max_new_tokens: int = 64
    num_samples: int = 5
    save_steps: int = 25
    logging_steps: int = 5
    eval_steps: int = 50
    seed: int = 42
    mlflow_tracking_uri: str = MLFLOW_DEFAULT_URI
    # LoRA / quantization
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    quant_type: str = "nf4"
    double_quant: bool = True
    gradient_checkpointing: bool = True
    # tuning plumbing (set by tune.py, no CLI/env)
    study_name: Optional[str] = None
    prune_steps: int = 10

    # (attr, env key, cast) — None default handled per-field
    _FIELD_MAP = [
        ("method", "PEFT_METHOD", str),
        ("model", "PEFT_MODEL", str),
        ("dataset_name", "PEFT_DATASET_NAME", str),
        ("dataset_subset", "PEFT_DATASET_SUBSET", str),
        ("output_dir", "PEFT_OUTPUT_DIR", str),
        ("run_name", "PEFT_RUN_NAME", str),
        ("num_train", "PEFT_NUM_TRAIN", int),
        ("num_eval", "PEFT_NUM_EVAL", int),
        ("epochs", "PEFT_EPOCHS", int),
        ("batch_size", "PEFT_BATCH_SIZE", int),
        ("grad_accum", "PEFT_GRAD_ACCUM", int),
        ("learning_rate", "PEFT_LEARNING_RATE", float),
        ("max_length", "PEFT_MAX_LENGTH", int),
        ("max_new_tokens", "PEFT_MAX_NEW_TOKENS", int),
        ("num_samples", "PEFT_NUM_SAMPLES", int),
        ("save_steps", "PEFT_SAVE_STEPS", int),
        ("logging_steps", "PEFT_LOGGING_STEPS", int),
        ("eval_steps", "PEFT_EVAL_STEPS", int),
        ("seed", "PEFT_SEED", int),
        ("mlflow_tracking_uri", "PEFT_MLFLOW_TRACKING_URI", str),
        ("r", "PEFT_R", int),
        ("lora_alpha", "PEFT_LORA_ALPHA", int),
        ("lora_dropout", "PEFT_LORA_DROPOUT", float),
        ("target_modules", "PEFT_TARGET_MODULES", str),
        ("quant_type", "PEFT_QUANT_TYPE", str),
        ("double_quant", "PEFT_DOUBLE_QUANT", bool),
        ("gradient_checkpointing", "PEFT_GRADIENT_CHECKPOINTING", bool),
    ]

    @classmethod
    def from_args(cls, args: argparse.Namespace, method: str, **overrides: Any) -> "RunConfig":
        cfg = cls(method=method)
        for attr, env_key, cast in cls._FIELD_MAP:
            cli_val = getattr(args, attr, None) if args else None
            if attr in overrides:
                cfg = dataclasses.replace(cfg, **{attr: overrides[attr]})
                continue
            if attr == "method":
                continue
            cfg = dataclasses.replace(cfg, **{attr: _pick(cli_val, env_key, getattr(cfg, attr), cast)})
        # store-true negation flags: only override when explicitly passed
        if args is not None:
            if getattr(args, "no_gc", False):
                cfg.gradient_checkpointing = False
            if getattr(args, "no_double_quant", False):
                cfg.double_quant = False
        return cfg

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def resolve(path: str, project_root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_root, path)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_processed(
    dataset_name: str,
    subset: Optional[str],
    data_dir: str,
    num_train: Optional[int],
    num_eval: Optional[int],
    seed: int,
) -> tuple[Dataset, Optional[Dataset]]:
    """Load (and cache processed parquet for) the experiment dataset.

    Accepts `prompt`/`response` columns, or the common alpaca-style
    `instruction`/`input`/`output` columns (normalized to prompt/response).
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(
        data_dir, f"processed_{dataset_name.split('/')[-1]}_{subset or 'default'}.parquet"
    )
    if os.path.exists(cache_path):
        print(f"[data] loading cached {cache_path}")
        ds = Dataset.from_parquet(cache_path)
    else:
        print(f"[data] loading {dataset_name} subset={subset} (first run downloads it)")
        ds = (
            load_dataset(dataset_name, subset, split="train")
            if subset
            else load_dataset(dataset_name, split="train")
        )
        ds = ds.shuffle(seed=seed)
        if num_train:
            ds = ds.select(range(min(num_train, len(ds))))
        cols = set(ds.column_names)
        if {"prompt", "response"} <= cols:
            ds = ds.map(lambda r: {"prompt": r["prompt"].strip(), "response": r["response"].strip()}, desc="normalize")
        elif {"instruction", "output"} <= cols:
            ds = ds.map(
                lambda r: {
                    "prompt": (r["instruction"] + ("\n" + r["input"] if r.get("input") else "")).strip(),
                    "response": r["output"].strip(),
                },
                desc="normalize",
            )
        else:
            raise ValueError(f"Unsupported dataset columns: {ds.column_names}")
        ds.to_parquet(cache_path)
        print(f"[data] cached {cache_path}")

    train_ds = ds
    eval_ds = ds.select(range(min(num_eval, len(ds)))) if num_eval else None
    return train_ds, eval_ds


# --------------------------------------------------------------------------- #
# Collator: chat-template formatting, loss masked to the response
# --------------------------------------------------------------------------- #
class ChatCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        input_ids, labels = [], []
        for f in features:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            full_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f["prompt"]}, {"role": "assistant", "content": f["response"]}],
                tokenize=True,
                return_dict=False,
            )
            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
            resp_len = max(0, len(full_ids) - len(prompt_ids))
            lbl = [-100] * (len(full_ids) - resp_len) + full_ids[-resp_len:] if resp_len else [-100] * len(full_ids)
            input_ids.append(full_ids)
            labels.append(lbl)

        max_len = max(len(x) for x in input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        batch_input_ids, batch_labels = [], []
        for ids, lbl in zip(input_ids, labels):
            pad = max_len - len(ids)
            batch_input_ids.append([pad_id] * pad + ids)
            batch_labels.append([-100] * pad + lbl)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Sample generation (qualitative check, logged to MLflow + local dir)
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def generate_samples(model, tokenizer, eval_ds, max_new_tokens: int, n: int):
    """Greedy completions for the first n eval prompts."""
    model.eval()
    device = next(model.parameters()).device
    samples = []
    for i in range(min(n, len(eval_ds))):
        row = eval_ds[i]
        prompt_out = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_out["input_ids"].to(device)
        out_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        samples.append({"prompt": row["prompt"], "reference": row["response"], "completion": text})
    model.train()
    return samples


def log_samples(samples, output_dir: str) -> None:
    """Log samples as MLflow artifact + local text file."""
    local_path = os.path.join(output_dir, "samples.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(local_path, "w") as fh:
        for i, s in enumerate(samples):
            fh.write(f"=== Sample {i + 1} ===\n")
            fh.write(f"prompt:     {s['prompt']}\n")
            fh.write(f"reference:  {s['reference']}\n")
            fh.write(f"completion: {s['completion']}\n\n")
    try:
        if mlflow.active_run() is None:
            print("[mlflow] no active run; skipping artifact upload")
            return
        mlflow.log_artifact(local_path)
    except Exception as e:  # tracking server down shouldn't kill the run
        print(f"[mlflow] could not log samples artifact: {e}")


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
class MlflowCallback(TrainerCallback):
    """Owns the MLflow run: logs params once, metrics on every log step,
    peak memory and runtime at the end. The run stays open after training so
    callers can log post-training artifacts before cb.end_run()."""

    def __init__(self, method: str, extra_params: dict):
        self.method = method
        self.extra_params = extra_params
        self.experiment = os.environ.get("PEFT_MLFLOW_EXPERIMENT", self.experiment)
        self._run = None
        self._t0 = None
        self._peak_mem = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        import platform

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        self._run = mlflow.start_run(run_name=self.run_name)
        self._t0 = time.time()
        mlflow.log_params(
            {
                **self.extra_params,
                "method": self.method,
                "model": self.model_name,
                "dataset": self.dataset_desc,
                "epochs": args.num_train_epochs,
                "batch_size": args.per_device_train_batch_size,
                "grad_accum": args.gradient_accumulation_steps,
                "lr": args.learning_rate,
                "max_length": self.max_length,
                "seed": args.seed,
                "device": "mps",
                "torch": torch.__version__,
                "platform": f"{platform.system()} {platform.machine()}",
            }
        )
        print(f"[mlflow] run started: {self._run.info.run_id}  (ui: {self.tracking_uri})")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.backends.mps.is_available():
            try:
                self._peak_mem = max(self._peak_mem, torch.mps.current_allocated_memory())
            except Exception:
                pass
        if self._run is not None and logs:
            mlflow.log_metrics(logs, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._run is not None and metrics:
            mlflow.log_metrics(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        try:
            mlflow.log_metrics(
                {
                    "peak_mps_mem_mb": round(self._peak_mem / 1e6, 1),
                    "runtime_s": round(time.time() - (self._t0 or time.time()), 1),
                },
                step=state.global_step,
            )
        except Exception as e:
            print(f"[mlflow] mem/runtime logging failed: {e}")

    def end_run(self):
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    # --- filled in by the caller before train() ------------------------------
    tracking_uri: str = MLFLOW_DEFAULT_URI
    experiment: str = "peft-experiments"
    run_name: str = "run"
    model_name: str = ""
    dataset_desc: str = ""
    max_length: int = 0


class TrialPruneCallback(TrainerCallback):
    """Reports a metric to an Optuna trial every `prune_steps`; stops training
    when the trial's pruner says to. `metric` must appear in the trainer's log
    dict (e.g. 'loss') or in eval metrics (e.g. 'eval_loss')."""

    def __init__(self, trial, metric: str, prune_steps: int):
        self.trial = trial
        self.metric = metric
        self.prune_steps = prune_steps
        self.pruned = False
        self.last_value = None

    def _maybe_prune(self, state):
        if self.pruned or state.global_step % self.prune_steps:
            return
        value = self.last_value
        if value is None:
            return
        self.trial.report(value, step=state.global_step)
        if self.trial.should_prune():
            self.pruned = True
            print(f"[optuna] trial {self.trial.number} pruned at step {state.global_step} (value={value:.4f})")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.metric in logs:
            self.last_value = float(logs[self.metric])
            self._maybe_prune(state)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and self.metric in metrics:
            self.last_value = float(metrics[self.metric])
            self._maybe_prune(state)
        return control


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("MPS not available — this setup targets Apple Silicon.")


def release_resources():
    """Free MPS memory between trials."""
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
