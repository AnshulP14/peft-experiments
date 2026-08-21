#!/usr/bin/env python3
"""Meta hyperparameter tuning layer on top of the PEFT methods.

Drives an Optuna study (TPE sampler + MedianPruner) where each trial runs one
full method job (QLoRA by default) through the same `run_training()` the CLI
uses. Every trial is an MLflow run with full metrics; the study itself is
persisted to SQLite so an overnight run can be killed and resumed.

    # overnight search: 150 trials or 12h, whichever comes first
    python tune.py --n_trials 150 --timeout 12h

    # resume a killed study (same name) with more budget
    python tune.py --n_trials 150 --timeout 12h --study myrun

Training hyperparameters are NOT set here — steer the whole study via the
PEFT_* env layer (dataset, model, max_length, ...), e.g.:

    PEFT_MODEL=Qwen/Qwen2.5-1.5B-Instruct PEFT_NUM_TRAIN=1000 \
    python tune.py --n_trials 100 --timeout 10h
"""

import argparse
import csv
import datetime as dt
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "methods"))

import optuna  # noqa: E402
import mlflow  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402

from common import RunConfig, get_device, release_resources, resolve, write_json  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_timeout(s: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s)
    if not m:
        raise SystemExit(f"bad --timeout: {s!r} (use e.g. 30m, 12h, 3600s)")
    val, unit = float(m.group(1)), m.group(2) or "s"
    return val * {"s": 1, "m": 60, "h": 3600}[unit]


def _dur_s(d):
    """Trial.duration is a timedelta in Optuna >=4 (a float in older versions)."""
    if not d:
        return None
    return d.total_seconds() if hasattr(d, "total_seconds") else float(d)


def _int_choices(env_key: str, default: list):
    """Search-space choices overridable via env, e.g. PEFT_TUNE_NUM_TRAIN=256,512."""
    raw = os.environ.get(env_key)
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def suggest_params(trial, method: str) -> dict:
    """The search space. Add a method branch here as new methods land.
    Choices are env-overridable: PEFT_TUNE_NUM_TRAIN / PEFT_TUNE_MAX_LENGTH / PEFT_TUNE_EPOCHS."""
    if method == "qlora":
        # Optuna >=4 dropped the `choices=` kwarg from suggest_int/float; use
        # suggest_categorical for an arbitrary discrete grid (exact same space).
        r = trial.suggest_categorical("r", [8, 16, 32, 64])
        nt = _int_choices("PEFT_TUNE_NUM_TRAIN", [500, 1000, 2000])
        ml = _int_choices("PEFT_TUNE_MAX_LENGTH", [256, 512])
        ep = _int_choices("PEFT_TUNE_EPOCHS", [2, 3])
        return {
            "r": r,
            "lora_alpha": 2 * r,  # keep alpha = 2r
            "lora_dropout": trial.suggest_categorical("lora_dropout", [0.0, 0.05]),
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
            "grad_accum": trial.suggest_categorical("grad_accum", [1, 2, 4]),
            "max_length": trial.suggest_categorical("max_length", ml),
            "num_train": trial.suggest_categorical("num_train", nt),
            "epochs": trial.suggest_categorical("epochs", ep),
            "quant_type": "nf4",  # fp4 on MPS untested — add once verified
        }
    raise SystemExit(f"no search space for method {method!r}")


def load_method(method: str):
    import importlib

    return importlib.import_module(f"methods.{method}")


def main():
    p = argparse.ArgumentParser(description="Optuna meta-tuning for PEFT methods on MPS")
    p.add_argument("--method", default=os.environ.get("PEFT_METHOD", "qlora"))
    p.add_argument("--study", default=None, help="study name (default: <method>-<timestamp>)")
    p.add_argument("--fresh", action="store_true", help="start a new study even if one exists in the db")
    p.add_argument("--n_trials", type=int, default=50, help="trials to run (resumable; total budget grows across resumes)")
    p.add_argument("--timeout", default="12h", help="wall-clock budget: 30m / 12h / 3600s")
    p.add_argument("--repeats", type=int, default=3, help="retrain best config N times (seeds +1000i) for error bars")
    p.add_argument("--prune_steps", type=int, default=10, help="report/prune every N training steps")
    p.add_argument("--n_startup_trials", type=int, default=5, help="MedianPruner startup trials")
    p.add_argument("--seed", type=int, default=42, help="sampler seed (training seed comes from PEFT_SEED)")
    args = p.parse_args()

    device = get_device()
    method_mod = load_method(args.method)
    study_name = args.study or f"{args.method}-{dt.datetime.now():%m%d-%H%M%S}"
    study_dir = resolve(os.path.join("output", "tuning", study_name), PROJECT_ROOT)
    os.makedirs(study_dir, exist_ok=True)
    db_path = resolve(os.path.join("output", "tuning", "study.db"), PROJECT_ROOT)
    storage = f"sqlite:///{db_path}"

    study = None
    if args.study:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            print(f"[tune] resuming study {study_name!r}")
        except KeyError:
            pass  # new study with an explicit name — created below
    elif os.path.exists(db_path) and not args.fresh:
        candidates = [optuna.load_study(study_name=s, storage=storage) for s in optuna.study.list_studies(storage=storage)]
        if candidates:
            def last_activity(s):
                ts = [t.datetime_start for t in s.trials if t.datetime_start]
                return max(ts) if ts else dt.datetime.min
            study = max(candidates, key=last_activity)
            study_name = study.study_name
            study_dir = resolve(os.path.join("output", "tuning", study_name), PROJECT_ROOT)
            os.makedirs(study_dir, exist_ok=True)
            print(f"[tune] auto-resuming most recent study {study_name!r} (--fresh to start new, --study to pick)")

    if study is None:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=TPESampler(seed=args.seed),
            pruner=MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=20),
            load_if_exists=True,
        )
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[tune] study={study_name} storage={storage}")
    print(f"[tune] device={device}  completed={n_done}  adding up to {args.n_trials} trials, budget={args.timeout}")
    print(f"[tune] mlflow: http://127.0.0.1:5000  (each trial is a run)")

    base_config = RunConfig.from_args(None, method=args.method)

    t_start = time.time()
    counters = {"since": 0}

    def objective(trial):
        suggested = suggest_params(trial, args.method)
        cfg = RunConfig.from_args(None, method=args.method, **suggested)
        cfg.run_name = f"{study_name}-t{trial.number:03d}"
        cfg.study_name = study_name
        cfg.prune_steps = args.prune_steps
        res = method_mod.run_training(cfg, prune=trial)
        value = res.get("eval_loss") or res.get("train_loss")
        if value is None:
            raise RuntimeError("no objective value produced")
        return value

    # Optuna >=4 study callbacks are 2-arg (study, trial); the old 3-arg
    # (study, trial, stats) signature was removed.
    def heartbeat(study, trial):
        counters["since"] += 1
        if counters["since"] % 5 == 0:
            best = study.best_trial
            elapsed = time.time() - t_start
            done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(
                f"\n[tune] heartbeat: {done} completed, best={best.value:.4f} "
                f"({best.params}), trials/hr≈{done / max(elapsed / 3600, 1e-9):.1f}, elapsed={elapsed / 60:.0f}m",
                flush=True,
            )

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=parse_timeout(args.timeout),
            catch=(Exception,),
            callbacks=[heartbeat],
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[tune] interrupted — study persisted, re-run with --study to resume")

    # ---- summary -------------------------------------------------------------
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rows = [
        {
            "trial": t.number,
            "state": str(t.state).split(".")[-1].lower(),
            "value": t.value,
            "duration_s": round(_dur_s(t.duration), 1),
            **t.params,
        }
        for t in study.trials
    ]
    csv_path = os.path.join(study_dir, "trials.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "study": study_name,
        "method": args.method,
        "n_complete": len(complete),
        "best_value": study.best_trial.value if complete else None,
        "best_params": study.best_trial.params if complete else None,
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(os.path.join(study_dir, "best_params.json"), summary)
    print(f"\n[tune] {len(complete)} complete trials. best eval_loss={summary['best_value']}")
    print(f"[tune] best params: {summary['best_params']}")
    print(f"[tune] wrote {csv_path}")

    # ---- repeat best config for error bars ------------------------------------
    if complete and args.repeats > 1:
        results = []
        for i in range(args.repeats):
            cfg = RunConfig.from_args(None, method=args.method, **study.best_trial.params)
            cfg.seed = base_config.seed + 1000 * i
            cfg.run_name = f"{study_name}-best-repeat{i}"
            cfg.study_name = study_name
            print(f"\n[tune] repeat {i + 1}/{args.repeats} (seed={cfg.seed})")
            res = method_mod.run_training(cfg)
            results.append(res.get("eval_loss"))
            summary[f"repeat_{i}_eval_loss"] = res.get("eval_loss")
        vals = [v for v in results if v is not None]
        if vals:
            mean = sum(vals) / len(vals)
            summary["best_eval_loss_mean"] = round(mean, 5)
            summary["best_eval_loss_std"] = round((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5, 5)
        write_json(os.path.join(study_dir, "best_params.json"), summary)

    # ---- log everything to MLflow as a summary run ----------------------------
    try:
        mlflow.set_tracking_uri(base_config.mlflow_tracking_uri)
        mlflow.set_experiment("peft-experiments")
        with mlflow.start_run(run_name=f"{study_name}-summary") as run:
            mlflow.log_params({"n_trials_requested": args.n_trials, "timeout": args.timeout})
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metrics({k: float(v)})
                elif isinstance(v, dict):
                    for pk, pv in v.items():
                        mlflow.log_params({f"best.{pk}": pv})
            mlflow.log_artifact(csv_path)
            print(f"[tune] summary run: {run.info.run_id}")
    except Exception as e:
        print(f"[tune] mlflow summary logging failed: {e}")

    print("[tune] done.")


if __name__ == "__main__":
    main()
