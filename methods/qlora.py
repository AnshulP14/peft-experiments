"""QLoRA (4-bit NF4 quantization + LoRA adapters) finetuning on MPS, with MLflow logging.

Run from the peft_experiments/ directory:

    # start the observability UI (once)
    make serve

    # single run — CLI flags:
    ../../.venv/bin/python methods/qlora.py --num_train 64 --epochs 1

    # single run — everything via env (CLI > PEFT_* env > default):
    PEFT_NUM_TRAIN=64 PEFT_EPOCHS=1 python methods/qlora.py

Every hyperparameter can also be set with a PEFT_* env var, which is how the
overnight tuning pipeline (tune.py) and shell sweep scripts steer runs.
"""

import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from common import (  # noqa: E402
    ChatCollator,
    MlflowCallback,
    RunConfig,
    TrialPruneCallback,
    add_common_args,
    generate_samples,
    get_device,
    load_processed,
    log_samples,
    release_resources,
    resolve,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_model(config: RunConfig):
    """4-bit base + LoRA adapters."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_compute_dtype=torch.float32,  # MPS: compute in f32 (f16/bf16 unstable on mps)
        bnb_4bit_use_double_quant=config.double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model, quantization_config=bnb_config, attn_implementation="sdpa"
    )
    model.to(get_device())
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = (
        ["all-linear"] if config.target_modules == "all-linear" else config.target_modules.split(",")
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        inference_mode=False,
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model, tokenizer


def run_training(config: RunConfig, prune=None) -> dict:
    """Run one QLoRA training job.

    Args:
        config: resolved RunConfig (CLI > env > default).
        prune: optional Optuna Trial. When set, the run reports
               `eval_loss`/`loss` to the trial every few steps and stops
               early if the pruner kills it.

    Returns:
        dict with at least {"pruned": bool, "eval_loss": float|None, "train_loss": float|None}
    """
    device = get_device()
    run_name = config.run_name or (
        f"{config.method}-{os.path.basename(config.model)}"
        f"-{config.dataset_name.split('/')[-1]}-{dt.datetime.now():%m%d-%H%M%S}"
    )

    # --- data ----------------------------------------------------------------
    train_ds, eval_ds = load_processed(
        config.dataset_name,
        config.dataset_subset,
        resolve("data", PROJECT_ROOT),
        config.num_train,
        config.num_eval,
        config.seed,
    )
    print(f"[data] train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")

    # --- model ---------------------------------------------------------------
    peft_model, tokenizer = build_model(config)
    peft_model.print_trainable_parameters()

    # --- training args ---------------------------------------------------------
    out_dir = resolve(os.path.join(config.output_dir, config.method), PROJECT_ROOT)
    training_args = TrainingArguments(
        output_dir=out_dir,
        run_name=run_name,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        save_steps=config.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=config.eval_steps if eval_ds else 0,
        max_grad_norm=1.0,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=config.seed,
        report_to=[],  # MLflow handled by MlflowCallback
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    cb = MlflowCallback(
        config.method,
        extra_params={
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.target_modules,
            "quantization": f"4bit-{config.quant_type}",
            "double_quant": config.double_quant,
            **(prune and {"trial_number": prune.number, "study": config.study_name or "tuned"} or {}),
        },
    )
    cb.tracking_uri = config.mlflow_tracking_uri
    cb.run_name = run_name
    cb.model_name = config.model
    cb.dataset_desc = (
        f"{config.dataset_name}"
        f"::{config.dataset_subset or 'default'}"
        f" (n_train={len(train_ds)}, n_eval={len(eval_ds) if eval_ds else 0})"
    )
    cb.max_length = config.max_length

    callbacks = [cb]
    prune_cb = (
        TrialPruneCallback(prune, metric="eval_loss" if eval_ds else "loss", prune_steps=config.prune_steps)
        if prune
        else None
    )
    if prune_cb:
        callbacks.append(prune_cb)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=ChatCollator(tokenizer, config.max_length),
        callbacks=callbacks,
    )

    if config.gradient_checkpointing:
        try:
            peft_model.enable_input_require_grads()  # needed for gc + frozen 4-bit base
        except Exception as e:
            print(f"[warn] enable_input_require_grads failed (continuing): {e}")

    output = trainer.train()
    train_loss = getattr(output, "training_loss", None) or getattr(output, "train_loss", None)
    pruned = prune_cb.pruned if prune_cb else False

    if pruned:
        cb.end_run()
        release_resources()
        return {"pruned": True, "eval_loss": prune_cb.last_value, "train_loss": train_loss}

    # --- final eval + qualitative check -----------------------------------------
    eval_loss = None
    if eval_ds is not None:
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        samples = generate_samples(peft_model, tokenizer, eval_ds, config.max_new_tokens, config.num_samples)
        log_samples(samples, out_dir)
        for s in samples:
            print(f"\n  prompt:     {s['prompt'][:70]}")
            print(f"  reference:  {s['reference'][:100]}")
            print(f"  completion: {s['completion'][-150:]}")

    trainer.save_model(os.path.join(out_dir, "final"))
    cb.end_run()
    release_resources()

    print(f"\n[done] adapter saved to {os.path.join(out_dir, 'final')}")
    print(f"[done] view the run: {config.mlflow_tracking_uri}")
    return {"pruned": False, "eval_loss": eval_loss, "train_loss": train_loss}


def main():
    p = argparse.ArgumentParser(
        description="QLoRA finetune on MPS", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p)
    args = p.parse_args()
    config = RunConfig.from_args(args, method="qlora")
    run_training(config)


if __name__ == "__main__":
    main()
