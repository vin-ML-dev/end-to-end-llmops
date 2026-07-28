"""Day 2 — QLoRA fine-tuning with TRL SFTTrainer + MLflow tracking.

Pipeline:
    load base in 4-bit (NF4) -> attach LoRA adapters -> SFT on chat data
    -> early stopping on eval_loss -> save best adapter + sample generations
    -> log everything to MLflow (params, metrics, artifacts, lineage)

Run:
    python -m src.training.train --config configs/train.yaml
    python -m src.training.train --config configs/train.yaml --run-name rank32 --lora-r 32

Design choices worth knowing:
  - Only ADAPTERS are saved (a few MB), not the merged model. Merge happens at
    release (Day 3), never during experiments — keeps runs cheap and comparable.
  - compute dtype is picked at runtime: bf16 on Ampere+ (RTX 30xx/40xx, A100),
    fp16 on older cards (T4). T4 has no bf16 support — using it there crashes.
  - lineage: data version (git tag) and git SHA are logged so every model can be
    traced back to the exact data and code that produced it.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- utils
def load_config(path: str) -> dict:
    return yaml.safe_load(open(ROOT / path))


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "nogit"


def data_version() -> str:
    """The git tag on the current data (e.g. data-v1.0), for lineage."""
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--match", "data-*", "--abbrev=0"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "untagged"


def pick_compute_dtype():
    """bf16 on Ampere+, fp16 otherwise. The #1 hardware gotcha in QLoRA."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    if torch.cuda.is_available():
        return torch.float16, "fp16"
    return torch.float32, "fp32-cpu"


def load_chat_dataset(path: str):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(ROOT / path), split="train")
    return ds


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run-name", default=None, help="override run_name")
    ap.add_argument("--lora-r", type=int, default=None, help="override lora.r")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--lr", type=float, default=None, help="override learning_rate")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r
    if args.epochs is not None:
        cfg["training"]["num_train_epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr

    import mlflow
    import torch
    from datasets import disable_progress_bar
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
    )
    from trl import SFTConfig, SFTTrainer

    disable_progress_bar()
    t = cfg["training"]
    compute_dtype, dtype_name = pick_compute_dtype()
    print(f">> compute dtype: {dtype_name}")

    # ---------------- tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"], trust_remote_code=cfg["model"]["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- 4-bit base model ----------------
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    model.config.use_cache = False
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=t["gradient_checkpointing"]
        )

    # ---------------- LoRA ----------------
    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias=lora["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f">> trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ---------------- data ----------------
    train_ds = load_chat_dataset(cfg["data"]["train_path"])
    val_ds = load_chat_dataset(cfg["data"]["val_path"])

    def formatting_func(batch):
        # apply the model's OWN chat template — must match inference exactly
        return tokenizer.apply_chat_template(batch["messages"], tokenize=False)

    # ---------------- SFT config ----------------
    out_dir = str(ROOT / t["output_dir"] / cfg["run_name"])
    sft = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        save_strategy=t["save_strategy"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        max_seq_length=cfg["data"]["max_seq_length"],
        bf16=(dtype_name == "bf16"),
        fp16=(dtype_name == "fp16"),
        seed=t["seed"],
        report_to=[t["report_to"]],
        run_name=cfg["run_name"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"])],
    )

    # ---------------- train with MLflow ----------------
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name=cfg["run_name"]):
        # lineage + config as params
        mlflow.log_params(
            {
                "base_model": cfg["model"]["base_model"],
                "lora_r": lora["r"],
                "lora_alpha": lora["alpha"],
                "epochs": t["num_train_epochs"],
                "learning_rate": t["learning_rate"],
                "max_seq_length": cfg["data"]["max_seq_length"],
                "compute_dtype": dtype_name,
                "trainable_pct": round(100 * trainable / total, 3),
                "data_version": data_version(),
                "git_sha": git_sha(),
                "seed": t["seed"],
            }
        )

        result = trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics(
            {
                "train_loss": result.training_loss,
                "eval_loss": metrics.get("eval_loss", float("nan")),
            }
        )

        # save best adapter (small — a few MB)
        adapter_dir = Path(out_dir) / "final_adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # qualitative "vibe check": generate on sanity prompts, log as artifact
        samples = generate_samples(trainer.model, tokenizer, cfg.get("eval_prompts", []))
        samples_path = Path(out_dir) / "sample_generations.json"
        samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False))

        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        mlflow.log_artifact(str(samples_path), artifact_path="samples")
        mlflow.log_artifact(str(ROOT / args.config), artifact_path="config")

        print(f"\n>> run '{cfg['run_name']}' done. eval_loss={metrics.get('eval_loss'):.4f}")
        print(f">> adapter: {adapter_dir}")
        print(">> sample generations:")
        for s in samples[:3]:
            print(f"   Q: {s['prompt'][:60]}")
            print(f"   A: {s['response'][:100]}\n")


def generate_samples(model, tokenizer, prompts, max_new_tokens=150):
    import torch

    system = "You are DomainBot, a helpful and precise assistant."
    out = []
    model.eval()
    for p in prompts:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        out.append({"prompt": p, "response": resp.strip()})
    return out


if __name__ == "__main__":
    sys.exit(main())
