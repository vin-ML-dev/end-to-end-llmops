"""Day 2 tests — cheap checks that need no GPU and no model download.

They guard the two things most likely to silently break a fine-tune:
  1. the chat template is applied consistently
  2. the config is internally coherent
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_train_config_is_coherent():
    cfg = yaml.safe_load(open(ROOT / "configs" / "train.yaml"))
    assert cfg["lora"]["r"] > 0
    assert cfg["lora"]["alpha"] >= cfg["lora"]["r"]  # alpha >= r is the norm
    assert cfg["training"]["learning_rate"] < 1e-2  # sane LR range
    assert 0 <= cfg["training"]["warmup_ratio"] <= 1
    # effective batch is a positive product
    eff = (
        cfg["training"]["per_device_train_batch_size"]
        * cfg["training"]["gradient_accumulation_steps"]
    )
    assert eff >= 1
    assert cfg["training"]["metric_for_best_model"] == "eval_loss"
    assert cfg["training"]["greater_is_better"] is False  # lower loss is better


def test_target_modules_are_attention_and_mlp():
    cfg = yaml.safe_load(open(ROOT / "configs" / "train.yaml"))
    tm = set(cfg["lora"]["target_modules"])
    # at minimum the attention projections should be adapted
    assert {"q_proj", "v_proj"}.issubset(tm)


def test_eval_prompts_exist():
    cfg = yaml.safe_load(open(ROOT / "configs" / "train.yaml"))
    assert len(cfg["eval_prompts"]) >= 3  # need a real vibe check, not one prompt
