"""Phase 0 smoke test: can this VLM be LoRA-fine-tuned with Unsloth on the current GPU?

Runs a tiny QLoRA training job (a few steps on synthetic receipts), then
generates once, and records load time, peak VRAM, step time and loss.
It does NOT measure quality; it only proves the training stack works.

Run in a separate Kaggle session from smoke_vllm.py: vLLM and Unsloth pin
different torch/transformers versions.

Usage:
    python scripts/smoke_unsloth.py --model unsloth/Qwen3-VL-2B-Instruct
    python scripts/smoke_unsloth.py --model unsloth/Qwen2.5-VL-3B-Instruct --steps 20
"""

from __future__ import annotations

# Unsloth must be imported before transformers / trl / peft.
import unsloth  # noqa: F401  isort: skip

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_vlm.gpu import gpu_info  # noqa: E402
from receipt_vlm.imaging import resize_max_side  # noqa: E402
from receipt_vlm.schema import PROMPT, field_exact_match, parse_json_output, to_target_json  # noqa: E402
from receipt_vlm.synthetic import make_receipt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="unsloth/Qwen3-VL-2B-Instruct")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--finetune-vision", action="store_true", help="Also train vision layers.")
    parser.add_argument("--save-adapter", default=None, help="Optional dir to save the LoRA adapter.")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "phase0"))
    return parser.parse_args()


def to_conversation(image, target: dict) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": to_target_json(target)}]},
        ]
    }


def build_trainer(model, tokenizer, dataset, config):
    from trl import SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    collator = UnslothVisionDataCollator(model, tokenizer)
    try:
        return SFTTrainer(model=model, tokenizer=tokenizer, data_collator=collator, train_dataset=dataset, args=config)
    except TypeError:  # newer TRL renamed tokenizer -> processing_class
        return SFTTrainer(
            model=model, processing_class=tokenizer, data_collator=collator, train_dataset=dataset, args=config
        )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"unsloth_{re.sub(r'[^A-Za-z0-9._-]+', '_', args.model)}.json"

    result: dict[str, Any] = {
        "test": "smoke_unsloth",
        "model": args.model,
        "args": vars(args),
        "env": gpu_info(),
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    try:
        import torch
        from trl import SFTConfig
        from unsloth import FastVisionModel, is_bfloat16_supported

        bf16 = is_bfloat16_supported()
        result["precision"] = "bf16" if bf16 else "fp16"

        # Data: synthetic receipts, seed 0 is held out for the generation check.
        samples = [make_receipt(seed=seed) for seed in range(1, args.n_samples + 1)]
        dataset = [to_conversation(resize_max_side(s.image, args.max_side), s.target) for s in samples]
        held_out = make_receipt(seed=0)

        # Load + LoRA.
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model, tokenizer = FastVisionModel.from_pretrained(
            args.model,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=args.finetune_vision,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=args.lora_r,
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
        result["load_seconds"] = round(time.perf_counter() - t0, 1)
        result["vram_after_load_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result["trainable_params_m"] = round(trainable / 1e6, 2)
        print(f"[ok] loaded in {result['load_seconds']}s, trainable {result['trainable_params_m']}M params")

        # Train.
        FastVisionModel.for_training(model)
        config = SFTConfig(
            output_dir=str(out_dir / "unsloth_tmp"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.steps,
            warmup_steps=1,
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            weight_decay=0.01,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=3407,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_length,
        )
        trainer = build_trainer(model, tokenizer, dataset, config)
        t0 = time.perf_counter()
        train_output = trainer.train()
        train_seconds = time.perf_counter() - t0
        losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
        result["train"] = {
            "steps": args.steps,
            "seconds": round(train_seconds, 1),
            "seconds_per_step": round(train_seconds / max(1, args.steps), 2),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "losses": losses,
            "train_loss": train_output.training_loss,
        }
        result["vram_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        train = result["train"]
        print(f"[ok] trained {args.steps} steps, loss {train['loss_first']} -> {train['loss_last']}")

        # Generate once.
        FastVisionModel.for_inference(model)
        image = resize_max_side(held_out.image, args.max_side)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(image, prompt_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True)
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        prediction, error = parse_json_output(text)
        result["generation"] = {
            "seconds": round(time.perf_counter() - t0, 2),
            "raw": text,
            "parsed": prediction,
            "parse_error": error,
            "json_valid": error is None,
            "target": held_out.target,
            "exact_match": field_exact_match(prediction, held_out.target),
        }
        print(f"[ok] generation json_valid={error is None}\n{text}")

        if args.save_adapter:
            model.save_pretrained(args.save_adapter)
            tokenizer.save_pretrained(args.save_adapter)
            result["adapter_dir"] = args.save_adapter

        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        print(f"[FAILED] {exc!r}", file=sys.stderr)
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"[saved] {out_path}")

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
