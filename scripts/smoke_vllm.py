"""Phase 0 smoke test: can this VLM be served by vLLM on the current GPU?

Checks, in order:
  1. The model loads in vLLM with dtype=half (T4 has no bf16).
  2. Free-form generation on one receipt image works; latency is measured.
  3. Structured outputs (JSON schema) work.
  4. A small batch runs, giving a rough throughput number.

Every run writes results/phase0/vllm_<model>[_eager].json, success or failure,
so the backbone decision in docs/decisions.md is backed by evidence.

Usage:
    python scripts/smoke_vllm.py --model Qwen/Qwen3-VL-2B-Instruct
    python scripts/smoke_vllm.py --model Qwen/Qwen2.5-VL-3B-Instruct --enforce-eager
    python scripts/smoke_vllm.py --model Qwen/Qwen3-VL-2B-Instruct --image path/to/receipt.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_vlm.gpu import gpu_info, nvidia_smi_memory  # noqa: E402
from receipt_vlm.imaging import load_image, resize_max_side, to_data_uri  # noqa: E402
from receipt_vlm.schema import JSON_SCHEMA, PROMPT, field_exact_match, parse_json_output  # noqa: E402
from receipt_vlm.synthetic import make_receipt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image", default=None, help="Real receipt image. Default: synthetic receipt.")
    parser.add_argument("--max-side", type=int, default=1280, help="Resize longest image side to this.")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="gpu_memory_utilization")
    parser.add_argument("--repeats", type=int, default=5, help="Sequential requests for latency.")
    parser.add_argument("--batch", type=int, default=8, help="Requests in one batched call for throughput.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs (slower, safer).")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "phase0"))
    return parser.parse_args()


def sampling_params(sampling_cls, max_tokens: int, schema: dict | None = None):
    """Build SamplingParams across vLLM versions (structured_outputs vs guided_decoding)."""
    kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": max_tokens, "seed": 0}
    if schema is None:
        return sampling_cls(**kwargs)
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return sampling_cls(structured_outputs=StructuredOutputsParams(json=schema), **kwargs)
    except (ImportError, TypeError):
        from vllm.sampling_params import GuidedDecodingParams

        return sampling_cls(guided_decoding=GuidedDecodingParams(json=schema), **kwargs)


def build_messages(data_uri: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def evaluate_output(text: str, target: dict | None) -> dict[str, Any]:
    prediction, error = parse_json_output(text)
    report: dict[str, Any] = {"raw": text, "parsed": prediction, "parse_error": error, "json_valid": error is None}
    if target is not None:
        report["exact_match"] = field_exact_match(prediction, target)
    return report


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vllm_{safe_name(args.model)}{'_eager' if args.enforce_eager else ''}.json"

    result: dict[str, Any] = {
        "test": "smoke_vllm",
        "model": args.model,
        "args": vars(args),
        "env": gpu_info(),
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    try:
        # Image + ground truth.
        if args.image:
            image = load_image(args.image)
            target = None
            result["image_source"] = str(args.image)
        else:
            receipt = make_receipt(seed=0)
            image, target = receipt.image, receipt.target
            result["image_source"] = "synthetic(seed=0)"
            result["target"] = target
        image = resize_max_side(image, args.max_side)
        result["image_size"] = list(image.size)
        image.save(out_dir / "smoke_input.jpg", quality=95)
        messages = build_messages(to_data_uri(image))

        # 1. Load.
        from vllm import LLM, SamplingParams

        t0 = time.perf_counter()
        llm = LLM(
            model=args.model,
            dtype="half",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=args.enforce_eager,
            seed=0,
        )
        result["load_seconds"] = round(time.perf_counter() - t0, 1)
        result["gpu_memory_after_load"] = nvidia_smi_memory()
        print(f"[ok] loaded {args.model} in {result['load_seconds']}s")

        free_params = sampling_params(SamplingParams, args.max_tokens)

        # 2. Free-form generation: warm-up + sequential latency.
        llm.chat(messages, free_params, use_tqdm=False)
        latencies, out_tokens, last_text = [], [], ""
        for _ in range(max(1, args.repeats)):
            t0 = time.perf_counter()
            outputs = llm.chat(messages, free_params, use_tqdm=False)
            latencies.append(time.perf_counter() - t0)
            completion = outputs[0].outputs[0]
            out_tokens.append(len(completion.token_ids))
            last_text = completion.text
        result["free_form"] = {
            "latency_s_mean": round(statistics.mean(latencies), 3),
            "latency_s_min": round(min(latencies), 3),
            "latency_s_max": round(max(latencies), 3),
            "output_tokens_mean": round(statistics.mean(out_tokens), 1),
            "decode_tokens_per_s": round(sum(out_tokens) / sum(latencies), 1),
            **evaluate_output(last_text, target),
        }
        print(f"[ok] free-form latency {result['free_form']['latency_s_mean']}s/image")
        print(last_text)

        # 3. Structured outputs (failure here is recorded but does not fail the test).
        try:
            structured = sampling_params(SamplingParams, args.max_tokens, schema=JSON_SCHEMA)
            t0 = time.perf_counter()
            outputs = llm.chat(messages, structured, use_tqdm=False)
            result["structured"] = {
                "latency_s": round(time.perf_counter() - t0, 3),
                **evaluate_output(outputs[0].outputs[0].text, target),
            }
            print(f"[ok] structured output json_valid={result['structured']['json_valid']}")
        except Exception as exc:  # noqa: BLE001 - record and continue
            result["structured"] = {"error": repr(exc), "traceback": traceback.format_exc()}
            print(f"[warn] structured outputs failed: {exc!r}")

        # 4. Batched throughput.
        if args.batch > 1:
            t0 = time.perf_counter()
            outputs = llm.chat([messages] * args.batch, free_params, use_tqdm=False)
            elapsed = time.perf_counter() - t0
            total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
            result["batch"] = {
                "size": args.batch,
                "seconds": round(elapsed, 3),
                "images_per_s": round(args.batch / elapsed, 2),
                "output_tokens_per_s": round(total_tokens / elapsed, 1),
            }
            print(f"[ok] batch of {args.batch}: {result['batch']['images_per_s']} images/s")

        result["gpu_memory_peak_seen"] = nvidia_smi_memory()
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - the whole point is to record failures
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        print(f"[FAILED] {exc!r}", file=sys.stderr)
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {out_path}")

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    raise SystemExit(main())
