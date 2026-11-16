#!/usr/bin/env bash
# Phase 0: decide the VLM backbone by testing vLLM serving on this GPU.
#
#   1. PRIMARY model (default Qwen3-VL-2B) with CUDA graphs
#   2. PRIMARY again with --enforce-eager (rules out CUDA-graph issues)
#   3. FALLBACK model (default Qwen2.5-VL-3B)
#
# Stops at the first success. Every attempt writes a JSON report to results/phase0/.
# Extra arguments are forwarded, e.g.:  bash scripts/run_phase0_vllm.sh --image my_receipt.jpg
#
# Override models:  PRIMARY=Qwen/Qwen3-VL-4B-Instruct bash scripts/run_phase0_vllm.sh

set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PRIMARY="${PRIMARY:-Qwen/Qwen3-VL-2B-Instruct}"
FALLBACK="${FALLBACK:-Qwen/Qwen2.5-VL-3B-Instruct}"

attempt() {
  echo
  echo "=================================================================="
  echo ">>> $*"
  echo "=================================================================="
  python scripts/smoke_vllm.py "$@"
}

if attempt --model "$PRIMARY" "$@"; then
  echo; echo "RESULT: $PRIMARY works with vLLM on this GPU."; exit 0
fi

if attempt --model "$PRIMARY" --enforce-eager "$@"; then
  echo; echo "RESULT: $PRIMARY works only with --enforce-eager (note this in docs/decisions.md)."; exit 0
fi

if attempt --model "$FALLBACK" "$@"; then
  echo; echo "RESULT: $PRIMARY failed; FALLBACK $FALLBACK works. Use the fallback backbone."; exit 0
fi

echo; echo "RESULT: all attempts failed. Read the tracebacks in results/phase0/*.json."
exit 1
