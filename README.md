# vietnamese-receipt-vlm

Key-information extraction from mobile-captured **Vietnamese receipts**: an OCR pipeline vs. a zero-shot
vision-language model vs. a **LoRA-fine-tuned VLM**, with error analysis, hallucination checks and
**vLLM** serving benchmarks.

> **Research question:** On Vietnamese receipts photographed with a phone, does a small fine-tuned VLM beat
> a classic OCR pipeline, and what does it cost in latency and compute?

Fields extracted (MC-OCR 2021 schema):

```json
{"seller": "...", "address": "...", "timestamp": "...", "total_cost": "..."}
```

## Status

- [x] **Phase 0** – Environment & risk check (vLLM + Unsloth on Kaggle T4) → ADR-001 in `docs/decisions.md`: Qwen3-VL-2B
- [ ] **Phase 1** – Data (MC-OCR 2021), store-disjoint split, text normalisation, metrics
- [ ] **Phase 2** – Baselines: PaddleOCR + VietOCR (+ rules / + LLM), zero-shot VLM
- [ ] **Phase 3** – LoRA fine-tuning + ablations (resolution, model size, vision layers, data size)
- [ ] **Phase 4** – Error analysis, hallucination cross-check, robustness
- [ ] **Phase 5** – vLLM serving, benchmarks, FastAPI + Gradio demo

## Results

_Filled in from Phase 2 onwards._

## Repository layout

```
src/receipt_vlm/   shared code: schema & prompt, image helpers, synthetic receipts, env info
scripts/           runnable entry points (smoke tests, later: data prep, train, eval, serve)
notebooks/         thin Kaggle notebooks that clone this repo and call scripts/
tests/             CPU-only unit tests
results/           JSON result of every experiment (committed)
docs/decisions.md  decision log
```

## Phase 0: run the smoke tests on Kaggle

The two stacks pin different `torch` versions, so use **two separate Kaggle sessions**.
In both: *Settings → Accelerator → GPU T4 x2*, *Internet → On*.

**Session A – vLLM serving** (`notebooks/00a_smoke_vllm.ipynb`)

```bash
git clone https://github.com/Licht1906/vietnamese-receipt-vlm.git && cd vietnamese-receipt-vlm
pip install -q "vllm>=0.11.0"
pip install -q -e . --no-deps
python scripts/check_env.py
bash scripts/run_phase0_vllm.sh                  # Qwen3-VL-2B → eager → Qwen2.5-VL-3B fallback
PRIMARY=Qwen/Qwen3-VL-4B-Instruct bash scripts/run_phase0_vllm.sh   # optional: also test 4B
```

**Session B – Unsloth training** (`notebooks/00b_smoke_unsloth.ipynb`)

```bash
git clone https://github.com/Licht1906/vietnamese-receipt-vlm.git && cd vietnamese-receipt-vlm
pip install -q --upgrade unsloth unsloth_zoo
pip install -q -e . --no-deps
python scripts/check_env.py
python scripts/smoke_unsloth.py --model unsloth/Qwen3-VL-2B-Instruct
```

Each script writes `results/phase0/<test>_<model>.json` (status, timings, VRAM, raw output, traceback on failure).
Download them, commit them, and fill in ADR-001 in `docs/decisions.md`.

## Local development (CPU)

```bash
pip install -e ".[dev]"
pytest
```

## Data

MC-OCR 2021 (RIVF 2021 challenge). Images are **not** redistributed in this repo; see `scripts/` (Phase 1) for download
and preparation.
