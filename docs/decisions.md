# Decision log

Short records of decisions that shape the project, each backed by a result file.

---

## ADR-001: VLM backbone

- **Date:** 2026-11-18
- **Status:** accepted
- **Context:** Training and serving both have to run on a Kaggle T4 (sm_75, 16 GB, no bf16).
  Qwen3-VL on Turing GPUs with vLLM is not guaranteed, so it was tested before any other work.
- **Evidence:** `results/phase0/*.json`

### Setup

- 1× Tesla T4, fp16 everywhere. Input: one synthetic receipt (`make_receipt(seed=0)`, 640×668 px).
- Serving session: vLLM 0.34.0, torch 2.14.0, transformers 5.20.2, xgrammar 0.2.8, `gpu_memory_utilization=0.85`,
  `max_model_len=4096`, CUDA graphs on (no `--enforce-eager`).
- Training session: Unsloth 2026.11.6, torch 2.11.0, transformers 5.17.0, TRL 1.15.0. QLoRA 4-bit, r=16,
  batch 1 × grad-accum 2, 10 steps on 8 synthetic receipts, `max_side=1024`.

### vLLM serving

| Model | Loads | Eager needed | Load (s) | Latency s/img: free / structured | Batch 8 (img/s) | Decode (tok/s) | JSON valid: free / structured | Exact match (zero-shot) |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | yes | no | 213 | 1.60 / 2.31 | 2.85 | 53.8 | yes / yes | 4/4 |
| Qwen/Qwen3-VL-4B-Instruct | yes | no | 265 | 3.76 / 4.94 | 1.55 | 21.8 | yes / yes | 3/4 (seller "CẦU GIÃY" vs "CẦU GIẤY") |
| Qwen/Qwen2.5-VL-3B-Instruct | not run | | | | | | | |

The Qwen2.5-VL-3B fallback did not run in vLLM because the primary model passed on the first attempt.

### Unsloth training

| Model | Status | Vision layers trained | Trainable params (M) | Load (s) | VRAM after load / peak (GiB) | s/step | Loss first → last |
|---|---|---|---|---|---|---|---|
| unsloth/Qwen3-VL-2B-Instruct | ok | **yes** | 23.7 | 14.7 | 2.34 / 2.73 | 6.47 | 2.56 → 1.15 |
| unsloth/Qwen2.5-VL-3B-Instruct | ok | no | 29.9 | 38.4 | 4.12 / 4.12 | 9.62 | 4.67 → 1.55 |
| unsloth/Qwen3-VL-4B-Instruct | not run | | | | | | |

### Limits of this evidence

- One clean synthetic image. Exact match only shows that the pipeline works end to end. It says nothing about quality on real phone photos.
- The vLLM "GPU memory" (~11 GB) is the memory pre-allocated by `gpu_memory_utilization=0.85`, not the size of the model.
- Training VRAM is `torch.cuda.max_memory_allocated` on a 640×668 image. Real receipts at `max_side=1024` give
  more visual tokens, so expect a higher peak.
- The two training runs are not directly comparable: the 2B run also trained the vision layers, the 3B run did not.
- The structured-output latency comes from one request, which probably includes compiling the JSON grammar. Measure it again
  with a warm cache before concluding that structured output is slower.
- Load times include downloading the weights for the first time.

### Decision

- **Qwen3-VL-2B-Instruct** is the main backbone: LoRA fine-tuning, all Phase 3 ablations, and the default served model.
  It works without workarounds in both stacks, it is the fastest, and it uses less than 3 GiB to train.
- **Qwen3-VL-4B-Instruct** is a second zero-shot baseline in Phase 2. Fine-tuning it is optional and needs an Unsloth
  smoke test first.
- **Qwen2.5-VL-3B-Instruct** is kept as a fallback only. It trains, but it was never tested in vLLM.

### Consequences

- Serve with `dtype=half` and CUDA graphs on. `--enforce-eager` is not needed.
- Always use fp16 on T4 (no native bf16).
- Training and serving stay in two separate environments, and their transformers versions differ (5.17.0 vs 5.20.2).
  As soon as Phase 3 produces its first adapter, check that it loads in the vLLM session.
  If vision layers are fine-tuned, plan to merge the LoRA into the weights before serving. vLLM's LoRA support for
  multimodal models may not cover the vision tower, so check this for vLLM 0.34.
- Free-form output from 2B sometimes comes wrapped in ```` ```json ```` fences. `parse_json_output` handles this, but JSON validity
  should be reported for both modes.
