# Decision log

Short records of decisions that shape the project, each backed by a result file.

---

## ADR-001: VLM backbone

- **Date:** YYYY-MM-DD
- **Status:** proposed
- **Context:** Training and serving both have to run on a Kaggle T4 (sm_75, 16 GB, no bf16).
  Qwen3-VL on Turing GPUs with vLLM is not guaranteed, so it was tested before any other work.
- **Evidence:** `results/phase0/*.json`

| Model | vLLM load | vLLM eager needed | Latency (s/img) | Structured JSON | Unsloth train | Peak VRAM train (GiB) |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | | | | | | |
| Qwen/Qwen3-VL-4B-Instruct | | | | | | |
| Qwen/Qwen2.5-VL-3B-Instruct | | | | | | |

- **Decision:** _fill in_
- **Consequences:** _e.g. "serve with --enforce-eager", "use 2B for ablations, 4B for the final run"_
