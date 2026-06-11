# OrthoCache GPU — Quality Benchmark Suite

Measures **actual task-accuracy degradation** from KV-cache eviction across
multiple eviction rates. Unlike perplexity-only benchmarks, this harness
evaluates downstream tasks (MMLU, HellaSwag) that reveal whether eviction
damages the model's ability to reason and answer questions.

## Architecture

```
benchmarks/quality/
├── harness.py           # CLI entry point — orchestrates the sweep
├── attention_hook.py    # Patches HF model to apply OrthoCache eviction
├── tasks/
│   ├── perplexity.py    # WikiText-2 perplexity (sliding window)
│   ├── mmlu.py          # 5-shot MMLU on selected subjects
│   └── hellaswag.py     # HellaSwag common-sense reasoning
└── README.md            # This file
```

## How it works

1. **Load model** — Any HuggingFace causal LM (default: TinyLlama 1.1B).
2. **For each (task, eviction_rate)**:
   - `patch_model_attention(model, eviction_rate)` hooks OrthoCache into the
     model's forward pass.
   - After each forward pass, the hook intercepts `past_key_values`,
     runs spectral analysis on the K cache, and **zeros out** the
     bottom-ranked blocks — exactly simulating production KV-cache eviction.
   - The downstream task evaluator then measures accuracy/perplexity on the
     degraded model.
   - `unpatch_model_attention(model)` restores baseline.
3. **Output** — JSON results to `benchmarks/results/` + formatted table.

## Quick start

```bash
# Full sweep (all tasks, default eviction rates)
python benchmarks/quality/harness.py

# Fast smoke test (perplexity only, two rates)
python benchmarks/quality/harness.py \
  --tasks perplexity \
  --eviction-rates 0.0,0.50 \
  --max-ppl-tokens 5000

# MMLU + HellaSwag with custom rates
python benchmarks/quality/harness.py \
  --tasks mmlu,hellaswag \
  --eviction-rates 0.0,0.25,0.50,0.75 \
  --max-questions 20 \
  --max-hellaswag 100

# Custom model
python benchmarks/quality/harness.py \
  --model microsoft/phi-2 \
  --tasks perplexity \
  --eviction-rates 0.0,0.50
```

## CLI reference

| Argument             | Default                                  | Description                               |
|----------------------|------------------------------------------|-------------------------------------------|
| `--model`            | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`   | HuggingFace model name                    |
| `--tasks`            | `perplexity,mmlu,hellaswag`             | Comma-separated task list                 |
| `--eviction-rates`   | `0.0,0.25,0.50,0.75`                   | Comma-separated eviction rates            |
| `--block-size`       | `64`                                     | KV-cache eviction block size (tokens)     |
| `--zeta-max`         | `5.0`                                    | Max spectral decay ratio threshold        |
| `--max-length`       | `1024`                                   | Perplexity window size                    |
| `--stride`           | `512`                                    | Perplexity window stride                  |
| `--max-ppl-tokens`   | all                                      | Limit perplexity token count              |
| `--max-questions`    | all                                      | Max MMLU questions per subject            |
| `--max-hellaswag`    | `200`                                    | Max HellaSwag examples                    |

## How the attention hook works

The hook uses a **post-forward interception** strategy:

1. `model.forward` is monkey-patched to wrap the original.
2. After each forward pass, `past_key_values` is extracted from the output.
3. For each layer's `(K, V)` tensor (shape `[batch, heads, seq, dim]`):
   - Reshape K into blocks of `block_size` tokens
   - Compute per-block spectral energy (high-freq vs low-freq ratio = ζ)
   - Rank blocks by ζ (higher = noisier = better eviction candidate)
   - Zero out the top `eviction_rate` fraction of blocks in both K and V
   - Protect the last block (most recent tokens)
4. The modified `past_key_values` is returned to the model.

This is functionally equivalent to what a production OrthoCache deployment
does: analyze the KV cache, identify low-value blocks, and evict them before
the next decode step.

When `block_size=512`, the hook uses OrthoCache's native FWHT-based spectral
analysis. For other block sizes, a lightweight spectral proxy is used.

## Output format

Results are written to `benchmarks/results/quality_results_<timestamp>.json`
and `benchmarks/results/quality_results_latest.json`.

Each entry contains:
```json
{
  "task": "mmlu",
  "eviction_rate": 0.50,
  "metrics": {
    "overall_accuracy": 0.256,
    "total_correct": 128,
    "total_count": 500,
    "per_subject": { ... }
  },
  "eviction_stats": {
    "num_layers": 22,
    "mean_eviction_rate": 0.49,
    "per_layer_eviction_rates": [0.50, ...]
  },
  "elapsed_seconds": 45.2,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU"
}
```

## Dependencies

- `torch` (with CUDA for GPU)
- `transformers`
- `datasets`
- OrthoCache GPU (from `src/orthocache_gpu/`)

## Design decisions

- **Block size 64** (default): OrthoCache's native FWHT uses 512-token
  blocks, but most benchmark prompts are shorter than 512 tokens. Using
  64-token blocks allows meaningful eviction even on short sequences.
  The spectral proxy gives qualitatively similar rankings to the full FWHT.

- **Rate-based eviction**: For benchmarking, we control the exact eviction
  rate rather than letting ζ_max determine it dynamically. This produces
  clean degradation curves. The ζ ranking still determines *which* blocks
  are evicted.

- **Last-block protection**: The most recent block is never evicted,
  since the model needs recent context for coherent generation.
