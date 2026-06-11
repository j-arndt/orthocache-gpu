"""Perplexity evaluation on WikiText-2.

Measures perplexity using a sliding-window approach with configurable stride.
The evaluation uses the ``datasets`` library to load WikiText-2 and handles
long contexts properly with chunked processing.

Usage (standalone):
    python benchmarks/quality/tasks/perplexity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Ensure the OrthoCache source is importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    max_length: int = 1024,
    stride: int = 512,
    max_samples: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """Compute perplexity on WikiText-2 test set using sliding windows.

    Args:
        model: A HuggingFace causal LM (e.g. ``LlamaForCausalLM``).
        tokenizer: Matching tokenizer.
        max_length: Maximum sequence length per window.
        stride: Stride for the sliding window.
        max_samples: If set, truncate the dataset to this many tokens.
        device: Device to run on.  If ``None``, inferred from model.

    Returns:
        Dict with keys ``"perplexity"``, ``"avg_loss"``, ``"num_tokens"``.
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    # Load WikiText-2 test split
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text into a single string, filter blanks
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    # Tokenize the full text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids  # (1, total_tokens)

    if max_samples is not None:
        input_ids = input_ids[:, :max_samples]

    total_len = input_ids.size(1)
    print(f"  [perplexity] Total tokens: {total_len}, "
          f"max_length={max_length}, stride={stride}")

    nlls: list[float] = []
    num_tokens_scored = 0

    model.eval()
    with torch.no_grad():
        prev_end = 0
        for begin in range(0, total_len, stride):
            end = min(begin + max_length, total_len)
            target_len = end - prev_end  # number of new tokens to score

            chunk_ids = input_ids[:, begin:end].to(device)

            outputs = model(chunk_ids, labels=chunk_ids)
            # outputs.loss is average cross-entropy over the chunk
            # We need to weight by target_len
            neg_log_likelihood = outputs.loss * target_len

            nlls.append(neg_log_likelihood.item())
            num_tokens_scored += target_len

            prev_end = end
            if end >= total_len:
                break

    avg_nll = sum(nlls) / num_tokens_scored
    ppl = float(torch.exp(torch.tensor(avg_nll)).item())

    print(f"  [perplexity] Perplexity: {ppl:.2f}  "
          f"(avg_loss={avg_nll:.4f}, tokens={num_tokens_scored})")

    return {
        "perplexity": ppl,
        "avg_loss": avg_nll,
        "num_tokens": num_tokens_scored,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    result = evaluate_perplexity(model, tokenizer, max_length=1024, stride=512)
    print(f"\nResult: {result}")
