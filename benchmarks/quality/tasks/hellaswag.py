"""HellaSwag common-sense reasoning evaluation.

For each example, computes the log-likelihood of each candidate continuation
and selects the most likely one.  Accuracy is reported over a configurable
number of examples (default 200 for speed).

Usage (standalone):
    python benchmarks/quality/tasks/hellaswag.py
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


def _compute_continuation_nll(
    model: torch.nn.Module,
    tokenizer,
    context: str,
    continuation: str,
    device: torch.device,
) -> float:
    """Compute negative log-likelihood of ``continuation`` given ``context``.

    Returns the average NLL per continuation token.
    """
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)

    if not cont_ids:
        return float("inf")

    full_ids = ctx_ids + cont_ids
    # Truncate if too long
    max_len = getattr(model.config, "max_position_embeddings", 2048)
    if len(full_ids) > max_len:
        # Trim context from the left to keep continuation intact
        trim = len(full_ids) - max_len
        full_ids = full_ids[trim:]
        ctx_len = len(ctx_ids) - trim
        if ctx_len < 0:
            ctx_len = 0
    else:
        ctx_len = len(ctx_ids)

    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # We score the continuation tokens only
    # The logit at position i predicts token at position i+1
    cont_start = ctx_len  # first continuation token position in full_ids
    cont_end = len(full_ids)

    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

    total_nll = 0.0
    num_tokens = 0
    for i in range(cont_start, cont_end):
        if i == 0:
            continue  # No preceding logit for position 0
        token_id = full_ids[i]
        total_nll -= log_probs[i - 1, token_id].item()
        num_tokens += 1

    return total_nll / max(num_tokens, 1)


def _preprocess_hellaswag_text(text: str) -> str:
    """Clean up HellaSwag text formatting quirks."""
    text = text.strip()
    # Remove [header] markers
    if text.startswith("["):
        idx = text.find("]")
        if idx != -1:
            text = text[idx + 1:].strip()
    return text


def evaluate_hellaswag(
    model: torch.nn.Module,
    tokenizer,
    max_examples: int = 200,
    device: torch.device | None = None,
) -> dict:
    """Run HellaSwag common-sense reasoning evaluation.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        max_examples: Number of examples to evaluate (default 200).
        device: Device for inference.

    Returns:
        Dict with ``"accuracy"``, ``"correct"``, ``"total"``.
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    ds = load_dataset("Rowan/hellaswag", split="validation")

    examples = list(ds)[:max_examples]
    correct = 0
    total = 0

    print(f"  [hellaswag] Evaluating {len(examples)} examples ...")

    for idx, ex in enumerate(examples):
        ctx = _preprocess_hellaswag_text(ex["ctx"])
        endings = ex["endings"]
        label = int(ex["label"])

        # Compute NLL for each candidate ending
        nlls = []
        for ending in endings:
            ending_clean = _preprocess_hellaswag_text(ending)
            nll = _compute_continuation_nll(
                model, tokenizer, ctx, ending_clean, device
            )
            nlls.append(nll)

        pred = int(torch.tensor(nlls).argmin().item())
        if pred == label:
            correct += 1
        total += 1

        if (idx + 1) % 50 == 0:
            acc_so_far = correct / total
            print(f"    [{idx + 1}/{len(examples)}] accuracy so far: {acc_so_far:.1%}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"  [hellaswag] Accuracy: {accuracy:.1%} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
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

    result = evaluate_hellaswag(model, tokenizer, max_examples=50)
    print(f"\nResult: {result}")
