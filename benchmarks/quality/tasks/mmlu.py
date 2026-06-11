"""Lightweight MMLU evaluation.

Runs 5-shot MMLU on a configurable subset of subjects using the ``cais/mmlu``
dataset from HuggingFace.  For each question the model is prompted with 5
few-shot examples and a test question, then the log-likelihood of each answer
choice (A/B/C/D) is compared to select the prediction.

Usage (standalone):
    python benchmarks/quality/tasks/mmlu.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

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

# Default subjects — small enough for quick iteration, diverse enough to be
# informative.
DEFAULT_SUBJECTS: list[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "computer_security",
]

CHOICES = ["A", "B", "C", "D"]


def _format_example(question: str, choices: list[str], answer: str | None = None) -> str:
    """Format a single MMLU example as a text block."""
    prompt = f"Question: {question}\n"
    for letter, choice in zip(CHOICES, choices):
        prompt += f"  {letter}. {choice}\n"
    if answer is not None:
        prompt += f"Answer: {answer}\n\n"
    else:
        prompt += "Answer:"
    return prompt


def _build_few_shot_prompt(
    train_examples: list[dict],
    test_question: str,
    test_choices: list[str],
    num_shots: int = 5,
) -> str:
    """Build a 5-shot prompt with the test question appended."""
    prompt = "The following are multiple choice questions (with answers).\n\n"
    for ex in train_examples[:num_shots]:
        answer_idx = ex["answer"]
        if isinstance(answer_idx, int):
            answer_letter = CHOICES[answer_idx]
        else:
            answer_letter = str(answer_idx)
        prompt += _format_example(ex["question"], ex["choices"], answer_letter)
    prompt += _format_example(test_question, test_choices, answer=None)
    return prompt


def _score_choices(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
) -> int:
    """Return index of the highest-likelihood answer choice (0–3)."""
    choice_tokens = []
    for c in CHOICES:
        ids = tokenizer.encode(c, add_special_tokens=False)
        # Take the last token in case the tokenizer splits the letter
        choice_tokens.append(ids[-1])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        # Logits at the last position predict the next token
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

    choice_logits = torch.tensor(
        [logits[tid].item() for tid in choice_tokens],
        device="cpu",
    )
    return int(torch.argmax(choice_logits).item())


def evaluate_mmlu(
    model: torch.nn.Module,
    tokenizer,
    subjects: Sequence[str] | None = None,
    num_shots: int = 5,
    max_questions: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """Run MMLU evaluation on selected subjects.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        subjects: List of MMLU subject names.  Defaults to 5 subjects.
        num_shots: Number of few-shot examples (default 5).
        max_questions: Max questions per subject (default: all).
        device: Device for inference.

    Returns:
        Dict with ``"overall_accuracy"`` and ``"per_subject"`` breakdown.
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device
    if subjects is None:
        subjects = DEFAULT_SUBJECTS

    model.eval()

    per_subject: dict[str, dict] = {}
    total_correct = 0
    total_count = 0

    for subject in subjects:
        print(f"  [mmlu] Subject: {subject} ... ", end="", flush=True)

        try:
            ds = load_dataset("cais/mmlu", subject)
        except Exception:
            # Fallback: try with 'all' config and filter
            try:
                ds = load_dataset("cais/mmlu", "all")
                ds = ds.filter(lambda x: x.get("subject") == subject)
            except Exception as e:
                print(f"SKIP ({e})")
                continue

        # Get train and test splits
        if "test" in ds:
            test_data = ds["test"]
        elif "validation" in ds:
            test_data = ds["validation"]
        else:
            print("SKIP (no test/validation split)")
            continue

        if "auxiliary_train" in ds:
            train_data = ds["auxiliary_train"]
        elif "dev" in ds:
            train_data = ds["dev"]
        elif "train" in ds:
            train_data = ds["train"]
        else:
            # Use first few test examples as few-shot (not ideal but workable)
            train_data = test_data

        train_examples = [train_data[i] for i in range(min(num_shots, len(train_data)))]

        questions = list(test_data)
        if max_questions is not None:
            questions = questions[:max_questions]

        correct = 0
        for q in questions:
            prompt = _build_few_shot_prompt(
                train_examples, q["question"], q["choices"], num_shots
            )
            pred = _score_choices(model, tokenizer, prompt, device)
            gold = q["answer"] if isinstance(q["answer"], int) else CHOICES.index(q["answer"])
            if pred == gold:
                correct += 1

        accuracy = correct / len(questions) if questions else 0.0
        per_subject[subject] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(questions),
        }
        total_correct += correct
        total_count += len(questions)

        print(f"{accuracy:.1%} ({correct}/{len(questions)})")

    overall = total_correct / total_count if total_count > 0 else 0.0
    print(f"  [mmlu] Overall: {overall:.1%} ({total_correct}/{total_count})")

    return {
        "overall_accuracy": overall,
        "total_correct": total_correct,
        "total_count": total_count,
        "per_subject": per_subject,
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

    result = evaluate_mmlu(model, tokenizer, max_questions=20)
    print(f"\nResult: {result}")
