"""OrthoCache GPU — Quality Benchmark Harness.

Measures actual task-accuracy degradation from KV-cache eviction across
multiple eviction rates.  Hooks OrthoCache's spectral analysis into real
HuggingFace model attention layers and runs downstream evaluations.

Supported tasks:
    * ``perplexity`` — WikiText-2 perplexity
    * ``mmlu``       — 5-shot MMLU (subset of subjects)
    * ``hellaswag``  — HellaSwag common-sense reasoning

Usage:
    python benchmarks/quality/harness.py
    python benchmarks/quality/harness.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --tasks perplexity,mmlu --eviction-rates 0.0,0.25,0.50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Ensure the OrthoCache source is importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Also add the quality benchmarks directory so task imports work
_QUALITY_DIR = Path(__file__).resolve().parent
if str(_QUALITY_DIR) not in sys.path:
    sys.path.insert(0, str(_QUALITY_DIR))

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch._dynamo
torch._dynamo.config.suppress_errors = True


# ---------------------------------------------------------------------------
# Device detection (follows existing benchmark pattern)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_metadata(device: torch.device) -> dict:
    """Collect GPU hardware metadata for result context."""
    if device.type != "cuda":
        return {"device": "cpu"}
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_mem / (1024 ** 3), 2),
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
    }


# ---------------------------------------------------------------------------
# Task dispatcher
# ---------------------------------------------------------------------------

AVAILABLE_TASKS = ["perplexity", "mmlu", "hellaswag"]

DEFAULT_EVICTION_RATES = [0.0, 0.25, 0.50, 0.75]

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def run_task(
    task_name: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    **kwargs,
) -> dict:
    """Dispatch to the appropriate task evaluator.

    Returns:
        A dict with task-specific metrics (perplexity, accuracy, etc.).
    """
    if task_name == "perplexity":
        from tasks.perplexity import evaluate_perplexity
        return evaluate_perplexity(
            model, tokenizer,
            max_length=kwargs.get("max_length", 1024),
            stride=kwargs.get("stride", 512),
            max_samples=kwargs.get("max_ppl_tokens", None),
            device=device,
        )
    elif task_name == "mmlu":
        from tasks.mmlu import evaluate_mmlu
        return evaluate_mmlu(
            model, tokenizer,
            subjects=kwargs.get("mmlu_subjects", None),
            num_shots=kwargs.get("num_shots", 5),
            max_questions=kwargs.get("max_questions", None),
            device=device,
        )
    elif task_name == "hellaswag":
        from tasks.hellaswag import evaluate_hellaswag
        return evaluate_hellaswag(
            model, tokenizer,
            max_examples=kwargs.get("max_hellaswag", 200),
            device=device,
        )
    else:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {AVAILABLE_TASKS}")


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_quality_harness(args: argparse.Namespace) -> list[dict]:
    """Run the full quality benchmark sweep.

    For each (task, eviction_rate):
      1. Patch model with OrthoCache at the given eviction rate
      2. Run the evaluation
      3. Record accuracy/perplexity + eviction stats
      4. Unpatch model

    Returns:
        List of result dicts, one per (task, eviction_rate) pair.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from attention_hook import (
        patch_model_attention,
        unpatch_model_attention,
        get_eviction_tracker,
    )

    device = get_device()
    gpu_meta = get_gpu_metadata(device)

    # Output directory
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse task list and eviction rates
    tasks = [t.strip() for t in args.tasks.split(",")]
    eviction_rates = [float(r) for r in args.eviction_rates.split(",")]

    print("=" * 74)
    print("  OrthoCache GPU — Quality Benchmark Harness")
    print("=" * 74)
    print(f"  Model          : {args.model}")
    print(f"  Tasks          : {tasks}")
    print(f"  Eviction rates : {eviction_rates}")
    print(f"  Device         : {device}")
    if device.type == "cuda":
        print(f"  GPU            : {gpu_meta.get('gpu_name', 'N/A')}")
        print(f"  VRAM           : {gpu_meta.get('total_memory_gb', 'N/A')} GB")
    print(f"  Block size     : {args.block_size}")
    print(f"  Zeta max       : {args.zeta_max}")
    print()

    # Load model and tokenizer
    print(f"Loading model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print()

    all_results: list[dict] = []

    for task_name in tasks:
        print(f"\n{'='*74}")
        print(f"  Task: {task_name}")
        print(f"{'='*74}")

        for rate in eviction_rates:
            print(f"\n--- eviction_rate = {rate:.0%} ---")

            # Patch or skip
            if rate > 0.0:
                wrapper = patch_model_attention(
                    model,
                    eviction_rate=rate,
                    zeta_max=args.zeta_max,
                    block_size=args.block_size,
                )
                wrapper.tracker.reset()
                print(f"  Patched model (eviction={rate:.0%}, "
                      f"zeta_max={args.zeta_max}, block_size={args.block_size})")
            else:
                print("  Baseline (no eviction)")

            # Run the task
            t0 = time.perf_counter()
            try:
                task_kwargs = {
                    "max_length": args.max_length,
                    "stride": args.stride,
                    "max_ppl_tokens": args.max_ppl_tokens,
                    "max_questions": args.max_questions,
                    "max_hellaswag": args.max_hellaswag,
                }
                metrics = run_task(task_name, model, tokenizer, device, **task_kwargs)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                metrics = {"error": str(e)}
            elapsed = time.perf_counter() - t0

            # Collect eviction stats
            eviction_stats = {}
            if rate > 0.0:
                tracker = get_eviction_tracker(model)
                if tracker is not None:
                    eviction_stats = tracker.summary()
                unpatch_model_attention(model)

            result = {
                "task": task_name,
                "eviction_rate": rate,
                "metrics": metrics,
                "eviction_stats": eviction_stats,
                "elapsed_seconds": round(elapsed, 2),
                "model": args.model,
                "block_size": args.block_size,
                "zeta_max": args.zeta_max,
                **gpu_meta,
            }
            all_results.append(result)

            # Clear CUDA cache between runs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Output: JSON
    # ------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"quality_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results written to {json_path.resolve()}")

    # Also write a "latest" symlink/copy
    latest_path = output_dir / "quality_results_latest.json"
    with open(latest_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Output: Summary table
    # ------------------------------------------------------------------
    _print_summary_table(all_results, tasks, eviction_rates)

    return all_results


def _print_summary_table(
    results: list[dict],
    tasks: list[str],
    eviction_rates: list[float],
) -> None:
    """Print a formatted summary table to stdout."""
    print("\n")
    print("=" * 90)
    print("  QUALITY BENCHMARK RESULTS")
    print("=" * 90)

    for task in tasks:
        task_results = [r for r in results if r["task"] == task]
        if not task_results:
            continue

        print(f"\n  Task: {task}")
        print(f"  {'-'*80}")

        if task == "perplexity":
            print(f"  {'Eviction Rate':>15} {'Perplexity':>12} {'Avg Loss':>10} "
                  f"{'Tokens':>8} {'Time (s)':>10} {'Delta':>8}")
            print(f"  {'-'*15:>15} {'-'*12:>12} {'-'*10:>10} "
                  f"{'-'*8:>8} {'-'*10:>10} {'-'*8:>8}")

            baseline_ppl = None
            for r in task_results:
                m = r["metrics"]
                if "error" in m:
                    print(f"  {r['eviction_rate']:>14.0%}  ERROR: {m['error']}")
                    continue
                ppl = m["perplexity"]
                if baseline_ppl is None:
                    baseline_ppl = ppl
                delta = ppl - baseline_ppl if baseline_ppl else 0.0
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                print(f"  {r['eviction_rate']:>14.0%} {ppl:>12.2f} "
                      f"{m['avg_loss']:>10.4f} {m['num_tokens']:>8} "
                      f"{r['elapsed_seconds']:>10.1f} {delta_str:>8}")

        elif task in ("mmlu", "hellaswag"):
            metric_key = "overall_accuracy" if task == "mmlu" else "accuracy"
            print(f"  {'Eviction Rate':>15} {'Accuracy':>10} "
                  f"{'Correct':>8} {'Total':>7} {'Time (s)':>10} {'Delta':>8}")
            print(f"  {'-'*15:>15} {'-'*10:>10} "
                  f"{'-'*8:>8} {'-'*7:>7} {'-'*10:>10} {'-'*8:>8}")

            baseline_acc = None
            for r in task_results:
                m = r["metrics"]
                if "error" in m:
                    print(f"  {r['eviction_rate']:>14.0%}  ERROR: {m['error']}")
                    continue
                acc = m[metric_key]
                correct_key = "correct" if "correct" in m else "total_correct"
                total_key = "total" if "total" in m else "total_count"
                if baseline_acc is None:
                    baseline_acc = acc
                delta = acc - baseline_acc if baseline_acc else 0.0
                delta_str = f"{delta:+.1%}"
                print(f"  {r['eviction_rate']:>14.0%} {acc:>9.1%} "
                      f"{m[correct_key]:>8} {m[total_key]:>7} "
                      f"{r['elapsed_seconds']:>10.1f} {delta_str:>8}")

    print(f"\n{'='*90}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OrthoCache Quality Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tasks", type=str, default="perplexity,mmlu,hellaswag",
        help="Comma-separated list of tasks (default: perplexity,mmlu,hellaswag)",
    )
    parser.add_argument(
        "--eviction-rates", type=str, default="0.0,0.25,0.50,0.75",
        help="Comma-separated eviction rates (default: 0.0,0.25,0.50,0.75)",
    )
    parser.add_argument(
        "--block-size", type=int, default=64,
        help="KV-cache eviction block size (default: 64)",
    )
    parser.add_argument(
        "--zeta-max", type=float, default=5.0,
        help="Maximum spectral decay ratio threshold (default: 5.0)",
    )
    # Perplexity options
    parser.add_argument(
        "--max-length", type=int, default=1024,
        help="Max sequence length for perplexity sliding window (default: 1024)",
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="Stride for perplexity sliding window (default: 512)",
    )
    parser.add_argument(
        "--max-ppl-tokens", type=int, default=None,
        help="Max tokens for perplexity evaluation (default: all)",
    )
    # MMLU options
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Max questions per MMLU subject (default: all)",
    )
    # HellaSwag options
    parser.add_argument(
        "--max-hellaswag", type=int, default=200,
        help="Max HellaSwag examples (default: 200)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_quality_harness(args)
