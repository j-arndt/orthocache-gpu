"""OrthoCache Post-Benchmark Target Validator.

Reads JSON benchmark results and validates them against hardware-specific
performance targets defined in a concrete target matrix.  Produces:

* Color-coded terminal table (ANSI with Windows fallback)
* Markdown report → benchmarks/results/target_validation_report.md

Usage
-----
    python benchmarks/target_validation.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Windows console fix + path setup
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Target matrix
# ---------------------------------------------------------------------------

TARGET_MATRIX: list[dict[str, Any]] = [
    {
        "gpu_pattern": "RTX 6000",
        "seq_len": 8192,
        "eviction": 0.50,
        "min_speedup": 1.50,
        "max_error": 0.015,
        "max_violations": 0,
    },
    {
        "gpu_pattern": "RTX 6000",
        "seq_len": 32768,
        "eviction": 0.75,
        "min_speedup": 2.60,
        "max_error": 0.020,
        "max_violations": 0,
    },
    {
        "gpu_pattern": "H100",
        "seq_len": 8192,
        "eviction": 0.50,
        "min_speedup": 0.95,
        "max_error": 0.015,
        "max_violations": 0,
    },
    {
        "gpu_pattern": "H100",
        "seq_len": 65536,
        "eviction": 0.50,
        "min_speedup": 2.10,
        "max_error": 0.020,
        "max_violations": 0,
    },
    {
        "gpu_pattern": "H100",
        "seq_len": 131072,
        "eviction": 0.75,
        "min_speedup": 4.20,
        "max_error": 0.025,
        "max_violations": 0,
    },
]


# ---------------------------------------------------------------------------
# ANSI colour helpers (with Windows fallback)
# ---------------------------------------------------------------------------

def _supports_ansi() -> bool:
    """Check whether the terminal supports ANSI escape codes."""
    if sys.platform == "win32":
        # Enable VT100 processing on Windows 10+
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_ANSI = _supports_ansi()


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _ANSI else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _ANSI else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _ANSI else text


def _cyan(text: str) -> str:
    return f"\033[96m{text}\033[0m" if _ANSI else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _ANSI else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _ANSI else text


# ---------------------------------------------------------------------------
# Result status
# ---------------------------------------------------------------------------

class Status:
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


def _colour_status(status: str) -> str:
    if status == Status.PASS:
        return _green(status)
    elif status == Status.FAIL:
        return _red(status)
    elif status == Status.WARN:
        return _yellow(status)
    else:
        return _dim(status)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    gpu_pattern: str
    seq_len: int
    eviction: float
    min_speedup: float
    max_error: float
    max_violations: int

    actual_speedup: Optional[float] = None
    actual_error: Optional[float] = None
    actual_violations: Optional[int] = None

    speedup_status: str = Status.SKIP
    error_status: str = Status.SKIP
    violation_status: str = Status.SKIP
    overall_status: str = Status.SKIP

    bottleneck_notes: list[str] = field(default_factory=list)

    # Timing breakdown for bottleneck analysis
    spectral_ms: Optional[float] = None
    compact_ms: Optional[float] = None
    attention_ms: Optional[float] = None
    total_ms: Optional[float] = None
    dense_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    """Load JSON from *path*, returning None if the file does not exist."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _detect_gpu_name(profiling_data: list[dict]) -> str:
    """Try to detect the GPU name from profiling results metadata.

    The profiling script records `device` but not the GPU name in each row.
    We rely on torch being importable to query the current GPU at validation
    time, falling back to 'Unknown GPU'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "Unknown GPU"


def _gpu_matches(gpu_name: str, pattern: str) -> bool:
    """Case-insensitive substring match."""
    return pattern.lower() in gpu_name.lower()


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _find_profiling_result(
    data: list[dict],
    seq_len: int,
    eviction_rate: float,
) -> Optional[dict]:
    """Find the profiling entry for (seq_len, eviction_rate).

    Returns the matching compact-mode row, or None.
    """
    for row in data:
        if row.get("seq_len") != seq_len:
            continue
        row_evict = row.get("eviction_rate")
        if row_evict is None:
            continue
        if abs(row_evict - eviction_rate) < 1e-6:
            return row
    return None


def _find_dense_baseline(data: list[dict], seq_len: int) -> Optional[dict]:
    """Find the dense (no eviction) baseline row for *seq_len*."""
    for row in data:
        if row.get("seq_len") == seq_len and row.get("mode") == "dense":
            return row
    return None


def _find_compaction_result(
    data: list[dict],
    seq_len: int,
    eviction_rate: float,
) -> Optional[dict]:
    """Find the compaction benchmark entry for (seq_len, eviction_rate)."""
    for row in data:
        if row.get("seq_len") != seq_len:
            continue
        row_evict = row.get("eviction_rate")
        if row_evict is None:
            continue
        if abs(row_evict - eviction_rate) < 1e-6:
            return row
    return None


def _find_reconstruction_result(
    data: list[dict],
    seq_len: int,
    eviction_rate: float,
) -> Optional[dict]:
    """Find a reconstruction-error entry for (seq_len, eviction_rate)."""
    for row in data:
        if row.get("seq_len") != seq_len:
            continue
        row_evict = row.get("eviction_rate", row.get("eviction"))
        if row_evict is None:
            continue
        if abs(row_evict - eviction_rate) < 1e-6:
            return row
    return None


def _find_spectral_dense_mode(data: list[dict], seq_len: int) -> Optional[dict]:
    """Find the dense_spectral row (spectral overhead) for *seq_len*."""
    for row in data:
        if row.get("seq_len") == seq_len and row.get("mode") == "dense_spectral":
            return row
    return None


# ---------------------------------------------------------------------------
# Bottleneck analysis
# ---------------------------------------------------------------------------

def _analyse_bottleneck(result: ValidationResult) -> list[str]:
    """Identify bottlenecks when speedup is below target."""
    notes: list[str] = []

    if result.actual_speedup is None or result.actual_speedup >= result.min_speedup:
        return notes

    deficit = result.min_speedup - result.actual_speedup
    pct_deficit = (deficit / result.min_speedup) * 100

    notes.append(
        f"Speedup deficit: {result.actual_speedup:.2f}x vs {result.min_speedup:.2f}x "
        f"target ({pct_deficit:.0f}% below)"
    )

    total = result.total_ms
    if total is not None and total > 0:
        # Spectral overhead check
        if result.spectral_ms is not None and (result.spectral_ms / total) > 0.30:
            frac = (result.spectral_ms / total) * 100
            notes.append(
                f"Spectral kernel overhead dominant: {result.spectral_ms:.3f}ms "
                f"= {frac:.0f}% of total pipeline"
            )
            notes.append(
                "  → Recommendation: fuse FWHT + decay-ratio into a single kernel"
            )

        # Compaction overhead check
        if result.compact_ms is not None and (result.compact_ms / total) > 0.30:
            frac = (result.compact_ms / total) * 100
            notes.append(
                f"Compaction overhead dominant: {result.compact_ms:.3f}ms "
                f"= {frac:.0f}% of total pipeline"
            )
            notes.append(
                "  → Recommendation: use warp-cooperative prefix-sum compaction"
            )

        # Memory bandwidth saturation check
        if result.dense_ms is not None and result.attention_ms is not None:
            # If compact attention takes nearly as long as dense, we're
            # bandwidth-bound (not compute-bound as expected)
            if result.attention_ms > 0.85 * result.dense_ms:
                notes.append(
                    f"Memory bandwidth saturated: compact attention "
                    f"({result.attention_ms:.3f}ms) ≈ dense attention "
                    f"({result.dense_ms:.3f}ms)"
                )
                notes.append(
                    "  → Recommendation: increase block_size or use "
                    "FlashAttention-2 for the inner kernel"
                )

    # Sequence-length-specific advice
    if result.seq_len >= 65536 and result.actual_speedup is not None:
        notes.append(
            "  → For very long sequences, ensure multi-stream overlap "
            "of spectral + compaction stages"
        )

    return notes


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_targets(
    gpu_name: str,
    profiling_data: Optional[list[dict]],
    reconstruction_data: Optional[list[dict]],
    compaction_data: Optional[list[dict]],
) -> list[ValidationResult]:
    """Run validation for every row in TARGET_MATRIX."""
    results: list[ValidationResult] = []

    for target in TARGET_MATRIX:
        vr = ValidationResult(
            gpu_pattern=target["gpu_pattern"],
            seq_len=target["seq_len"],
            eviction=target["eviction"],
            min_speedup=target["min_speedup"],
            max_error=target["max_error"],
            max_violations=target["max_violations"],
        )

        # Skip if GPU does not match
        if not _gpu_matches(gpu_name, target["gpu_pattern"]):
            vr.overall_status = Status.SKIP
            results.append(vr)
            continue

        # --- Speedup from profiling ---
        if profiling_data is not None:
            compact_row = _find_profiling_result(
                profiling_data, target["seq_len"], target["eviction"]
            )
            dense_row = _find_dense_baseline(profiling_data, target["seq_len"])

            if compact_row is not None and dense_row is not None:
                dense_ms = dense_row["mean_ms"]
                compact_ms_total = compact_row["mean_ms"]
                speedup = dense_ms / compact_ms_total if compact_ms_total > 0 else 0.0
                vr.actual_speedup = round(speedup, 4)
                vr.dense_ms = dense_ms
                vr.attention_ms = compact_ms_total  # profiling wraps entire pipeline

                # Evaluate status
                if speedup >= target["min_speedup"]:
                    vr.speedup_status = Status.PASS
                elif speedup >= target["min_speedup"] * 0.90:
                    vr.speedup_status = Status.WARN
                else:
                    vr.speedup_status = Status.FAIL

            # Spectral-mode overhead
            spectral_row = _find_spectral_dense_mode(profiling_data, target["seq_len"])
            if spectral_row is not None and dense_row is not None:
                vr.spectral_ms = spectral_row["mean_ms"] - dense_row["mean_ms"]

        # --- Compaction timing breakdown ---
        if compaction_data is not None:
            comp_row = _find_compaction_result(
                compaction_data, target["seq_len"], target["eviction"]
            )
            if comp_row is not None:
                vr.compact_ms = comp_row.get("compact_mean_ms")
                vr.total_ms = comp_row.get("total_orthocache_ms")
                if vr.attention_ms is None:
                    vr.attention_ms = comp_row.get("attention_mean_ms")
                if vr.dense_ms is None:
                    vr.dense_ms = comp_row.get("dense_mean_ms")

        # --- Reconstruction error ---
        if reconstruction_data is not None:
            recon_row = _find_reconstruction_result(
                reconstruction_data, target["seq_len"], target["eviction"]
            )
            if recon_row is not None:
                # Support various key names
                error_val = (
                    recon_row.get("relative_error")
                    or recon_row.get("mean_error")
                    or recon_row.get("reconstruction_error")
                    or recon_row.get("error")
                )
                if error_val is not None:
                    vr.actual_error = float(error_val)
                    if vr.actual_error <= target["max_error"]:
                        vr.error_status = Status.PASS
                    elif vr.actual_error <= target["max_error"] * 1.10:
                        vr.error_status = Status.WARN
                    else:
                        vr.error_status = Status.FAIL

                violations = recon_row.get("bound_violations", recon_row.get("violations"))
                if violations is not None:
                    vr.actual_violations = int(violations)
                    if vr.actual_violations <= target["max_violations"]:
                        vr.violation_status = Status.PASS
                    else:
                        vr.violation_status = Status.FAIL

        # --- Overall status ---
        statuses = [vr.speedup_status, vr.error_status, vr.violation_status]
        non_skip = [s for s in statuses if s != Status.SKIP]

        if not non_skip:
            vr.overall_status = Status.SKIP
        elif Status.FAIL in non_skip:
            vr.overall_status = Status.FAIL
        elif Status.WARN in non_skip:
            vr.overall_status = Status.WARN
        else:
            vr.overall_status = Status.PASS

        # --- Bottleneck analysis ---
        vr.bottleneck_notes = _analyse_bottleneck(vr)

        results.append(vr)

    return results


# ---------------------------------------------------------------------------
# Terminal table rendering
# ---------------------------------------------------------------------------

def _fmt_speedup(vr: ValidationResult) -> str:
    if vr.actual_speedup is None:
        return "  ---  "
    return f"{vr.actual_speedup:.2f}× ≥{vr.min_speedup:.2f}"


def _fmt_error(vr: ValidationResult) -> str:
    if vr.actual_error is None:
        return "  ---  "
    pct = vr.actual_error * 100
    max_pct = vr.max_error * 100
    return f"{pct:.1f}% ≤{max_pct:.1f}%"


def _print_terminal_table(results: list[ValidationResult], gpu_name: str) -> None:
    """Print a box-drawing table with colour-coded status."""
    print()
    print(_bold(f"  Target Validation — {gpu_name}"))
    print()

    # Header
    print("╔══════════════════╦════════╦═════════╦═══════════════╦═══════════════╦════════╗")
    print("║ GPU              ║ SeqLen ║ Evict%  ║ Speedup       ║ Error         ║ Status ║")
    print("╠══════════════════╬════════╬═════════╬═══════════════╬═══════════════╬════════╣")

    for vr in results:
        gpu_col = f"{vr.gpu_pattern:<16}"
        seq_col = f"{vr.seq_len:>6}"
        evict_col = f"{int(vr.eviction * 100):>3}%   "
        speedup_col = f"{_fmt_speedup(vr):<13}"
        error_col = f"{_fmt_error(vr):<13}"
        status_col = _colour_status(f" {vr.overall_status:^4} ")

        print(f"║ {gpu_col} ║ {seq_col} ║ {evict_col}║ {speedup_col} ║ {error_col} ║{status_col}║")

    print("╚══════════════════╩════════╩═════════╩═══════════════╩═══════════════╩════════╝")
    print()


def _print_bottleneck_analysis(results: list[ValidationResult]) -> None:
    """Print bottleneck notes for any failing/warning targets."""
    has_notes = any(vr.bottleneck_notes for vr in results)
    if not has_notes:
        return

    print(_bold("  Bottleneck Analysis"))
    print("  " + "─" * 66)

    for vr in results:
        if not vr.bottleneck_notes:
            continue
        header = f"  {vr.gpu_pattern} / seq={vr.seq_len} / evict={int(vr.eviction*100)}%"
        print(_yellow(header))
        for note in vr.bottleneck_notes:
            print(f"    {note}")
        print()


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _generate_markdown_report(
    results: list[ValidationResult],
    gpu_name: str,
    output_path: Path,
) -> None:
    """Write a full Markdown validation report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# OrthoCache Target Validation Report")
    lines.append("")
    lines.append(f"**Generated**: {now}  ")
    lines.append(f"**GPU**: {gpu_name}  ")
    lines.append("")

    # Summary counts
    counts = {Status.PASS: 0, Status.FAIL: 0, Status.WARN: 0, Status.SKIP: 0}
    for vr in results:
        counts[vr.overall_status] += 1

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ✅ PASS | {counts[Status.PASS]} |")
    lines.append(f"| ❌ FAIL | {counts[Status.FAIL]} |")
    lines.append(f"| ⚠️ WARN | {counts[Status.WARN]} |")
    lines.append(f"| ⏭️ SKIP | {counts[Status.SKIP]} |")
    lines.append("")

    total_applicable = counts[Status.PASS] + counts[Status.FAIL] + counts[Status.WARN]
    if total_applicable > 0:
        pass_rate = (counts[Status.PASS] / total_applicable) * 100
        lines.append(f"**Pass rate**: {pass_rate:.0f}% ({counts[Status.PASS]}/{total_applicable})")
    else:
        lines.append("**Pass rate**: N/A (no applicable targets for this GPU)")
    lines.append("")

    # Full results table
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| GPU | SeqLen | Evict% | Speedup | Target | Error | Target | Violations | Status |")
    lines.append("|-----|--------|--------|---------|--------|-------|--------|------------|--------|")

    for vr in results:
        speedup_str = f"{vr.actual_speedup:.2f}×" if vr.actual_speedup is not None else "—"
        target_speedup_str = f"≥{vr.min_speedup:.2f}×"
        error_str = f"{vr.actual_error*100:.2f}%" if vr.actual_error is not None else "—"
        target_error_str = f"≤{vr.max_error*100:.1f}%"
        violations_str = str(vr.actual_violations) if vr.actual_violations is not None else "—"

        status_emoji = {
            Status.PASS: "✅",
            Status.FAIL: "❌",
            Status.WARN: "⚠️",
            Status.SKIP: "⏭️",
        }[vr.overall_status]

        lines.append(
            f"| {vr.gpu_pattern} | {vr.seq_len} | {int(vr.eviction*100)}% "
            f"| {speedup_str} | {target_speedup_str} | {error_str} "
            f"| {target_error_str} | {violations_str} | {status_emoji} {vr.overall_status} |"
        )

    lines.append("")

    # Bottleneck analysis
    has_notes = any(vr.bottleneck_notes for vr in results)
    if has_notes:
        lines.append("## Bottleneck Analysis")
        lines.append("")
        for vr in results:
            if not vr.bottleneck_notes:
                continue
            lines.append(
                f"### {vr.gpu_pattern} — seq_len={vr.seq_len}, "
                f"eviction={int(vr.eviction*100)}%"
            )
            lines.append("")
            for note in vr.bottleneck_notes:
                if note.startswith("  →"):
                    lines.append(f"  - {note.strip()}")
                else:
                    lines.append(f"- {note}")
            lines.append("")

    # Recommendations
    if counts[Status.FAIL] > 0 or counts[Status.WARN] > 0:
        lines.append("## Recommendations")
        lines.append("")

        if counts[Status.FAIL] > 0:
            lines.append("### Critical (FAIL)")
            lines.append("")
            for vr in results:
                if vr.overall_status != Status.FAIL:
                    continue
                lines.append(
                    f"- **{vr.gpu_pattern} seq={vr.seq_len}**: "
                )
                if vr.speedup_status == Status.FAIL and vr.actual_speedup is not None:
                    lines.append(
                        f"  Speedup {vr.actual_speedup:.2f}× is below "
                        f"{vr.min_speedup:.2f}× target"
                    )
                if vr.error_status == Status.FAIL and vr.actual_error is not None:
                    lines.append(
                        f"  Error {vr.actual_error*100:.2f}% exceeds "
                        f"{vr.max_error*100:.1f}% limit"
                    )
                if vr.violation_status == Status.FAIL and vr.actual_violations is not None:
                    lines.append(
                        f"  {vr.actual_violations} bound violations "
                        f"(max allowed: {vr.max_violations})"
                    )
            lines.append("")

        if counts[Status.WARN] > 0:
            lines.append("### Warning (within 10% of target)")
            lines.append("")
            for vr in results:
                if vr.overall_status != Status.WARN:
                    continue
                lines.append(
                    f"- **{vr.gpu_pattern} seq={vr.seq_len}**: marginally "
                    f"passing — optimisation needed before production"
                )
            lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `target_validation.py` at {now}*")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def _print_summary(results: list[ValidationResult]) -> None:
    """Print a one-line pass/fail summary."""
    counts = {Status.PASS: 0, Status.FAIL: 0, Status.WARN: 0, Status.SKIP: 0}
    for vr in results:
        counts[vr.overall_status] += 1

    total = counts[Status.PASS] + counts[Status.FAIL] + counts[Status.WARN]
    parts = []
    if counts[Status.PASS]:
        parts.append(_green(f"{counts[Status.PASS]} PASS"))
    if counts[Status.FAIL]:
        parts.append(_red(f"{counts[Status.FAIL]} FAIL"))
    if counts[Status.WARN]:
        parts.append(_yellow(f"{counts[Status.WARN]} WARN"))
    if counts[Status.SKIP]:
        parts.append(_dim(f"{counts[Status.SKIP]} SKIP"))

    summary = " │ ".join(parts)
    if total > 0:
        pct = (counts[Status.PASS] / total) * 100
        print(f"  Result: {summary}  ({pct:.0f}% pass rate)")
    else:
        print(f"  Result: {summary}  (no applicable targets)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_target_validation() -> list[ValidationResult]:
    """Execute the full target-validation pipeline.

    Returns the list of :class:`ValidationResult` objects.
    """
    results_dir = Path("benchmarks/results")

    print()
    print("=" * 70)
    print("  OrthoCache — Post-Benchmark Target Validation")
    print("=" * 70)
    print()

    # Load profiling data
    profiling_path = results_dir / "gpu_profiling_results.json"
    profiling_data = _load_json(profiling_path)
    if profiling_data is not None:
        print(f"  [✓] Loaded profiling data  ({len(profiling_data)} entries)")
    else:
        print(f"  [✗] Profiling data not found: {profiling_path}")

    # Load reconstruction-error data
    recon_path = results_dir / "reconstruction_error_results.json"
    recon_data = _load_json(recon_path)
    if recon_data is not None:
        count = len(recon_data) if isinstance(recon_data, list) else 1
        print(f"  [✓] Loaded reconstruction data  ({count} entries)")
    else:
        print(f"  [·] Reconstruction data not found: {recon_path} (error checks skipped)")

    # Load compaction data (for bottleneck breakdown)
    compaction_path = results_dir / "compaction_results.json"
    compaction_data = _load_json(compaction_path)
    if compaction_data is not None:
        print(f"  [✓] Loaded compaction data  ({len(compaction_data)} entries)")
    else:
        print(f"  [·] Compaction data not found: {compaction_path} (breakdown unavailable)")

    # Detect GPU
    gpu_name = _detect_gpu_name(profiling_data or [])
    print(f"\n  GPU detected: {_bold(gpu_name)}")

    # Validate
    results = validate_targets(
        gpu_name=gpu_name,
        profiling_data=profiling_data,
        reconstruction_data=recon_data if isinstance(recon_data, list) else None,
        compaction_data=compaction_data,
    )

    # Terminal output
    _print_terminal_table(results, gpu_name)
    _print_bottleneck_analysis(results)
    _print_summary(results)

    # Markdown report
    report_path = results_dir / "target_validation_report.md"
    _generate_markdown_report(results, gpu_name, report_path)
    print(f"  Report written to {report_path.resolve()}")
    print()

    return results


if __name__ == "__main__":
    run_target_validation()
