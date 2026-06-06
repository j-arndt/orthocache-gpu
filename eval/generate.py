"""Gold 1+3+4 Convergence: Autoregressive Generation with OrthoCache.

Splits the monkey-patched attention into two physics:
  PREFILL: Full FWHT, populate SpectralNormCache (one-time O(N log N))
  DECODE:  O(1) gate from cached norms (per token)

This harness enables:
  Gold 1: Measure decode latency with/without norm cache bypass
  Gold 3: Watch Hallucination Exhaust (H_score) evolve per decode step
  Gold 4: Needle-In-A-Haystack retrieval accuracy under eviction

Usage:
    # Basic generation
    python eval/generate.py --model-path C:/LearningFolder/tinyllama1.1b \\
        --prompt "The capital of France is" --max-tokens 50 --tau 1.06

    # Needle-In-A-Haystack
    python eval/generate.py --model-path C:/LearningFolder/tinyllama1.1b \\
        --needle --tau 1.06

    # Compare dense vs OrthoCache
    python eval/generate.py --model-path C:/LearningFolder/tinyllama1.1b \\
        --prompt "Explain quantum entanglement:" --max-tokens 100 --compare
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orthocache_gpu.norm_cache import SpectralNormCache

# Import from perplexity_eval
sys.path.insert(0, str(Path(__file__).parent))
from perplexity_eval import (
    OrthoCache_GQA_Attention,
    patch_model_attention,
    generate_walsh_matrix,
)

try:
    from orthocache_gpu.eviction_governor import ResidualGovernor
except ImportError:
    ResidualGovernor = None


# ============================================================================
# Needle-In-A-Haystack Prompts
# ============================================================================

HAYSTACK_FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "Meanwhile, birds chirped melodiously in the distant oak trees. "
    "The wind carried leaves across the meadow as clouds drifted. "
    "Sunlight filtered through branches casting dappled shadows. "
    "A butterfly landed on a wildflower swaying gently in the breeze. "
    "The old stone bridge arched over the babbling brook below. "
    "Children played in the village square while merchants sold goods. "
    "Farmers tended their fields as the sun climbed higher in the sky. "
)

NEEDLE_SENTENCE = "The secret code for the vault is BLUE-FALCON-42."

NEEDLE_QUESTION = "\n\nQuestion: What is the secret code for the vault?\nAnswer: The secret code is"


def build_needle_prompt(num_filler_repeats: int = 8) -> str:
    """Build a Needle-In-A-Haystack prompt.
    
    Structure: [filler] [NEEDLE] [filler] [question]
    The needle is buried at ~25% depth into the context.
    """
    filler_before = HAYSTACK_FILLER * (num_filler_repeats // 4)
    filler_after = HAYSTACK_FILLER * (num_filler_repeats * 3 // 4)
    return filler_before + NEEDLE_SENTENCE + " " + filler_after + NEEDLE_QUESTION


# ============================================================================
# Generation Engine
# ============================================================================

def generate_with_orthocache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    tau: float = 1.06,
    alpha: float = 0.3,
    device: str = "cpu",
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = True,
) -> Dict:
    """Generate text token-by-token with OrthoCache prefill/decode split.
    
    Returns dict with:
        text: generated text
        tokens: list of generated token IDs
        prefill_time_ms: time for prefill phase
        decode_time_ms: time for decode phase (total)
        decode_time_per_token_ms: average per-token decode time
        h_scores_per_step: Gold 3 hallucination exhaust per decode step
        decode_eviction_rate: fraction of tiles skipped during decode
    """
    device = torch.device(device)
    num_layers = model.config.num_hidden_layers
    min_layer = num_layers // 2
    num_kv_heads = model.config.num_key_value_heads
    G = model.config.num_attention_heads // num_kv_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    # Create OrthoCache with norm cache for decode
    governor = ResidualGovernor(alpha=alpha) if (ResidualGovernor and alpha > 0) else None
    orthocache = OrthoCache_GQA_Attention(
        tau=tau, verbose=False, governor=governor,
    )
    
    # Create SpectralNormCache
    # Max tiles = max_seq_len / tile_size
    max_tiles = (2048 + max_new_tokens) // 64 + 1
    norm_cache = SpectralNormCache(
        num_kv_heads=num_kv_heads,
        max_tiles=max_tiles,
        device=device,
    )
    orthocache.norm_cache = norm_cache
    
    # Patch model
    patched_model = patch_model_attention(model, orthocache, min_layer=min_layer)
    
    # Governor reset hook
    reset_handle = None
    if governor is not None:
        first_patched_layer = model.model.layers[min_layer]
        def make_reset_hook(gov):
            def hook(module, args):
                gov.reset()
            return hook
        reset_handle = first_patched_layer.register_forward_pre_hook(
            make_reset_hook(governor)
        )
    
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    
    if verbose:
        print(f"\n  Prompt: {prompt_len} tokens")
        print(f"  Config: tau={tau}, alpha={alpha}, max_new={max_new_tokens}")
        print(f"  Norm cache: {num_kv_heads} heads × {max_tiles} tiles")
    
    # ========================================================================
    # PHASE 1: PREFILL — Full FWHT, populate norm cache
    # ========================================================================
    if verbose:
        print(f"\n  [PREFILL] Processing {prompt_len} tokens...")
    
    t_prefill_start = time.perf_counter()
    
    with torch.no_grad():
        outputs = patched_model(
            input_ids,
            use_cache=True,
        )
    
    t_prefill_end = time.perf_counter()
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000
    
    past_key_values = outputs.past_key_values
    prefill_eviction = orthocache.eviction_rate
    prefill_h_scores = list(orthocache.stats['hallucination_scores'])
    
    if verbose:
        print(f"  [PREFILL] Done in {prefill_ms:.1f}ms")
        print(f"  [PREFILL] Eviction: {prefill_eviction*100:.1f}%")
        print(f"  [PREFILL] Norm cache populated: {norm_cache.is_populated}")
        print(f"  [PREFILL] Tiles cached per head: "
              f"{[norm_cache.valid_tiles[h].item() for h in range(min(4, num_kv_heads))]}"
              f"{'...' if num_kv_heads > 4 else ''}")
    
    # ========================================================================
    # PHASE 2: DECODE — O(1) gate from norm cache, token by token
    # ========================================================================
    if verbose:
        print(f"\n  [DECODE] Generating up to {max_new_tokens} tokens...")
    
    generated_tokens = []
    h_scores_per_step = []
    decode_times = []
    
    # Get the last token's logits for the first new token
    next_token_logits = outputs.logits[:, -1, :] / temperature
    
    for step in range(max_new_tokens):
        # Sample next token
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_tokens.append(next_token.item())
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            if verbose:
                print(f"    Step {step+1}: <EOS>")
            break
        
        # Reset stats for this decode step
        orthocache.reset_stats()
        if governor is not None:
            governor.reset()
        
        # Forward pass with KV cache — seq_len_q = 1
        t_decode_start = time.perf_counter()
        
        with torch.no_grad():
            outputs = patched_model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        t_decode_end = time.perf_counter()
        decode_ms = (t_decode_end - t_decode_start) * 1000
        decode_times.append(decode_ms)
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :] / temperature
        
        # Collect Gold 3 telemetry for this step
        step_h_scores = orthocache.stats['hallucination_scores']
        if step_h_scores:
            max_h = max(step_h_scores)
            h_scores_per_step.append(max_h)
        
        # Print progress
        if verbose and (step < 5 or step % 10 == 0):
            token_text = tokenizer.decode([next_token.item()])
            decode_info = f"    Step {step+1}: '{token_text}' ({decode_ms:.1f}ms)"
            if step_h_scores:
                decode_info += f" H={max_h:.4f}"
            if orthocache.stats['decode_tiles_total'] > 0:
                skip_rate = orthocache.stats['decode_tiles_skipped'] / orthocache.stats['decode_tiles_total']
                decode_info += f" skip={skip_rate*100:.0f}%"
            print(decode_info)
    
    # Cleanup
    if reset_handle is not None:
        reset_handle.remove()
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Compute summary stats
    total_decode_ms = sum(decode_times) if decode_times else 0
    avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0
    
    results = {
        'prompt': prompt[:200],  # truncate for JSON
        'generated_text': generated_text,
        'prompt_tokens': prompt_len,
        'generated_tokens': len(generated_tokens),
        'tau': tau,
        'alpha': alpha,
        'prefill_time_ms': round(prefill_ms, 2),
        'prefill_eviction_rate': round(prefill_eviction, 4),
        'decode_time_total_ms': round(total_decode_ms, 2),
        'decode_time_per_token_ms': round(avg_decode_ms, 2),
        'decode_times_ms': [round(t, 2) for t in decode_times],
        'h_scores_per_step': [round(h, 6) for h in h_scores_per_step],
        'norm_cache_populated': norm_cache.is_populated,
    }
    
    if verbose:
        print(f"\n  {'='*60}")
        print(f"  GENERATION SUMMARY")
        print(f"  {'='*60}")
        print(f"  Generated: {len(generated_tokens)} tokens")
        print(f"  Prefill:   {prefill_ms:.1f}ms ({prompt_len} tokens)")
        print(f"  Decode:    {total_decode_ms:.1f}ms total, {avg_decode_ms:.1f}ms/token")
        print(f"  Prefill eviction: {prefill_eviction*100:.1f}%")
        if h_scores_per_step:
            print(f"  H_score range: [{min(h_scores_per_step):.4f}, {max(h_scores_per_step):.4f}]")
        print(f"\n  Output: {generated_text}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="OrthoCache Autoregressive Generation")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--tau", type=float, default=1.06)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--needle", action="store_true",
                        help="Run Needle-In-A-Haystack test")
    parser.add_argument("--needle-repeats", type=int, default=8,
                        help="Filler repetitions for needle test")
    parser.add_argument("--compare", action="store_true",
                        help="Compare dense vs OrthoCache generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float32,
        device_map=args.device if args.device == 'cuda' else None,
    )
    if args.device != 'cuda':
        model = model.to(torch.device(args.device))
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    G = model.config.num_attention_heads // model.config.num_key_value_heads
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  GQA: G={G}, {model.config.num_key_value_heads} KV heads")
    
    # Select prompt
    if args.needle:
        prompt = build_needle_prompt(args.needle_repeats)
        print(f"\n{'='*60}")
        print(f"NEEDLE-IN-A-HAYSTACK TEST")
        print(f"{'='*60}")
        print(f"  Needle: '{NEEDLE_SENTENCE}'")
        print(f"  Filler repeats: {args.needle_repeats}")
    else:
        prompt = args.prompt
    
    # Run generation
    print(f"\n{'='*60}")
    print(f"ORTHOCACHE GENERATION (tau={args.tau})")
    print(f"{'='*60}")
    
    results = generate_with_orthocache(
        model, tokenizer, prompt,
        max_new_tokens=args.max_tokens,
        tau=args.tau,
        alpha=args.alpha,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Optional: compare with dense baseline
    if args.compare:
        print(f"\n{'='*60}")
        print(f"DENSE BASELINE (no OrthoCache)")
        print(f"{'='*60}")
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        
        t_start = time.perf_counter()
        with torch.no_grad():
            dense_output = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
            )
        t_end = time.perf_counter()
        
        dense_text = tokenizer.decode(
            dense_output[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        dense_ms = (t_end - t_start) * 1000
        
        print(f"  Dense generation: {dense_ms:.1f}ms")
        print(f"  Dense output: {dense_text}")
        
        results['dense_text'] = dense_text
        results['dense_time_ms'] = round(dense_ms, 2)
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
