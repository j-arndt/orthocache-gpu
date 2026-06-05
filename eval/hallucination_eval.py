"""Gold 3: Hallucination Exhaust — Zero-Cost Hallucination Detector.

Uses the spectral norms already computed by OrthoCache's Cauchy-Schwarz gate
to detect when the model is "searching" for information that doesn't exist
in context (hallucination).

Hallucination Score:
    H_score(l) = max_g ||Q_{g,high}||_2 / (max_tile ||K_high||_F + eps)

    H_score >> 1: Likely hallucination (high search, low info)
    H_score ~= 1: Normal retrieval
    H_score << 1: Lazy generation (low search intensity)

This detector costs ZERO extra FLOPs — it reuses values already in registers.
"""

import sys
import os
import json
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.perplexity_eval import (
    OrthoCache_GQA_Attention,
    patch_model_attention,
    generate_walsh_matrix,
    ResidualGovernor,
)


# ============================================================================
# Test Prompts
# ============================================================================

FACTUAL_PROMPTS = [
    # Facts the model knows from training
    "The Eiffel Tower is located in Paris, France. It was built in 1889 for the World's Fair. "
    "The tower is 324 metres tall and was the tallest man-made structure in the world until 1930. "
    "Question: Where is the Eiffel Tower located? Answer: The Eiffel Tower is located in",

    "Water has the chemical formula H2O. It consists of two hydrogen atoms bonded to one oxygen atom. "
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius at standard atmospheric pressure. "
    "Question: What is the chemical formula of water? Answer: The chemical formula of water is",

    "The speed of light in a vacuum is approximately 299,792,458 metres per second. "
    "This constant is fundamental to physics and is denoted by the letter c. "
    "Question: What is the speed of light? Answer: The speed of light is approximately",

    "Python is a high-level programming language created by Guido van Rossum. "
    "It was first released in 1991 and emphasizes code readability with significant whitespace. "
    "Question: Who created Python? Answer: Python was created by",

    "The human heart has four chambers: two atria and two ventricles. "
    "The right side pumps blood to the lungs, while the left side pumps blood to the body. "
    "Question: How many chambers does the human heart have? Answer: The human heart has",
]

HALLUCINATION_PROMPTS = [
    # Fabricated facts the model CANNOT find in context
    "The Glorpnax Corporation was founded in 2019 in New Zolandra by CEO Thrumbus McFinkle. "
    "It specializes in quantum florbination for industrial moldavite processing. "
    "Question: What year was the Glorpnax Corporation's revenue target for q4? Answer: The Glorpnax Corporation's q4 revenue target was",

    "Professor Xanthium Belverdere published the seminal paper on chromatic destabilization in 2017. "
    "His research at the University of Krendalia showed that destabilization occurs at 47.3 kelvin. "
    "Question: What was Professor Belverdere's h-index in 2022? Answer: Professor Belverdere's h-index was",

    "The Trantulian Protocol requires all member states to submit biannual reports on zephyrite emissions. "
    "Non-compliance results in sanctions under Article 47b of the Trantulian Charter. "
    "Question: How many countries have ratified the Trantulian Protocol? Answer: The number of countries that ratified is",

    "FluxoMatic 3000 is an enterprise software platform for managing distributed cronkite pipelines. "
    "Version 7.2 introduced the breakthrough NeuroPlex engine for real-time thrombosis calculation. "
    "Question: What is the maximum throughput of FluxoMatic 3000 version 7.2? Answer: The maximum throughput is",

    "The island nation of Meridisia has a population of 2.3 million and uses the Meridisian Drachma. "
    "Its capital city, Port Aurelia, is famous for the Grand Basilica of Saint Thornmund. "
    "Question: What is the GDP per capita of Meridisia? Answer: The GDP per capita of Meridisia is",
]


def run_hallucination_eval(
    model_path: str,
    tau: float = 1.0,
    alpha: float = 0.3,
    device_str: str = "cpu",
) -> Dict[str, List[float]]:
    """Run hallucination detection evaluation.
    
    Returns dict with 'factual_scores' and 'hallucination_scores' lists.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device_str)
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32,
        device_map=device.type if device.type == 'cuda' else None,
    )
    if device.type != 'cuda':
        model = model.to(device)
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    min_layer = num_layers // 2
    G = model.config.num_attention_heads // model.config.num_key_value_heads
    
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  GQA: G={G}, patching layers {min_layer}-{num_layers-1}")
    print(f"  tau={tau}, alpha={alpha}")

    results = {
        'factual_scores': [],
        'factual_q_intensity': [],
        'factual_k_density': [],
        'hallucination_scores': [],
        'hallucination_q_intensity': [],
        'hallucination_k_density': [],
    }

    def evaluate_prompts(prompts, label, score_key, q_key, k_key):
        for i, prompt in enumerate(prompts):
            # Create fresh OrthoCache for each prompt
            governor = ResidualGovernor(alpha=alpha) if alpha > 0 else None
            orthocache = OrthoCache_GQA_Attention(
                tau=tau, verbose=False, governor=governor,
            )
            patched_model = patch_model_attention(model, orthocache, min_layer=min_layer)
            
            # Reset hook for governor
            if governor is not None:
                first_patched_layer = model.model.layers[min_layer]
                def make_reset_hook(gov):
                    def hook(module, args):
                        gov.reset()
                    return hook
                reset_handle = first_patched_layer.register_forward_pre_hook(
                    make_reset_hook(governor)
                )

            # Tokenize and run
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                outputs = patched_model(input_ids)
            
            # Collect hallucination telemetry
            h_scores = orthocache.stats['hallucination_scores']
            q_intensities = orthocache.stats['q_search_intensity']
            k_densities = orthocache.stats['k_info_density']
            
            if h_scores:
                # Use the max H_score across all layers (worst case)
                max_h = max(h_scores)
                mean_h = sum(h_scores) / len(h_scores)
                max_q = max(q_intensities) if q_intensities else 0
                max_k = max(k_densities) if k_densities else 0
                
                results[score_key].append(max_h)
                results[q_key].append(max_q)
                results[k_key].append(max_k)
                
                print(f"  [{label} {i+1}] H_score={max_h:.4f} "
                      f"(mean={mean_h:.4f}) "
                      f"Q_high={max_q:.4f} K_high={max_k:.4f}")
            
            # Cleanup
            if governor is not None:
                reset_handle.remove()
            
            # Unpatch
            model_fresh = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.float32,
                device_map=device.type if device.type == 'cuda' else None,
            )
            if device.type != 'cuda':
                model_fresh = model_fresh.to(device)
            # Re-assign for next iteration
            # (model variable in outer scope won't change, but we
            #  need to re-patch each time anyway)

    print(f"\n{'='*60}")
    print(f"FACTUAL PROMPTS (expected H_score ~ 1)")
    print(f"{'='*60}")
    evaluate_prompts(
        FACTUAL_PROMPTS, "Factual",
        'factual_scores', 'factual_q_intensity', 'factual_k_density'
    )

    print(f"\n{'='*60}")
    print(f"HALLUCINATION-INDUCING PROMPTS (expected H_score >> 1)")
    print(f"{'='*60}")
    evaluate_prompts(
        HALLUCINATION_PROMPTS, "Halluc",
        'hallucination_scores', 'hallucination_q_intensity', 'hallucination_k_density'
    )

    # Compute separation metrics
    print(f"\n{'='*60}")
    print(f"HALLUCINATION EXHAUST ANALYSIS")
    print(f"{'='*60}")
    
    f_scores = results['factual_scores']
    h_scores = results['hallucination_scores']
    
    if f_scores and h_scores:
        f_mean = sum(f_scores) / len(f_scores)
        h_mean = sum(h_scores) / len(h_scores)
        
        print(f"\n  Factual H_score:        mean={f_mean:.4f}, "
              f"range=[{min(f_scores):.4f}, {max(f_scores):.4f}]")
        print(f"  Hallucination H_score:  mean={h_mean:.4f}, "
              f"range=[{min(h_scores):.4f}, {max(h_scores):.4f}]")
        
        # Simple AUROC via Mann-Whitney U
        correct = 0
        total = 0
        for h in h_scores:
            for f in f_scores:
                total += 1
                if h > f:
                    correct += 1
                elif h == f:
                    correct += 0.5
        
        auroc = correct / total if total > 0 else 0.5
        separation = h_mean / (f_mean + 1e-10)
        
        print(f"\n  Separation ratio:  {separation:.2f}x")
        print(f"  AUROC:             {auroc:.4f}")
        
        if auroc > 0.7:
            print(f"\n  >>> HALLUCINATION DETECTOR: VIABLE (AUROC > 0.7)")
        elif auroc > 0.5:
            print(f"\n  >>> HALLUCINATION DETECTOR: WEAK SIGNAL (AUROC > 0.5)")
        else:
            print(f"\n  >>> HALLUCINATION DETECTOR: NO SIGNAL")
        
        results['auroc'] = auroc
        results['separation_ratio'] = separation
    
    return results


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "C:/LearningFolder/tinyllama1.1b"
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    results = run_hallucination_eval(
        model_path=model_path,
        tau=tau,
        alpha=0.3,
        device_str="cpu",
    )
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "hallucination_exhaust.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
