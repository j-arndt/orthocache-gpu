"""Calibrate h_median for the EntropyGovernor.

Measures attention entropy across layers and windows to find
the median entropy for use as h_median in the sigmoid modulator.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, statistics

# Add parent to path
sys.path.insert(0, ".")
from eval.perplexity_eval import load_wikitext2

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "C:/LearningFolder/tinyllama1.1b"
MAX_LENGTH = int(sys.argv[2]) if len(sys.argv) > 2 else 256

print(f"Calibrating entropy for {MODEL_PATH}, max_length={MAX_LENGTH}")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float32,
    attn_implementation="eager",
)
model.eval()

# Measure entropy from a few forward passes
windows = load_wikitext2(tokenizer, max_length=MAX_LENGTH, stride=MAX_LENGTH)

num_layers = model.config.num_hidden_layers
patched_layers = list(range(num_layers // 2, num_layers))  # 11-21
print(f"Measuring entropy for layers {patched_layers[0]}-{patched_layers[-1]}")

all_entropies = []
layer_entropies = {l: [] for l in patched_layers}

with torch.no_grad():
    for i, input_ids in enumerate(windows[:5]):
        input_ids = input_ids.unsqueeze(0)
        outputs = model(input_ids, output_attentions=True)
        
        for layer_idx in patched_layers:
            attn = outputs.attentions[layer_idx]  # (batch, heads, seq, seq)
            p = attn.clamp(min=1e-10)
            H = -(p * p.log()).sum(dim=-1)  # (batch, heads, seq)
            mean_H = H.mean().item()
            all_entropies.append(mean_H)
            layer_entropies[layer_idx].append(mean_H)
        
        print(f"  Window {i+1}/5 done")

print(f"\n{'='*60}")
print(f"ENTROPY CALIBRATION RESULTS")
print(f"{'='*60}")

print(f"\nPer-layer mean entropy:")
for l in patched_layers:
    vals = layer_entropies[l]
    print(f"  Layer {l:2d}: H = {statistics.mean(vals):.4f} "
          f"(std={statistics.stdev(vals) if len(vals) > 1 else 0:.4f})")

print(f"\nGlobal statistics:")
print(f"  Median:  {statistics.median(all_entropies):.4f}")
print(f"  Mean:    {statistics.mean(all_entropies):.4f}")
print(f"  Std:     {statistics.stdev(all_entropies):.4f}")
print(f"  Range:   [{min(all_entropies):.4f}, {max(all_entropies):.4f}]")
print(f"\n  >>> Recommended h_median = {statistics.median(all_entropies):.4f}")

# Also calibrate at 2048 if we're at 256
if MAX_LENGTH < 2048:
    print(f"\nNote: Re-run with max_length=2048 for long-context h_median calibration.")
