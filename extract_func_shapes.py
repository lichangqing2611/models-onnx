#!/usr/bin/env python3
"""Enumerate every intermediate tensor inside BertEncoderLayer with its inferred shape."""
# Shapes derived from code analysis + ONNX broadcasting rules
shapes = {
    # Inputs
    "hidden_states":    "[B, S, 768]",
    "attention_mask":   "[B, S] (int64)",

    # Q/K/V projections
    "Q_mm":  "[B, S, 768]",  "K_mm":  "[B, S, 768]",  "V_mm":  "[B, S, 768]",
    "Q":     "[B, S, 768]",  "K":     "[B, S, 768]",  "V":     "[B, S, 768]",

    # Reshape → 4D
    "Q_4d": "[B, S, 12, 64]", "K_4d": "[B, S, 12, 64]", "V_4d": "[B, S, 12, 64]",

    # Transpose → [B, heads, S, head_dim]
    "Q_t":  "[B, 12, S, 64]", "K_t":  "[B, 12, S, 64]", "V_t":  "[B, 12, S, 64]",

    # K^T
    "K_tT": "[B, 12, 64, S]",

    # Scores
    "scores_raw":    "[B, 12, S, S]",
    "scores":        "[B, 12, S, S]",

    # Mask processing
    "mask_f":   "[B, S] (float)",
    "mask_4d":  "[B, 1, 1, S]",
    "mask_inv": "[B, 1, 1, S]",
    "mask_add": "[B, 1, 1, S]",

    # After softmax
    "scores_masked": "[B, 12, S, S]",
    "probs":         "[B, 12, S, S]",

    # Context
    "ctx_t":  "[B, 12, S, 64]",
    "ctx_4d": "[B, S, 12, 64]",
    "ctx":    "[B, S, 768]",

    # Attention output projection
    "ao_mm": "[B, S, 768]",
    "ao":    "[B, S, 768]",

    # Residual + LayerNorm 1
    "res1":  "[B, S, 768]",
    "norm1": "[B, S, 768]",

    # Feed-forward
    "fi_mm": "[B, S, 3072]",
    "fi":    "[B, S, 3072]",
    "gelu":  "[B, S, 3072]",
    "fo_mm": "[B, S, 768]",
    "fo":    "[B, S, 768]",

    # Residual + LayerNorm 2
    "res2":  "[B, S, 768]",
    "output":"[B, S, 768]",
}

print("=== BertEncoderLayer Internal Tensor Shapes ===\n")
for name, shape in shapes.items():
    print(f"  {name:20s}  {shape}")
