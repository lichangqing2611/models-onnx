#!/usr/bin/env python3
"""
Manually build a BERT-base model using the ONNX Python API.
Uses model-local functions: each encoder layer is a reusable
BertEncoderLayer function. In Netron, each function call node
appears as an expandable block showing all internal ops.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, checker, shape_inference

# ── BERT-Base Configuration ─────────────────────────────────
HIDDEN       = 768
HEADS        = 12
HEAD_DIM     = HIDDEN // HEADS       # 64
INTERMEDIATE = 3072
VOCAB        = 30522
MAX_POS      = 512
TYPE_VOCAB   = 2
LAYERS       = 12
EPS          = 1e-12
OPSET        = 20

# Dynamic dimension names
B = "batch"
S = "seq_len"

# ── Global initializer list ─────────────────────────────────
init_list = []

def add_tensor(name: str, data: np.ndarray, dtype: int = TensorProto.FLOAT):
    """Add a named initializer to the global list."""
    arr = np.asarray(data, dtype=np.float32 if dtype == TensorProto.FLOAT else np.int64)
    t = helper.make_tensor(name, dtype, list(arr.shape), arr.tobytes(), raw=True)
    init_list.append(t)
    return name

def add_weight(name: str, shape: tuple):
    """Add a weight initializer ~ N(0, 0.02²)."""
    return add_tensor(name, np.random.randn(*shape).astype(np.float32) * 0.02)

def add_zero(name: str, shape: tuple):
    """Add a zero bias initializer."""
    return add_tensor(name, np.zeros(shape, dtype=np.float32))

# ── Constants (shared across main graph) ───────────────────
add_tensor("zero_idx",  np.array(0, dtype=np.int64), TensorProto.INT64)   # scalar 0 for Gather

# ── Embedding weights ───────────────────────────────────────
add_weight("emb_word",  (VOCAB,    HIDDEN))
add_weight("emb_pos",   (MAX_POS,  HIDDEN))
add_weight("emb_type",  (TYPE_VOCAB, HIDDEN))
add_weight("emb_ln_w",  (HIDDEN,))
add_zero  ("emb_ln_b",  (HIDDEN,))

# ── Encoder-layer weights (16 per layer) ────────────────────
for l in range(LAYERS):
    p = f"L{l}"
    add_weight(f"{p}_q_w",   (HIDDEN,      HIDDEN))
    add_zero  (f"{p}_q_b",   (HIDDEN,))
    add_weight(f"{p}_k_w",   (HIDDEN,      HIDDEN))
    add_zero  (f"{p}_k_b",   (HIDDEN,))
    add_weight(f"{p}_v_w",   (HIDDEN,      HIDDEN))
    add_zero  (f"{p}_v_b",   (HIDDEN,))
    add_weight(f"{p}_ao_w",  (HIDDEN,      HIDDEN))
    add_zero  (f"{p}_ao_b",  (HIDDEN,))
    add_weight(f"{p}_aln_w", (HIDDEN,))
    add_zero  (f"{p}_aln_b", (HIDDEN,))
    add_weight(f"{p}_fi_w",  (HIDDEN,      INTERMEDIATE))   # (K, H) for MatMul(hidden, W^T)
    add_zero  (f"{p}_fi_b",  (INTERMEDIATE,))
    add_weight(f"{p}_fo_w",  (INTERMEDIATE, HIDDEN))
    add_zero  (f"{p}_fo_b",  (HIDDEN,))
    add_weight(f"{p}_fln_w", (HIDDEN,))
    add_zero  (f"{p}_fln_b", (HIDDEN,))

# ── Pooler weights ──────────────────────────────────────────
add_weight("pooler_w", (HIDDEN, HIDDEN))
add_zero  ("pooler_b", (HIDDEN,))


# ════════════════════════════════════════════════════════════
#  Encoder Layer Function  (model-local, reusable)
# ════════════════════════════════════════════════════════════

def make_encoder_func():
    """Return a FunctionProto for one full BERT encoder layer.

    Inputs (19 positional):
        0  hidden_states   (B, S, H)
        1  attention_mask  (B, S)  int64
        2  q_w,  q_b
        4  k_w,  k_b
        6  v_w,  v_b
        8  ao_w, ao_b     attention-output dense
       10  aln_w, aln_b    attention LayerNorm
       12  fi_w, fi_b      feed-forward intermediate
       14  fo_w, fo_b      feed-forward output
       16  fln_w, fln_b    feed-forward LayerNorm
    Output:
        output  (B, S, H)
    """
    inputs = [
        "hidden_states", "attention_mask",
        "q_w","q_b","k_w","k_b","v_w","v_b",
        "ao_w","ao_b","aln_w","aln_b",
        "fi_w","fi_b","fo_w","fo_b","fln_w","fln_b",
    ]
    nodes = []

    # ── 0. Constants (self-contained, no external refs) ──
    nodes.append(helper.make_node("Constant", [], ["shape_qkv4d"],
        value_ints=[0, 0, HEADS, HEAD_DIM]))
    nodes.append(helper.make_node("Constant", [], ["shape_3d"],
        value_ints=[0, 0, HIDDEN]))
    nodes.append(helper.make_node("Constant", [], ["shape_mask4d"],
        value_ints=[0, 1, 1, 0]))
    nodes.append(helper.make_node("Constant", [], ["hd_scale"],
        value_float=float(1.0 / np.sqrt(HEAD_DIM))))
    nodes.append(helper.make_node("Constant", [], ["one_f"],
        value_float=1.0))
    nodes.append(helper.make_node("Constant", [], ["neg_10000"],
        value_float=-10000.0))

    # ── 1. Q, K, V projections ──
    for proj, name in [("q","Q"), ("k","K"), ("v","V")]:
        nodes.append(helper.make_node("MatMul", ["hidden_states", f"{proj}_w"], [f"{name}_mm"]))
        nodes.append(helper.make_node("Add",    [f"{name}_mm",      f"{proj}_b"],  [name]))

    # ── 2. Reshape → 4D + Transpose → [B, Nh, S, Hd] ──
    for name in ["Q", "K", "V"]:
        nodes.append(helper.make_node("Reshape",   [name, "shape_qkv4d"], [f"{name}_4d"]))
        nodes.append(helper.make_node("Transpose", [f"{name}_4d"],        [f"{name}_t"],
                                      perm=[0, 2, 1, 3]))

    # ── 3. K^T for matmul: [B,12,S,64] → [B,12,64,S] ──
    nodes.append(helper.make_node("Transpose", ["K_t"], ["K_tT"], perm=[0, 1, 3, 2]))

    # ── 4. Q·K^T / √d ──
    nodes.append(helper.make_node("MatMul", ["Q_t", "K_tT"],        ["scores_raw"]))
    nodes.append(helper.make_node("Mul",    ["scores_raw", "hd_scale"], ["scores"]))

    # ── 5. Attention mask  (B,S) int64 → (B,1,1,S) float → additive mask ──
    nodes.append(helper.make_node("Cast",    ["attention_mask"],  ["mask_f"],  to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Reshape", ["mask_f", "shape_mask4d"], ["mask_4d"]))
    nodes.append(helper.make_node("Sub",     ["one_f", "mask_4d"],   ["mask_inv"]))
    nodes.append(helper.make_node("Mul",     ["mask_inv", "neg_10000"], ["mask_add"]))
    nodes.append(helper.make_node("Add",     ["scores", "mask_add"],    ["scores_masked"]))

    # ── 6. Softmax ──
    nodes.append(helper.make_node("Softmax", ["scores_masked"], ["probs"], axis=-1))

    # ── 7. Context = probs · V ──
    nodes.append(helper.make_node("MatMul", ["probs", "V_t"], ["ctx_t"]))

    # ── 8. Transpose back → Reshape to [B, S, H] ──
    nodes.append(helper.make_node("Transpose", ["ctx_t"],          ["ctx_4d"], perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Reshape",   ["ctx_4d", "shape_3d"], ["ctx"]))

    # ── 9. Attention output projection ──
    nodes.append(helper.make_node("MatMul", ["ctx", "ao_w"], ["ao_mm"]))
    nodes.append(helper.make_node("Add",    ["ao_mm", "ao_b"], ["ao"]))

    # ── 10. Residual + LayerNorm ──
    nodes.append(helper.make_node("Add",  ["hidden_states", "ao"], ["res1"]))
    nodes.append(helper.make_node("LayerNormalization",
        ["res1", "aln_w", "aln_b"], ["norm1"], epsilon=EPS))

    # ── 11. Feed-Forward: up → GELU → down ──
    nodes.append(helper.make_node("MatMul", ["norm1", "fi_w"], ["fi_mm"]))
    nodes.append(helper.make_node("Add",    ["fi_mm", "fi_b"], ["fi"]))
    nodes.append(helper.make_node("Gelu",   ["fi"],            ["gelu"], approximate="tanh"))
    nodes.append(helper.make_node("MatMul", ["gelu", "fo_w"],  ["fo_mm"]))
    nodes.append(helper.make_node("Add",    ["fo_mm", "fo_b"], ["fo"]))

    # ── 12. Residual + LayerNorm ──
    nodes.append(helper.make_node("Add",  ["norm1", "fo"], ["res2"]))
    nodes.append(helper.make_node("LayerNormalization",
        ["res2", "fln_w", "fln_b"], ["output"], epsilon=EPS))

    return helper.make_function(
        domain="bert.model",
        fname="BertEncoderLayer",
        inputs=inputs,
        outputs=["output"],
        nodes=nodes,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )


# ════════════════════════════════════════════════════════════
#  Main Graph
# ════════════════════════════════════════════════════════════

def build_main_graph():
    nodes = []
    vi = []                                       # value_info for edge shapes

    def add_vi(name, dtype, shape):
        vi.append(helper.make_tensor_value_info(name, dtype, shape))

    # ── Inputs ──
    model_inputs = [
        helper.make_tensor_value_info("input_ids",       TensorProto.INT64, [B, S]),
        helper.make_tensor_value_info("attention_mask",  TensorProto.INT64, [B, S]),
        helper.make_tensor_value_info("token_type_ids",  TensorProto.INT64, [B, S]),
        helper.make_tensor_value_info("position_ids",    TensorProto.INT64, [S]),
    ]

    # ── Embedding ──
    nodes.append(helper.make_node("Gather", ["emb_word", "input_ids"],
                                  ["word_emb"], axis=0, name="emb_word"))
    add_vi("word_emb", TensorProto.FLOAT, [B, S, HIDDEN])

    nodes.append(helper.make_node("Gather", ["emb_pos", "position_ids"],
                                  ["pos_emb"], axis=0, name="emb_pos"))
    add_vi("pos_emb", TensorProto.FLOAT, [S, HIDDEN])

    nodes.append(helper.make_node("Gather", ["emb_type", "token_type_ids"],
                                  ["type_emb"], axis=0, name="emb_type"))
    add_vi("type_emb", TensorProto.FLOAT, [B, S, HIDDEN])

    # Broadcasting: (B,S,H) + (S,H) → (B,S,H)  ✓
    nodes.append(helper.make_node("Add", ["word_emb", "pos_emb"], ["emb1"], name="emb_add_pos"))
    add_vi("emb1", TensorProto.FLOAT, [B, S, HIDDEN])
    nodes.append(helper.make_node("Add", ["emb1", "type_emb"], ["emb2"], name="emb_add_type"))
    add_vi("emb2", TensorProto.FLOAT, [B, S, HIDDEN])

    nodes.append(helper.make_node("LayerNormalization",
        ["emb2", "emb_ln_w", "emb_ln_b"], ["hidden0"],
        epsilon=EPS, name="emb_ln"))
    add_vi("hidden0", TensorProto.FLOAT, [B, S, HIDDEN])

    # ── 12 Encoder Layers  (model-local function calls) ──
    h = "hidden0"
    for l in range(LAYERS):
        p = f"L{l}"
        out = f"hidden{l+1}"
        nodes.append(helper.make_node(
            "BertEncoderLayer",
            [
                h, "attention_mask",
                f"{p}_q_w", f"{p}_q_b",
                f"{p}_k_w", f"{p}_k_b",
                f"{p}_v_w", f"{p}_v_b",
                f"{p}_ao_w", f"{p}_ao_b",
                f"{p}_aln_w", f"{p}_aln_b",
                f"{p}_fi_w", f"{p}_fi_b",
                f"{p}_fo_w", f"{p}_fo_b",
                f"{p}_fln_w", f"{p}_fln_b",
            ],
            [out],
            domain="bert.model",
            name=f"encoder/layer_{l}",
        ))
        add_vi(out, TensorProto.FLOAT, [B, S, HIDDEN])
        h = out

    # ── Output: last hidden state ──
    nodes.append(helper.make_node("Identity", [h], ["last_hidden_state"], name="output_lhs"))

    # ── Pooler ──
    # Gather CLS token (index 0) along seq axis → scalar index squeezes axis
    nodes.append(helper.make_node("Gather", [h, "zero_idx"], ["cls"], axis=1, name="pooler_gather"))
    add_vi("cls", TensorProto.FLOAT, [B, HIDDEN])

    nodes.append(helper.make_node("MatMul", ["cls", "pooler_w"], ["pooler_mm"], name="pooler_mm"))
    add_vi("pooler_mm", TensorProto.FLOAT, [B, HIDDEN])
    nodes.append(helper.make_node("Add", ["pooler_mm", "pooler_b"], ["pooler_d"], name="pooler_add"))
    add_vi("pooler_d", TensorProto.FLOAT, [B, HIDDEN])
    nodes.append(helper.make_node("Tanh", ["pooler_d"], ["pooler_output"], name="pooler_tanh"))

    # ── Outputs ──
    model_outputs = [
        helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [B, S, HIDDEN]),
        helper.make_tensor_value_info("pooler_output",     TensorProto.FLOAT, [B, HIDDEN]),
    ]

    return helper.make_graph(
        nodes=nodes,
        name="bert-base",
        inputs=model_inputs,
        outputs=model_outputs,
        initializer=init_list,
        value_info=vi,
    )


# ════════════════════════════════════════════════════════════
#  Assemble & Save
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building BERT-base with model-local functions ...")

    graph = build_main_graph()
    encoder_func = make_encoder_func()

    model = helper.make_model(
        graph,
        producer_name="onnx-api-manual",
        opset_imports=[
            helper.make_opsetid("", OPSET),
            helper.make_opsetid("bert.model", 1),
        ],
        functions=[encoder_func],
    )
    model.ir_version = 8

    # ── Validate ──
    print("Checking model validity ...")
    checker.check_model(model)
    print("  ✓ Check passed!")

    # ── Save ──
    onnx.save(model, "bert.onnx")
    print("  ✓ Saved bert.onnx")

    # ── Stats ──
    total_params = sum(int(np.prod(t.dims)) for t in model.graph.initializer)
    mb = total_params * 4 / 1024 / 1024
    main_nodes = len(model.graph.node)
    func_nodes = len(encoder_func.node)

    print(f"\n{'='*55}")
    print(f"  Model file:      bert.onnx")
    print(f"  Size:            {mb:.0f} MB  ({total_params:,} params)")
    print(f"  Approach:        Model-Local Function (reused {LAYERS}×)")
    print(f"  Main-graph ops:  {main_nodes}  (embedding + {LAYERS} layer calls + pooler)")
    print(f"  Function ops:    {func_nodes}  (inside each encoder layer)")
    print(f"  Total ops:       {main_nodes + func_nodes} unique, {main_nodes - LAYERS + LAYERS * func_nodes} executed")
    print(f"  Functions:       {[f.name for f in model.functions]}")
    print(f"{'='*55}")
    print(f"\n  ► Open bert.onnx in Netron:")
    print(f"    Top level shows: Embed → [BertEncoderLayer]×12 → Pooler")
    print(f"    Click any BertEncoderLayer block → see all {func_nodes} internal ops!")
