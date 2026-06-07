#!/usr/bin/env python3
"""Extract shapes of every intermediate tensor from the ONNX model after shape inference."""
import sys
sys.path.insert(0, ".")

import numpy as np
import onnx
from onnx import shape_inference, TensorProto

# Rebuild the model using the same functions
from build_bert import build_main_graph, make_encoder_func, HIDDEN, HEADS, HEAD_DIM, LAYERS

print("Building model...")
graph = build_main_graph()
encoder_func = make_encoder_func()

model = onnx.helper.make_model(
    graph,
    producer_name="onnx-api-manual",
    opset_imports=[
        onnx.helper.make_opsetid("", 20),
        onnx.helper.make_opsetid("bert.model", 1),
    ],
    functions=[encoder_func],
)
model.ir_version = 8

print("Running shape inference...")
inferred = shape_inference.infer_shapes(model, strict_mode=False)

# Collect all shapes from value_info
shapes = {}
for vi in inferred.graph.value_info:
    if vi.type.HasField("tensor_type"):
        shape = vi.type.tensor_type.shape
        dims = []
        for d in shape.dim:
            if d.HasField("dim_param"):
                dims.append(d.dim_param)
            elif d.HasField("dim_value"):
                dims.append(str(d.dim_value))
            else:
                dims.append("?")
        shapes[vi.name] = dims

# Also add input/output shapes
for io in inferred.graph.input:
    if io.type.HasField("tensor_type"):
        shape = io.type.tensor_type.shape
        dims = []
        for d in shape.dim:
            if d.HasField("dim_param"):
                dims.append(d.dim_param)
            elif d.HasField("dim_value"):
                dims.append(str(d.dim_value))
            else:
                dims.append("?")
        shapes[io.name] = dims

for io in inferred.graph.output:
    if io.type.HasField("tensor_type"):
        shape = io.type.tensor_type.shape
        dims = []
        for d in shape.dim:
            if d.HasField("dim_param"):
                dims.append(d.dim_param)
            elif d.HasField("dim_value"):
                dims.append(str(d.dim_value))
            else:
                dims.append("?")
        shapes[io.name] = dims

# Also get node outputs
node_outputs = {}
for node in inferred.graph.node:
    for out in node.output:
        if out not in shapes:
            # Try to infer from node type
            pass

print(f"\n=== Value Info Shapes ({len(shapes)} tensors) ===\n")
formatted = {k: f"[{', '.join(v)}]" for k, v in shapes.items()}
for name in sorted(formatted.keys()):
    print(f"  {name:30s}  {formatted[name]}")

# Print all node outputs with their shapes
print(f"\n=== Node Output Mappings ===\n")
for node in inferred.graph.node:
    for out in node.output:
        s = formatted.get(out, "[unknown]")
        print(f"  {node.op_type:20s}  {out:30s}  {s}")

# Also handle function-local constants by reading the function
print(f"\n=== Function Constants (BertEncoderLayer) ===\n")
for node in encoder_func.node:
    if node.op_type == "Constant":
        for out in node.output:
            print(f"  Constant  {out:30s}")
