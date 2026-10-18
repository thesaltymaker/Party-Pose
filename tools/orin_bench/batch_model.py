"""Write a copy of a batch-1 landmark model with a free batch dimension, so several crops run in one model run.

    python tools/orin_bench/batch_model.py models/pose_landmarks_detector_opset15.onnx
    -> models/pose_landmarks_detector_opset15.dynbatch.onnx (the original file is kept)

The pose and hand models were exported with batch 1 on the input, the outputs and one Reshape (the pose model's
segmentation output, [1, 256, 256, 1]); everything else already works on any batch. Needs the `onnx` package
(on the Orin's env; on the laptop a separate venv).
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def make_dynamic(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    for v in list(g.input) + list(g.output):
        v.type.tensor_type.shape.dim[0].dim_param = 'N'
    del g.value_info[:]  # shapes inside were inferred for batch 1
    reshape_shapes = {n.input[1] for n in g.node if n.op_type == 'Reshape'}
    for t in g.initializer:
        if t.name in reshape_shapes:
            a = numpy_helper.to_array(t)
            if a.ndim == 1 and a.size > 1 and a[0] == 1:
                a = a.copy()
                a[0] = -1
                t.CopyFrom(numpy_helper.from_array(a.astype(np.int64), t.name))
    onnx.checker.check_model(model)
    return model


if __name__ == '__main__':
    src = sys.argv[1]
    dst = src[:-len('.onnx')] + '.dynbatch.onnx'
    onnx.save(make_dynamic(onnx.load(src)), dst)
    print(dst)
