"""Make a PINTO model zoo YOLOX-Body-Head-Hand `_post_` model (426_YOLOX-Body-Head-Hand, NCHW input) usable as
`poser.py --detector FILE`: a Transpose in front gives it the NHWC input [1, H, W, 3] named 'input' that
PersonDetector and GpuDetectorInput feed, and a `.shapeinferred.onnx` copy (ONNX Runtime's symbolic shape
inference) is written next to it for the Orin's TensorRT build, as for the default detector (commit 42045e1).

  python tools/orin_bench/detector_nhwc.py ZOO_MODEL.onnx models/OUT.onnx
Needs the onnx package (the Orin's env has it). Copy both files to models/ on each machine.
"""
import subprocess
import sys

import onnx
from onnx import helper


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    graph = model.graph
    old = graph.input[0]
    n, c, h, w = [d.dim_value for d in old.type.tensor_type.shape.dim]
    if c != 3:
        raise SystemExit(f'{src}: input is not [N, 3, H, W]')
    old.name = 'input_nchw'
    for node in graph.node:
        node.input[:] = ['input_nchw' if i == 'input' else i for i in node.input]
    new = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [n, h, w, 3])
    graph.input.remove(old)
    graph.input.insert(0, new)
    graph.node.insert(0, helper.make_node('Transpose', ['input'], ['input_nchw'], perm=[0, 3, 1, 2],
                                          name='input_nhwc_to_nchw'))
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    inferred = dst[:-len('.onnx')] + '.shapeinferred.onnx'
    subprocess.run([sys.executable, '-m', 'onnxruntime.tools.symbolic_shape_infer',
                    '--input', dst, '--output', inferred], check=True)
    print(f'{dst}: input [1, {h}, {w}, 3]; {inferred}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
