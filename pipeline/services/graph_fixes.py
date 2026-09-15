"""
Graph rewrites that keep a model on the GPU.

ONNX Runtime keeps the CPU provider registered behind CUDA and silently places
any node CUDA cannot run on it — with a device round trip on either side. The
session still reports `CUDAExecutionProvider`, `execution.verify` still
passes, and the model quietly costs several times what it should. Measured
2026-09-15 on the RTX 5880 Ada: the mask stage was 18ms of a 67ms frame, and
six of XSeg's 362 nodes were on the CPU — all six `ConvTranspose` layers, the
decoder's upsampling, because a TensorFlow export gives them **asymmetric
padding** (`pads = [0, 0, 1, 1]`) and cuDNN's transposed convolution wants
symmetric.

The rewrite here is lossless: a ConvTranspose with asymmetric pads equals the
same op with zero pads — whose output is larger by exactly the padding — then
a `Slice` that drops the rows and columns the pads would have removed.

    out[i] = (in[i] - 1) * stride - (pad_begin + pad_end) + kernel + output_padding
    with pads zero:   full = out + pad_begin + pad_end
    slice [pad_begin : full - pad_end]  ==  out

Every value the original produced, bit for bit in exact arithmetic, and both
ops run on CUDA. The result is written beside the original as
`<name>-cuda.onnx` and preferred when present, the way the fp16 copy is: the
original is never touched, and reverting is deleting a file.
"""

import os
from typing import Any, List, Optional, Tuple

from pipeline.logging import emit_status, emit_warning


def cuda_sibling(model_path: str) -> str:
    """Where the rewritten copy of `model_path` lives."""
    root, ext = os.path.splitext(model_path)
    return root + '-cuda' + ext


def symmetrise_convtranspose(src: str, dst: str) -> Optional[int]:
    """
    Rewrite every ConvTranspose with asymmetric pads as zero pads + Slice.

    Args:
        src: The model to read
        dst: Where to write the rewritten model. Not written when nothing
            needed rewriting

    Returns:
        The number of nodes rewritten, 0 when the graph was already symmetric,
        or None when the rewrite could not be attempted (no `onnx` package,
        unreadable file) — never raises into the caller's load path
    """
    try:
        import onnx
        from onnx import helper, numpy_helper
        import numpy as np
    except ImportError:
        emit_warning(
            'onnx is not installed; XSeg keeps its asymmetric ConvTranspose '
            'pads and those six layers run on the CPU', scope='MASKER')
        return None

    try:
        model = onnx.load(src)
    except Exception as e:
        emit_warning(f'Could not read {src} for rewriting: {e}', scope='MASKER')
        return None

    graph = model.graph
    rewritten = 0
    new_nodes: List[Any] = []
    counter = 0

    for node in graph.node:
        if node.op_type != 'ConvTranspose':
            new_nodes.append(node)
            continue

        pads: Optional[List[int]] = None
        for attribute in node.attribute:
            if attribute.name == 'pads':
                pads = list(attribute.ints)
        if not pads:
            new_nodes.append(node)
            continue
        rank = len(pads) // 2
        begin, end = pads[:rank], pads[rank:]
        if begin == end:
            new_nodes.append(node)
            continue

        # Zero the pads on the node itself and route its output through a
        # Slice that removes what the pads would have.
        fixed = onnx.NodeProto()
        fixed.CopyFrom(node)
        for attribute in fixed.attribute:
            if attribute.name == 'pads':
                attribute.ints[:] = [0] * (2 * rank)
        original_output = node.output[0]
        full_output = original_output + '_unpadded'
        fixed.output[0] = full_output

        # Spatial axes follow N and C.
        axes = [2 + i for i in range(rank)]
        starts = [int(b) for b in begin]
        # An end of `-pad_end` counts from the tensor's own extent, so the
        # Slice needs no knowledge of the output shape. A zero pad_end is the
        # whole axis, which `INT64_MAX` expresses without a shape.
        ends = [(-int(e) if e > 0 else np.iinfo(np.int64).max) for e in end]
        counter += 1
        prefix = f'{node.name or "convtranspose"}_sym{counter}'
        starts_init = numpy_helper.from_array(np.array(starts, dtype=np.int64), prefix + '_starts')
        ends_init = numpy_helper.from_array(np.array(ends, dtype=np.int64), prefix + '_ends')
        axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), prefix + '_axes')
        graph.initializer.extend([starts_init, ends_init, axes_init])

        slicer = helper.make_node(
            'Slice',
            inputs=[full_output, starts_init.name, ends_init.name, axes_init.name],
            outputs=[original_output],
            name=prefix + '_slice',
        )
        new_nodes.extend([fixed, slicer])
        rewritten += 1

    if rewritten == 0:
        return 0

    del graph.node[:]
    graph.node.extend(new_nodes)
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        emit_warning(f'Rewritten XSeg graph failed the ONNX checker: {e}', scope='MASKER')
        return None
    onnx.save(model, dst)
    emit_status(
        f'Rewrote {rewritten} ConvTranspose node(s) with asymmetric pads as zero '
        f'pads + Slice so they run on CUDA: {os.path.basename(dst)}',
        scope='MASKER')
    return rewritten


def prefer_cuda_copy(model_path: str) -> Tuple[str, bool]:
    """
    The path to load: the rewritten sibling when it exists or can be made.

    Args:
        model_path: The original weights

    Returns:
        (path to load, whether it is the rewritten copy)
    """
    sibling = cuda_sibling(model_path)
    if os.path.isfile(sibling):
        return sibling, True
    if not os.path.isfile(model_path):
        return model_path, False
    rewritten = symmetrise_convtranspose(model_path, sibling)
    if rewritten and os.path.isfile(sibling):
        return sibling, True
    return model_path, False
