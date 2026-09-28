#!/usr/bin/env python3
"""Run the compiled MNIST-12 network on the KV260 SmartCam DPU."""

import argparse
import gzip
import struct
import time
from pathlib import Path

import numpy as np
import vart
import xir


EXPECTED_FINGERPRINT = 0x101000016010406


def load_mnist(images_path: Path, labels_path: Path, limit: int):
    with gzip.open(str(images_path), "rb") as stream:
        magic, count, height, width = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise RuntimeError("invalid MNIST image file")
        images = np.frombuffer(stream.read(), dtype=np.uint8)
        images = images.reshape(count, height, width, 1)

    with gzip.open(str(labels_path), "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise RuntimeError("invalid MNIST label file")
        labels = np.frombuffer(stream.read(), dtype=np.uint8)

    limit = min(limit, len(images), len(labels))
    return images[:limit], labels[:limit]


def get_dpu_subgraph(graph):
    children = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_children = [
        child
        for child in children
        if child.has_attr("device") and child.get_attr("device").upper() == "DPU"
    ]
    if len(dpu_children) != 1:
        raise RuntimeError("expected one DPU subgraph, found {}".format(len(dpu_children)))
    return dpu_children[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument(
        "--show-predictions",
        type=int,
        default=10,
        help="print this many individual predicted and expected digits",
    )
    args = parser.parse_args()

    graph = xir.Graph.deserialize(str(args.model))
    subgraph = get_dpu_subgraph(graph)
    fingerprint = subgraph.get_attr("dpu_fingerprint")
    if fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(
            "model fingerprint 0x{:x} does not match SmartCam B3136 0x{:x}".format(
                fingerprint, EXPECTED_FINGERPRINT
            )
        )

    runner = vart.Runner.create_runner(subgraph, "run")
    input_tensor = runner.get_input_tensors()[0]
    output_tensor = runner.get_output_tensors()[0]
    input_shape = tuple(input_tensor.dims)
    output_shape = tuple(output_tensor.dims)
    input_fix = input_tensor.get_attr("fix_point")
    output_fix = output_tensor.get_attr("fix_point")
    input_scale = 2 ** input_fix

    if input_shape != (1, 28, 28, 1):
        raise RuntimeError("unexpected DPU input shape: {}".format(input_shape))

    images, labels = load_mnist(args.images, args.labels, args.limit)
    input_buffer = np.empty(input_shape, dtype=np.int8, order="C")
    output_buffer = np.empty(output_shape, dtype=np.int8, order="C")

    # Warm the runner without including initialization in the throughput number.
    warmup = min(20, len(images))
    for index in range(warmup):
        input_buffer[0] = (images[index].astype(np.float32) * input_scale).astype(np.int8)
        job_id = runner.execute_async([input_buffer], [output_buffer])
        runner.wait(job_id)

    correct = 0
    started = time.perf_counter()
    for index, image in enumerate(images):
        input_buffer[0] = (image.astype(np.float32) * input_scale).astype(np.int8)
        job_id = runner.execute_async([input_buffer], [output_buffer])
        runner.wait(job_id)
        prediction = int(np.argmax(output_buffer[0]))
        expected = int(labels[index])
        correct += int(prediction == expected)
        if index < args.show_predictions:
            print(
                "sample={} predicted_digit={} expected_digit={}".format(
                    index, prediction, expected
                )
            )
    elapsed = time.perf_counter() - started

    print("execution_device=DPU_in_PL")
    print("dpu_arch={}".format(subgraph.get_attr("dpu_name")))
    print("fingerprint=0x{:x}".format(fingerprint))
    print("input={} dtype=int8 fix_point={}".format(input_shape, input_fix))
    print("output={} dtype=int8 fix_point={}".format(output_shape, output_fix))
    print("images={} correct={} accuracy={:.6f}".format(len(images), correct, correct / len(images)))
    print("inference_seconds={:.6f} throughput_fps={:.2f}".format(elapsed, len(images) / elapsed))


if __name__ == "__main__":
    main()
