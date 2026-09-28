"""Calibrate and export MNIST-12 with the Vitis AI 2.5 PyTorch quantizer."""

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
import torch
from pytorch_nndct.apis import torch_quantizer

from mnist12_model import Mnist12


def load_mnist(images_path: Path, labels_path: Path):
    with gzip.open(images_path, "rb") as stream:
        magic, count, height, width = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected MNIST image magic number: {magic}")
        images = np.frombuffer(stream.read(), dtype=np.uint8).copy()
        images = images.reshape(count, 1, height, width).astype(np.float32)

    with gzip.open(labels_path, "rb") as stream:
        magic, label_count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected MNIST label magic number: {magic}")
        labels = np.frombuffer(stream.read(), dtype=np.uint8).copy()

    if label_count != count:
        raise ValueError(f"image/label count mismatch: {count} != {label_count}")
    return images, labels


def evaluate(model, images, labels, batch_size, limit):
    correct = 0
    total = min(limit, len(images))
    model.eval()
    with torch.no_grad():
        for offset in range(0, total, batch_size):
            batch = torch.from_numpy(images[offset : offset + batch_size])
            scores = model(batch)
            predictions = scores.argmax(dim=1).cpu().numpy()
            correct += int(np.sum(predictions == labels[offset : offset + len(batch)]))
    accuracy = correct / total
    print(f"evaluated={total} correct={correct} accuracy={accuracy:.6f}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("float", "calib", "test"), required=True)
    parser.add_argument("--root", type=Path, default=Path("/workspace"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    weights = args.root / "models" / "mnist12_float_weights.npz"
    images_path = args.root / "data" / "t10k-images-idx3-ubyte.gz"
    labels_path = args.root / "data" / "t10k-labels-idx1-ubyte.gz"
    output_dir = args.root / "build" / "quant_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    images, labels = load_mnist(images_path, labels_path)
    model = Mnist12(weights).cpu().eval()

    # The official INT8 ONNX file uses Input3_scale=1.0. Empirical validation
    # on all 10,000 test images confirms that it expects raw 0..255 floats.
    if args.mode == "float":
        evaluate(model, images, labels, args.batch_size, args.limit)
        return

    quant_batch_size = 1 if args.mode == "test" else args.batch_size
    dummy_input = torch.zeros((quant_batch_size, 1, 28, 28), dtype=torch.float32)
    quantizer = torch_quantizer(
        args.mode,
        model,
        dummy_input,
        output_dir=str(output_dir),
        device=torch.device("cpu"),
    )
    quantized_model = quantizer.quant_model.eval()
    evaluate(quantized_model, images, labels, quant_batch_size, args.limit)

    if args.mode == "calib":
        quantizer.export_quant_config()
    elif args.deploy:
        quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=True)


if __name__ == "__main__":
    main()

