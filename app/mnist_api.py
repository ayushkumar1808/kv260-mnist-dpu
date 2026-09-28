#!/usr/bin/env python3
"""Small HTTP inference service for the KV260 MNIST DPU model."""

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import vart
import xir


EXPECTED_FINGERPRINT = 0x101000016010406
EXPECTED_INPUT_SHAPE = (1, 28, 28, 1)
MAX_REQUEST_BYTES = 32 * 1024


def get_dpu_subgraph(graph):
    children = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_children = [
        child
        for child in children
        if child.has_attr("device") and child.get_attr("device").upper() == "DPU"
    ]
    if len(dpu_children) != 1:
        raise RuntimeError(
            "expected one DPU subgraph, found {}".format(len(dpu_children))
        )
    return dpu_children[0]


class MnistDpu:
    def __init__(self, model_path: Path):
        self.graph = xir.Graph.deserialize(str(model_path))
        self.subgraph = get_dpu_subgraph(self.graph)
        self.fingerprint = int(self.subgraph.get_attr("dpu_fingerprint"))
        if self.fingerprint != EXPECTED_FINGERPRINT:
            raise RuntimeError(
                "model fingerprint 0x{:x} does not match SmartCam DPU 0x{:x}".format(
                    self.fingerprint, EXPECTED_FINGERPRINT
                )
            )

        self.runner = vart.Runner.create_runner(self.subgraph, "run")
        self.input_tensor = self.runner.get_input_tensors()[0]
        self.output_tensor = self.runner.get_output_tensors()[0]
        self.input_shape = tuple(self.input_tensor.dims)
        self.output_shape = tuple(self.output_tensor.dims)
        self.input_fix = int(self.input_tensor.get_attr("fix_point"))
        self.output_fix = int(self.output_tensor.get_attr("fix_point"))
        self.input_scale = 2 ** self.input_fix
        self.dpu_name = str(self.subgraph.get_attr("dpu_name"))

        if self.input_shape != EXPECTED_INPUT_SHAPE:
            raise RuntimeError("unexpected DPU input shape: {}".format(self.input_shape))
        if self.output_shape != (1, 10):
            raise RuntimeError("unexpected DPU output shape: {}".format(self.output_shape))

        self.input_buffer = np.empty(self.input_shape, dtype=np.int8, order="C")
        self.output_buffer = np.empty(self.output_shape, dtype=np.int8, order="C")
        self.lock = threading.Lock()

        # Initialize buffers and the hardware execution path once at startup.
        self.predict(np.zeros((28, 28), dtype=np.uint8))

    def predict(self, pixels):
        image = np.asarray(pixels, dtype=np.uint8)
        if image.size != 28 * 28:
            raise ValueError("pixels must contain exactly 784 grayscale values")
        image = image.reshape(28, 28, 1)

        with self.lock:
            self.input_buffer[0] = (
                image.astype(np.float32) * self.input_scale
            ).astype(np.int8)
            started_ns = time.perf_counter_ns()
            job_id = self.runner.execute_async(
                [self.input_buffer], [self.output_buffer]
            )
            self.runner.wait(job_id)
            inference_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            scores = [int(value) for value in self.output_buffer[0].tolist()]

        return {
            "prediction": int(np.argmax(scores)),
            "scores": scores,
            "inference_ms": round(inference_ms, 3),
            "execution_device": "DPU_in_PL",
            "dpu_arch": self.dpu_name,
            "fingerprint": "0x{:x}".format(self.fingerprint),
            "input_shape": list(self.input_shape),
            "input_fix_point": self.input_fix,
            "output_fix_point": self.output_fix,
        }

    def health(self):
        return {
            "ready": True,
            "execution_device": "DPU_in_PL",
            "dpu_arch": self.dpu_name,
            "fingerprint": "0x{:x}".format(self.fingerprint),
            "input_shape": list(self.input_shape),
        }


class MnistServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, model, index_path):
        super().__init__(address, handler)
        self.model = model
        self.index_path = index_path


class Handler(BaseHTTPRequestHandler):
    server_version = "KV260MNIST/1.0"

    def send_json(self, status, payload):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                body = self.server.index_path.read_bytes()
            except OSError as error:
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "cannot read interface: {}".format(error)},
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/health":
            self.send_json(HTTPStatus.OK, self.server.model.health())
            return

        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            pixels = payload.get("pixels")
            if not isinstance(pixels, list) or len(pixels) != 784:
                raise ValueError("pixels must be a list containing 784 values")
            if any(
                not isinstance(value, int) or value < 0 or value > 255
                for value in pixels
            ):
                raise ValueError("each pixel must be an integer from 0 to 255")
            result = self.server.model.predict(pixels)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:  # Keep malformed requests from killing service.
            self.log_error("prediction failed: %s", error)
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "DPU inference failed: {}".format(error)},
            )
            return

        self.send_json(HTTPStatus.OK, result)

    def log_message(self, message_format, *args):
        print(
            "{} - {}".format(self.client_address[0], message_format % args),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    model = MnistDpu(args.model)
    server = MnistServer((args.bind, args.port), Handler, model, args.index)
    print(
        "KV260 MNIST interface ready on http://{}:{} using {}".format(
            args.bind, args.port, model.dpu_name
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
