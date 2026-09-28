# MNIST digit recognition on KV260

A browser drawing pad sends a 28 × 28 grayscale image to a service on the
Kria KV260. The service runs the quantized MNIST-12 model on the SmartCam
B3136 DPU and returns the predicted digit and inference timing.

The saved v1.1 release records a working hardware deployment. This repository
captures the latest local application, model conversion code and model assets.
Publication did not repeat the hardware test.

## Files

- `app/`: browser UI and Python HTTP inference service.
- `models/`: source/intermediate models and floating-point weights.
- `build/compiled/`: compiled DPU model.
- `build/quant_model/`: quantization output and deployment checks.
- `quantize_vitis.py`: Vitis AI calibration and export.
- `run_mnist_dpu.py`: dataset evaluation through VART/XIR.
- `systemd/`: service configuration from the deployed board.

## Runtime

The target requires the SmartCam `DPUCZDX8G_ISA1_B3136` overlay, NumPy, and
compatible VART/XIR libraries. The expected DPU fingerprint is
`0x101000016010406`. The original environment used Vitis AI 2.5 quantization
and a VART 5 runtime adapted for Canonical XRT 2.18.

Inspect `app/mnist_api.py --help` and `run_mnist_dpu.py --help` in the target
environment for model and listen-address arguments. The supplied service and
`open_mnist_ui.sh` retain the original board paths; adjust them for your board.
Tool installations, the SmartCam platform binary and private connection setup
are not part of this repository. See `PUBLICATION.md` for snapshot scope.
