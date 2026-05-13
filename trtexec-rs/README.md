# trtexec-rs

This is a small tool that mimics [https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec).

While the real `trtexec` runs ONNX files, `trtexec-rs` can run ONNX files or WebNN files [https://github.com/rustnn/webnn-graph/](https://github.com/rustnn/webnn-graph/).
By leveraging [RustNN]([https://github.com/rustnn/rustnn/]) for the WebNN to trtx conversion we can test
whether any changes in this repo would break RustNN.
It's also possible to compare ONNX -> TRT vs ONNX -> WebNN -> TRT.

## Installation

Run the following command in the root directory of this repo

```bash
cargo install trtexec-rs
```

or to just run the sample

```bash
cargo r -- --help
```

## Usage

The CLI arguments are not identical to the original.

```
Usage: trtexec-rs [OPTIONS] --engine-dir <ENGINE_DIR> [INPUTS]...

Arguments:
  [INPUTS]...
          ONNX files or WebNN files to compile and execute

Options:
      --engine-dir <ENGINE_DIR>
          Directory to store/cache engines

  -s, --save-engine <SAVE_ENGINE>
          Save engine to file path

      --shell-completion <SHELL_COMPLETION>
          Generate shell completion to stdout
          
          [possible values: bash, elvish, fish, powershell, zsh]

  -n, --num-inferences <NUM_INFERENCES>
          Number of inferences
          
          [default: 100]

      --cuda-graph
          Use cuda graphs

      --nvtx
          Set extra nvtx spans

      --no-cache
          Don't use engines from cache

      --skip-inference
          Don't run inference

      --cuda-device-idx <CUDA_DEVICE_IDX>
          Which CUDA device index to use
          
          [default: 0]

      --non-interactive
          Fail on dynamic input shapes instead of prompting (for scripts/CI)

  -a, --api-capture <API_CAPTURE>
          Capture TensorRT API usage to JSON for possible later replay
          
          See https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/capture-replay.html#capture-replay

      --report-layer-time
          Report layer time

      --profile-json <PATH>
          Write TensorRT layer timing profile JSON (trtexec-style); implies layer profiling

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
