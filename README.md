# TokenHawk

[LLaMA](https://arxiv.org/abs/2302.13971) inference using hand-written WebGPU code.

# Description

TokenHawk uses WebGPU to perform Llama inference. Contained in this repo is a native (C++) implementation that relies on Google's Dawn library for CLI access. Examples for cross-compiling to the web are also provided.

There are two files:

* th.cpp - Contains GPU shaders to support running LLMs.
* th-llama.cpp - GPU implementation of llama.

In addition to Dawn, [llama.cpp](https://github.com/ggerganov/llama.cpp) is used to load models in ggml format and perform tokenization.

# Usage

## Command Line

Use the command line for performance tuning WebGPU code. It builds the native C++ Dawn library directly into the generated binary making it easier to debug and profile.

```
$ ./th -m models/llama-7B/ggml-model-f16.bin "<prompt goes here>"
```

## Web UI

For simple and quick access, use the Web UI. You can try it out online here, or host it locally:

```
python web/serve.py
```

# Building

## Command Line

See the [CLI directory](web/README.md).

## Web UI

See the [Web directory](web/README.md).

# Performance

There's still a ton of room for optimization.

* Profile and optimize matrix multiplication.
* Optimize single token generation.
* Investigate feasibility of GPU-only operation without hitting the CPU.

An RTX 4090 executes 10 tokens per second using a 7B parameter f16 model. The original CUDA implementation of llama yields 50 tokens per second. While we may not be able to reach CUDA-levels of performance, we should be able to get close.

