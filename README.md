# TokenHawk

[LLaMA](https://arxiv.org/abs/2302.13971) inference using WebGPU.

# Description

TokenHawk uses WebGPU to perform Llama inference. Contained in this repo is a native (C++) implementation that relies on Google's Dawn library for CLI access. Examples for cross-compiling to the web are also provided.

There are two files:

* th.cpp - Contains GPU shaders to support running LLMs.
* th-llama.cpp - GPU implementation of llama.

In addition to Dawn, [llama.cpp](https://github.com/ggerganov/llama.cpp) is used to load models in ggml format and perform tokenization.

# Usage

## Command Line

```
$ ./th -m models/llama-7B/ggml-model-f16.bin "<prompt goes here>"
```

## Web UI

To run locally, use:

```
python serve.py
```

A precompiled WASM library is provided. So if you have a WebGPU compatible browser this should work out-of-the-box.

Or, go to this github page and try it out.

# Building

## Command Line

```
```

## Web UI

The Web UI requires emscripten to cross compile.

```
```

# Performance

There's still a ton of room for optimization.

* Profile and optimize matrix multiplication.
* Optimize single token generation.
* Investigate feasibility of GPU-only operation without hitting the CPU.

An RTX 4090 executes 10 tokens per second using a 7B parameter f16 model. The original CUDA implementation of llama yields 50 tokens per second. While we may not be able to reach CUDA-levels of performance, we should be able to get close.

