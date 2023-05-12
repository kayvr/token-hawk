# TokenHawk

[LLaMA](https://arxiv.org/abs/2302.13971) inference using hand-written WebGPU code.

# Description

TokenHawk uses WebGPU to perform Llama inference. All code is written by hand and there are two files:

* th.cpp - Contains GPU shaders to support running LLMs.
* th-llama.cpp - GPU implementation of llama.

The command line version of TokenHawk is all native C++ code. It statically links to Google's C++ WebGPU library which makes profiling and debuging simple.

The Web UI version uses emcripten to cross-compile these two files into WASM.

[llama.cpp](https://github.com/ggerganov/llama.cpp) is currently used to load models and perform tokenization.

# Command Line

See the [CLI directory](cli/README.md) for build and usage instructions.

Use the command line for performance tuning WebGPU code. Here's an example of usage:

```
$ ./th -m models/llama-7B/ggml-model-f16.bin "<prompt goes here>"
```

# Web UI

See the [Web directory](web/README.md) for build and usage instructions.

For simple and quick access, use the Web UI. You can try it out online here, or host it locally:

```
python web/serve.py
```

# Performance

While fast, TokenHawk underperforms CUDA. There is a a lot of room for optimization.

Next areas of focus:

* Profile and optimize matrix multiplication.
* Optimize single token generation.
* Investigate feasibility of GPU-only operation. Not hitting the CPU.
* Investigate native f16 support (currently emulated in shaders).
* Add 4-bit quantization.

Using an RTX 4090, TokenHawk executes 10 tokens per second using a 7B parameter f16 model. The original CUDA implementation of llama yields 50 tokens per second. We should be able to get close to CUDA performance.

## Data

More data to come.

# Acknowledgments

Thanks to [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGML, tokenization, and its file format.

And Google's [Dawn](https://dawn.googlesource.com/dawn) for the WebGPU implementation.
