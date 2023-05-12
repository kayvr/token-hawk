# TokenHawk

[LLaMA](https://arxiv.org/abs/2302.13971) inference using hand-written WebGPU code.

# Description

TokenHawk uses WebGPU to perform Llama inference. All code is written by hand and there are two files:

* th.cpp - Contains GPU shaders to support running LLMs.
* th-llama.cpp - GPU implementation of llama.

The command line version of TokenHawk is native C++ code. It statically links to Google's C++ WebGPU library which makes profiling and debuging simpler.

The Web UI version uses emcripten to cross-compile these two files into WASM.

[llama.cpp](https://github.com/ggerganov/llama.cpp) is currently used to load models and perform tokenization.

As of, May 13, 2023, only 7B llama models are supported. Wider model support should evolve quickly.

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

TokenHawk is pretty fast. On a 4090 using 7B-f16, TokenHawk clocks in at 30 tk/s while CUDA is 50 tk/s. And there is still room for improvement. We'll focus on the following perf improvements in the coming weeks:

* Profile and optimize matrix multiplication.
* Optimize single token generation.
    * Add a two-stage parallel reduction step.
* Optimize WARP and Wavefront sizes for Nvidia and AMD.
* Per-GPU hyper-parameter optimization.
* Investigate feasibility of GPU-only operation. No hitting the CPU.
* Investigate native f16 support. f16 is currently emulated in shaders.
* Store intermediate GPU buffers in fp16. Specifically the context and working buffers.
* Add 4-bit quantization.

## Data

More data to come.

# See Also

## Compilers

While TokenHawk focuses on hand-tuning models, here are compiler projects that aim to automatically generate GPU code for models.

* [Triton](https://github.com/openai/triton). OpenAI's triton compiler.
* [mlc-llm](https://github.com/mlc-ai/mlc-llm). LLMs running using hardware acceleration.

# Acknowledgments

Thanks to [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGML, tokenization, and its file format.

And Google's [Dawn](https://dawn.googlesource.com/dawn) for the WebGPU implementation.
