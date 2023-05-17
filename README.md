# TokenHawk

Hand-written [LLaMA](https://arxiv.org/abs/2302.13971) inference using WebGPU.

TokenHawk is very new. Time will be needed to get to feature parity. Please see the limitations section.

# Description

WebGPU powers TokenHawk's LLM inference. There are three core files:

* th.cpp - Provides GPU support for running LLMs.
* th-llama.cpp - GPU implementation of llama.
* th-llama-loader.cpp - Routines to load model files.

Native C++ is used for command line (CLI) TokenHawk. This CLI version statically links Google's C++ WebGPU library, which is the only dependency.

The Web UI version uses emcripten to cross-compile the C++ code into WASM. Other than emscripten, there are no dependencies.

TokenHawk aims to be reproducible from run-to-run. TokenHawk's output has been verified against the original llama implementation.

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

TokenHawk is fast. On a 4090 using 7B-f16, TokenHawk clocks in at 37 tk/s and there is still room for improvement. Here are single-token timings for the original 7B, f16 llama dataset (as of May 17th, 2023):

| Video Card          | llama-pytorch | TokenHawk | llama.cpp-Cuda | llama.cpp-CPU |
| ------------------- | ------------- | --------- | -------------- | ------------- |
| Nvidia 4090 (lin)   | 46 (tk/s)     | 37 (tk/s) | 19 (tk/s)      | 5 (tk/s)      |
| RX 7900 XTX (win)   | (unsupported) | 36 (!)    | (unsupported)  |               |

All tests were executed on the GPU, except for llama.cpp-CPU. In the case of llama.cpp-Cuda, all layers were loaded onto the GPU using `-ngl 32`.

We'll focus on the following perf improvements in the coming weeks:

* Profile and optimize matrix multiplication.
* Further optimize single token generation.
* Optimize WARP and Wavefront sizes for Nvidia and AMD.
* Per-GPU hyper-parameter optimization.
* Investigate native f16 support. f16 is currently emulated in shaders.
* Store intermediate GPU buffers in fp16. Specifically the context and working buffers.
* Add 8-bit and 4-bit quantization.
* Optimize transpose. Can we ensure better memory coalescing?

## Batch Prompt Performance

While TokenHawk's single-token performance is good, it's batch processing of prompt input isn't nearly as fast as it could be. This is due to a suboptimal matrix multiplication. 

# Limitations

Given that TokenHawk is new and something of a prototype, there's a few gaps in it's features:

* Needs more VRAM. TokenHawk runs entirely on the GPU.
* Only 512 token context has been tested. Theoretically higher context lengths are supported but have not been tested.
* Only tested 7B llama. Other models that follow the 7B llama architecture should work, but have not been tested.
* GGML is the only file format supported.

# See Also

I would like to include some performance measurements from the following projects.

* [Triton](https://github.com/openai/triton). OpenAI's triton compiler.
* [mlc-llm](https://github.com/mlc-ai/mlc-llm). LLMs running using hardware acceleration. I would have gathered timings from mlc-llm as it has WebGPU support but I was unable to get it running locally.

# Acknowledgments

A big thanks to [llama.cpp](https://github.com/ggerganov/llama.cpp). Beyond inspiration, I have borrowed and adapted a number of functions from this project including it's tokenizer and model loading functions.

And Google's [Dawn](https://dawn.googlesource.com/dawn) for the WebGPU implementation.
