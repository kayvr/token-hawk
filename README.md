# TokenHawk

Hand-written [LLaMA](https://arxiv.org/abs/2302.13971) inference using WebGPU. It's fast, and you can try it out [online](https://ui.tokenhawk.chat/). It also works on the Windows, macOS, or Linux command line.

⚠️  TokenHawk is under active development. Only llama 7B-f16 is supported.  ⚠️

# Description

WebGPU powers TokenHawk's LLM inference, and there are only three files:

* th.cpp - Provides WebGPU support for running LLMs.
* th-llama.cpp - GPU implementation of llama.
* th-llama-loader.cpp - Routines to load model files.

Dependencies are also minimal. For the command line app, TokenHawk requires only Google's Dawn library. And the Web app has no dependencies.

TokenHawk also integrates well with other WebGPU apps, like this one:




https://github.com/kayvr/token-hawk/assets/98552926/57ef453d-5ec9-4d81-85a0-36ef0313a51b




You'll find the app [here](https://tokenhawk.chat).

# Command Line

See the [CLI directory](cli/README.md) for build and usage instructions.

Use the command line for performance tuning WebGPU code. Here's an example of usage:

```
$ ./th -m models/llama-7B/ggml-model-f16.bin "<prompt goes here>"
```



https://github.com/kayvr/token-hawk/assets/98552926/6ba95cbe-2c1f-41bb-a4fb-4e502c7ef440



# Web UI

See the [Web directory](web/README.md) for build and usage instructions.

For simple and quick access, use the [online demo](https://ui.tokenhawk.chat/).




https://github.com/kayvr/token-hawk/assets/98552926/c718bc1d-2246-43de-9e8c-7efb5e5d7c49




## How to load models into the WebUI?

Now that chunking has been [removed](https://github.com/kayvr/token-hawk/issues/2), simply click on 'Load Model'. From there you will be directed to select a file to upload. Select a 7B-f16 GGML file to upload as that is currently the only format supported.

## How do I convert llama weights into a GGML file?

Download [llama.cpp](https://github.com/ggerganov/llama.cpp) and follow its documentation and utilities to convert llama weights. More details to come.

# Performance

TokenHawk is fast. On a 4090 using 7B-f16, TokenHawk clocks in at 37 tk/s. And there is still room for improvement. Here are single-token timings for the original 7B, f16 llama dataset (as of May 17th, 2023):

| Video Card          | llama-pytorch (tk/s) | TokenHawk | llama.cpp-Cuda | llama.cpp-CPU |
| ------------------- | -------------------- | --------- | -------------- | ------------- |
| Nvidia 4090 (lin)   | 46                   | 37        | 19             | 5             |
| RX 7900 XTX (win)   | (unsupported)        | 36 (!)    | (unsupported)  |               |

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

While TokenHawk's single-token performance is good, its batch processing of prompt input isn't as fast as it could be. This is due to a suboptimal matrix multiplication. 

# Limitations

Given that TokenHawk is new and something of a prototype, there's a few gaps in it's features:

* Needs VRAM. TokenHawk runs entirely on the GPU.
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
