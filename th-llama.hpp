#pragma once

// The token-hawk llama ram is backed by llama.cpp.

#include "th.hpp"

#include <stdint.h>
#include <string>
#include <memory>
#include <vector>
#include <random>

struct llama_context; // From GGML's llama.h.

namespace th {

struct LlamaModel;
std::shared_ptr<LlamaModel> load_llama(WGPUDevice device, WGPUQueue queue, const std::string& filename, int32_t n_batch_tokens);

struct ThLlamaParameters {
    std::string modelFile = "models/7B/ggml-model-f16.bin";
    std::string prompt = "";
};

struct LlamaLayer {
    int64_t index{};

    TensorBuffer attention_norm{};

    TensorBuffer wq{};
    TensorBuffer wk{};
    TensorBuffer wv{};
    TensorBuffer wo{};

    TensorBuffer ffn_norm{};

    TensorBuffer w1{};
    TensorBuffer w2{};
    TensorBuffer w3{};

    TensorBuffer key_cache{};
    TensorBuffer value_cache{};
};

struct LlamaLayerComputePipeline {
    ComputePipeline p01{true}; // RMS
    ComputePipeline p02{true}; // Row norm
    ComputePipeline p03_mm{true};
    ComputePipeline p04_rope{true};
    ComputePipeline p05_trans{true};
    ComputePipeline p06_mm{true};
    ComputePipeline p07_softmax{true};
    ComputePipeline p08_mm{true};
    ComputePipeline p09_t{true};
    ComputePipeline p10_mm{true};
    ComputePipeline p11_add{true};
    ComputePipeline p12_rms{true};
    ComputePipeline p13_norm{true};
    ComputePipeline p14_mm{true};
    ComputePipeline p15_silu{true};
    ComputePipeline p16_hadamard{true};
    ComputePipeline p17_mm{true};
    ComputePipeline p18_add{true};
};

struct LlamaFinalComputePipeline {
    ComputePipeline p01{true};
    ComputePipeline p02{true};
    ComputePipeline p03{true};
};

struct LlamaModel {
    LlamaModel(llama_context* ctx);
    ~LlamaModel(); // Because of the raw llama_context pointer below. Should disallow copying.

    std::mt19937 rng{};

    // Hyper parameters. Defaults are 7B llama model.
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t n_batch = 8;  // Tokens batch.
    int32_t f16     = 1;

    int64_t n_past  = 0; // Number of tokens ingested so far.

    // Tensors.
    TensorBuffer tok_embeddings{};    // 
    std::vector<LlamaLayer> layers{};

    LlamaLayerComputePipeline ps{}; // Single token pipeline.
    LlamaLayerComputePipeline pb{}; // Batch token pipeline.

    LlamaFinalComputePipeline pfs{}; // Single token pipeline.
    LlamaFinalComputePipeline pfb{}; // Batch token pipeline

    TensorBuffer norm{};
    TensorBuffer outputMat{};

    TensorBuffer out{}; // Destination for NN output.
    
    // Buffers are (batch x n_embd) size.
    static const int nInpBuffers = 7;
    TensorBuffer inp[nInpBuffers]{}; // 1 x Batch x n_embd

    // Feed-forward working buffers.
    TensorBuffer ffWorking[2]{};
    
    TensorBuffer working_key_cache{};
    TensorBuffer working_val_cache{};

    // TODO Clean these buffers up!
    WGPUBuffer networkUniforms{}; // n_past and context size.
    std::array<WGPUBuffer, 5> dimsUniforms{}; // Uniform buffers for different.

    // llama.cpp context for RAM backing.
    llama_context* llamacpp_context{};
};

static const size_t kLlamaUniformsSize = 32;
struct LlamaNetworkUniforms {
    uint32_t n_past{};
    uint32_t n_tokens{};
    float padding02{};
    float padding03{};
};
static_assert(sizeof(LlamaNetworkUniforms) <= kLlamaUniformsSize, "Uniforms too large");

struct LlamaTensorDimsUniforms {
    uint32_t A_B{};
    uint32_t A_M{};
    uint32_t A_N{};
    float scale{};
    uint32_t B_B{};
    uint32_t B_M{};
    uint32_t B_N{};
    float offset{};
};
static_assert(sizeof(LlamaTensorDimsUniforms) <= kLlamaUniformsSize, "Uniforms too large");

void reset_layer_tensors(LlamaLayer& layer);
void reset_working_memory_tensors(LlamaModel& m);

void do_inference(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, const th::ThLlamaParameters& thParams);

} // namespace th
