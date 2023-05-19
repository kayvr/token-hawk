// TokenHawk llama model.
#include "th-llama.hpp"

#include <functional>
#include <cinttypes>
#include <atomic>
#include <cstring>
#include <queue>

namespace th {

static const int64_t kBatchTokenSize = 8;
static const int64_t kAllowedSubsequentBatchSize = 1;
static const bool kShowTiming = false;
static const bool kSplitCompute = false;
static const int64_t kMaxOutputTokens = 500;
static const int64_t kMaxAsyncTokensPerMsg = 128;
static const int kMaxContext = 512;
#if defined(__EMSCRIPTEN__)
// Emscripten must use async computation.
static const bool kUseAsyncComputation = true;
#else
static const bool kUseAsyncComputation = false;
#endif


tk_llama_token th_eval_gpu(WGPUDevice device,
        WGPUQueue queue,
        std::shared_ptr<LlamaModel> m,
           const tk_llama_token*       tokens,
                         int        n_tokens,
                         int        n_past);

std::vector<tk_llama_token> tk_llama_tokenize(
    std::shared_ptr<LlamaModel> m, const std::string & text, bool add_bos);

const char * tk_llama_token_to_str(std::shared_ptr<LlamaModel> m, tk_llama_token token);
tk_llama_token tk_llama_token_eos();

tk_llama_token llama_sample_top_p_top_k(
        std::shared_ptr<LlamaModel> m,
        const std::vector<tk_llama_token> & last_n_tokens,
        int top_k,
        float top_p,
        float temp,
        float repeat_penalty,
        std::vector<float>& logits);

void build_layer_cmdbuf(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        std::shared_ptr<LlamaModel> m,
        LlamaLayer& l,
        LlamaLayerComputePipeline& p,
        int n_tokens,
        int n_past);

void build_final_compute_cmdbuf(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        std::shared_ptr<LlamaModel> m,
        LlamaFinalComputePipeline& p,
        int n_tokens);

void build_pipelines_llama(WGPUDevice device, WGPUQueue /*queue*/, std::shared_ptr<LlamaModel> m) {
    printf("Constructing compute pipelines.\n");
    LlamaLayer& l = m->layers[0];
    
    build_layer_cmdbuf(device, nullptr, m, l, m->ps, 1, 0);
    build_layer_cmdbuf(device, nullptr, m, l, m->pb, kBatchTokenSize, 0);

    build_final_compute_cmdbuf(device, nullptr, m, m->pfs, 1);
    build_final_compute_cmdbuf(device, nullptr, m, m->pfb, kBatchTokenSize);
    printf("Finished constructing pipelines.\n");
}

void reset_layer_tensors(LlamaLayer& layer) {
    layer.attention_norm.reset_shape();

    layer.wq.reset_shape();
    layer.wk.reset_shape();
    layer.wv.reset_shape();
    layer.wo.reset_shape();

    layer.ffn_norm.reset_shape();

    layer.w1.reset_shape();
    layer.w2.reset_shape();
    layer.w3.reset_shape();

    layer.key_cache.reset_shape();
    layer.value_cache.reset_shape();
}

void reset_working_memory_tensors(LlamaModel& m) {
    m.working_key_cache.reset_shape();
    m.working_val_cache.reset_shape();

    for (int i = 0; i < LlamaModel::nInpBuffers; ++i) {
        m.inp[i].reset_shape();
    }

    m.ffWorking[0].reset_shape();
    m.ffWorking[1].reset_shape();
}

bool sync_continue_inference(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m);
void async_continue_inference(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m);

void do_inference(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, std::string prompt) {
    if (m->n_past >= kMaxOutputTokens) {
        std::string error = "Maximum context reached (" + std::to_string(kMaxContext) + "). Please reset context by typing '[cmd] reset'";
        printf("Warning: %s\n", error.c_str());
        if (m->onError) {
            m->onError(error);
        }
        return;
    }

    if (m->n_past == 0) {
        prompt.insert(0, 1, ' ');
    }
    
    m->embd_inp = tk_llama_tokenize(m, prompt.c_str(), true);
    m->n_consumed = 0;

    if ((int)m->embd_inp.size() > kMaxContext - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) m->embd_inp.size(), kMaxContext - 4);
        return;
    }

    //// determine newline token
    //std::vector<tk_llama_token> llama_token_newline;
    //llama_token_newline = tk_llama_tokenize(m, "\n", false);
    
    printf("n_past: %d\n", m->n_past);

    if (m->last_n_tokens.size() == 0) {
        m->last_n_tokens.resize(kMaxContext);
        std::fill(m->last_n_tokens.begin(), m->last_n_tokens.end(), 0);
    }

    if (kUseAsyncComputation) {
        printf("=================\n");
        printf("ASYNC OPERATION\n");
        printf("=================\n");
        int64_t nStartTokens = m->n_past;
        m->nTargetTokens = nStartTokens + kMaxAsyncTokensPerMsg;
        async_continue_inference(device, queue, m);
#if !defined(__EMSCRIPTEN__)
        while (m->n_past < kMaxOutputTokens) { // May want to make n_past atomic.
            wgpuDeviceTick(device);
            usleep(10);
        }
#endif
    } else {
        printf("=================\n");
        printf("SYNC OPERATION\n");
        printf("=================\n");
        for (int i = 0; i < kMaxOutputTokens; i++) {
            if (!sync_continue_inference(device, queue, m)) {
                printf("EOS\n");
                break;
            }
        }
    }
}

void async_finalize_inference(std::shared_ptr<LlamaModel> m) {
    m->n_past += m->embd.size();

    fflush(stdout);
}

void async_continue_inference(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m) {
    m->embd.clear();

    while ((int) m->embd_inp.size() > m->n_consumed && m->embd.size() < kAllowedSubsequentBatchSize) {
        m->embd.push_back(m->embd_inp[m->n_consumed]);
        m->last_n_tokens.erase(m->last_n_tokens.begin());
        m->last_n_tokens.push_back(m->embd_inp[m->n_consumed]);
        ++m->n_consumed;
    }

    if (m->embd.size() == 0) {
        m->embd.push_back(m->lastGeneratedToken);

        if (m->onNewToken) {
            const char* str = tk_llama_token_to_str(m, m->lastGeneratedToken);
            m->generatedMessage += str;
            m->onNewToken(str, m->generatedMessage);
        }
    }

    /*m->lastGeneratedToken =*/ th_eval_gpu(device, queue, m, m->embd.data(), m->embd.size(), m->n_past);
}

bool sync_continue_inference(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m) {
    std::vector<tk_llama_token> embd;

    while ((int) m->embd_inp.size() > m->n_consumed && embd.size() < kAllowedSubsequentBatchSize) {
        embd.push_back(m->embd_inp[m->n_consumed]);
        m->last_n_tokens.erase(m->last_n_tokens.begin());
        m->last_n_tokens.push_back(m->embd_inp[m->n_consumed]);
        ++m->n_consumed;
    }

    if (embd.size() == 0) {
        embd.push_back(m->lastGeneratedToken);
    }

    m->lastGeneratedToken = th_eval_gpu(device, queue, m, embd.data(), embd.size(), m->n_past);
    m->n_past += embd.size();

    if ((int) m->embd_inp.size() < m->n_consumed) {
        embd.clear();
        embd.push_back(m->lastGeneratedToken);
        if (m->onNewToken) {
            const char* str = tk_llama_token_to_str(m, m->lastGeneratedToken);
            m->onNewToken(str, m->generatedMessage);
        }
    }

    if (!m->embd.empty() && m->embd.back() == tk_llama_token_eos()) {
        return false;
    }

    for (auto id : embd) {
        const char* str = tk_llama_token_to_str(m, id);
        printf("%s", str);
        m->generatedMessage += str;
    }

    fflush(stdout);

    return true;
}

void build_final_compute_cmdbuf(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        std::shared_ptr<LlamaModel> m,
        LlamaFinalComputePipeline& p,
        int n_tokens) {
    reset_working_memory_tensors(*m);

    const int n_embd  = m->n_embd;

    m->inp[0].shape = {.b=0, .r=n_tokens, .c=n_embd};
    m->out = TensorBuffer({.r=1, .c=m->outputMat.shape.r}, TensorType_F32, device);

    cmdbuf_rms_norm(            device, encoder, nullptr, &p.p01, m->inp[0]);
    cmdbuf_row_element_multiply(device, encoder, nullptr, &p.p02, m->inp[0], m->norm);

    if (kSplitFinalMultiply) {
        cmdbuf_vector_multi_mat_mul_split_trans(device, encoder, nullptr, &p.p03, m->inp[0], {&m->outputMatSplit1, &m->outputMatSplit2}, m->out,
            {&m->outScratch},
            ((m->inp[0].shape.r-1)*m->inp[0].shape.c)*get_TensorType_size(m->inp[0].type),
            m->splitBuffers,
            false);
        cmdbuf_vector_reduce( device, encoder, nullptr, &p.p03_reduce, m->out, m->outScratch, 8);
    } else {
        cmdbuf_vector_mat_mul_trans(
            device, encoder, nullptr, &p.p03, m->inp[0], m->outputMat, m->out,
            ((m->inp[0].shape.r-1)*m->inp[0].shape.c)*get_TensorType_size(m->inp[0].type)); // XXX What if less than 8 bits?
    }
}

void build_layer_cmdbuf(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        std::shared_ptr<LlamaModel> m,
        LlamaLayer& l,
        LlamaLayerComputePipeline& p,
        int n_tokens,
        int n_past) {
    reset_working_memory_tensors(*m);
    reset_layer_tensors(l);

    TensorBuffer& queryBuf = m->inp[1];
    TensorBuffer& keyBuf = m->inp[2];
    TensorBuffer& valueBuf = m->inp[3];
    TensorBuffer& queryTranspose = m->inp[4];

    const int n_embd  = m->n_embd;
    const int n_head  = m->n_head;
    
    m->inp[0].shape.r = n_tokens;
    queryBuf.shape.r = n_tokens;
    keyBuf.shape.r = n_tokens;
    valueBuf.shape.r = n_tokens;
    queryTranspose.shape.r = n_tokens;

    WGPUComputePassEncoder pass{};
    
    //bool constructingPipeline = p.p01.buildPipelineFlag;

    cmdbuf_rms_norm(            device, encoder, nullptr, &p.p01, m->inp[0]);
    cmdbuf_row_element_multiply(device, encoder, nullptr, &p.p02, m->inp[0], l.attention_norm);

    if (encoder) { pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr); };
    if (n_tokens == 1) {
        if (kSplitCompute) {
            m->splitScratch[0].shape = queryBuf.shape;
            m->splitScratch[1].shape = keyBuf.shape;
            m->splitScratch[2].shape = valueBuf.shape;
            cmdbuf_vector_mat_mul_split_trans(
                device, encoder, pass, &p.p03_mm,
                 m->inp[0], l.wq, queryBuf, {&m->splitScratch[0]},
                 0,
                 m->splitBuffers, false);
            cmdbuf_vector_mat_mul_split_trans(
                device, encoder, pass, &p.p03_mm,
                 m->inp[0], l.wk, keyBuf, {&m->splitScratch[1]},
                 0,
                 m->splitBuffers, false);
            cmdbuf_vector_mat_mul_split_trans(
                device, encoder, pass, &p.p03_mm,
                 m->inp[0], l.wv, valueBuf, {&m->splitScratch[2]},
                 0,
                 m->splitBuffers, false);
        } else {
            cmdbuf_vector_mat_mul_trans( device, encoder, pass, &p.p03_mm, m->inp[0], l.wq, queryBuf, 0);
            cmdbuf_vector_mat_mul_trans( device, encoder, pass, &p.p03_mm, m->inp[0], l.wk, keyBuf, 0);
            cmdbuf_vector_mat_mul_trans( device, encoder, pass, &p.p03_mm, m->inp[0], l.wv, valueBuf, 0);
        }
    } else {
        cmdbuf_mat_mul(             device, encoder, pass, &p.p03_mm, m->inp[0], l.wq, queryBuf, 1);
        cmdbuf_mat_mul(             device, encoder, pass, &p.p03_mm, m->inp[0], l.wk, keyBuf, 1);
        cmdbuf_mat_mul(             device, encoder, pass, &p.p03_mm, m->inp[0], l.wv, valueBuf, 1);
    }
    if (encoder) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    
    //if (n_tokens == 1 && kSplitCompute) {
    //    if (encoder) { pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr); };
    //    cmdbuf_vector_reduce( device, encoder, pass, &p.p03_mm_reduce, queryBuf, m->splitScratch[0], 2);
    //    cmdbuf_vector_reduce( device, encoder, pass, &p.p03_mm_reduce, keyBuf, m->splitScratch[1], 2);
    //    cmdbuf_vector_reduce( device, encoder, pass, &p.p03_mm_reduce, valueBuf, m->splitScratch[2], 2);
    //    if (encoder) {
    //        wgpuComputePassEncoderEnd(pass);
    //        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    //    }
    //}

    queryBuf.shape = {.b=n_tokens, .r=n_head, .c=n_embd/n_head};
    keyBuf.shape = {.b=n_tokens, .r=n_head, .c=n_embd/n_head};
    valueBuf.shape = {.b=n_head, .r=n_tokens, .c=n_embd/n_head};
    if (encoder) { pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr); }
    cmdbuf_RoPE(                device, encoder, pass, &p.p04_rope, queryBuf, m->networkUniforms);
    cmdbuf_RoPE(                device, encoder, pass, &p.p04_rope, keyBuf, m->networkUniforms);
    if (encoder) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
        pass = nullptr;
    }

    //--------------------------------------------------------------------------------
    // Context-dependent encoding
    //--------------------------------------------------------------------------------
    if (encoder) {
        uint64_t keyTypeSize = get_TensorType_size(keyBuf.type);
        uint64_t keyOffset = n_past*n_embd*keyTypeSize;
        uint64_t valueTypeSize = get_TensorType_size(valueBuf.type);
        uint64_t valueOffset = n_past*n_embd*valueTypeSize;
        wgpuCommandEncoderCopyBufferToBuffer(encoder, keyBuf.gpu, 0, l.key_cache.gpu, keyOffset, keyBuf.get_size_bytes());
        wgpuCommandEncoderCopyBufferToBuffer(encoder, valueBuf.gpu, 0, l.value_cache.gpu, valueOffset, valueBuf.get_size_bytes());
    }

    l.key_cache.shape.b = n_past + n_tokens;
    m->working_key_cache.shape = l.key_cache.shape;
    std::swap(m->working_key_cache.shape.b, m->working_key_cache.shape.r);

    l.value_cache.shape.b = n_past + n_tokens;
    m->working_val_cache.shape = l.value_cache.shape;
    std::swap(m->working_val_cache.shape.b, m->working_val_cache.shape.r);

    queryTranspose.shape = queryBuf.shape;
    std::swap(queryTranspose.shape.b, queryTranspose.shape.r);

    if (encoder) { pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr); }
    cmdbuf_transpose(device, encoder, pass, &p.p05_trans, l.key_cache, m->working_key_cache, true, m->dimsUniforms[0]);
    cmdbuf_transpose(device, encoder, pass, &p.p05_trans, l.value_cache, m->working_val_cache, true, m->dimsUniforms[0]);
    cmdbuf_transpose(device, encoder, pass, &p.p05_trans, queryBuf, queryTranspose, true, m->dimsUniforms[1]);
    if (encoder) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }

    std::swap(m->working_key_cache.shape.r, m->working_key_cache.shape.c);
    m->inp[5].shape = {.b=n_head, .r=n_tokens, .c=n_tokens + n_past};
    
    // Potential optimization: add uniform usage to vector mat_mul.
    cmdbuf_mat_mul(device, encoder, nullptr, &p.p06_mm, queryTranspose, m->working_key_cache, m->inp[5], /*transpose*/1, m->dimsUniforms[2]);

    if (n_tokens == kBatchTokenSize) {
        cmdbuf_masked_softmax(device, encoder, nullptr, &p.p07_softmax, m->inp[5], m->dimsUniforms[3]);
    } else if (n_tokens > 1) {
        printf("Masking unimplemented.\n");
        assert(false);
    } else {
        cmdbuf_row_softmax(device, encoder, nullptr, &p.p07_softmax, m->inp[5], m->dimsUniforms[3]);
    }


    keyBuf.shape = {.b=n_head, .r=n_tokens, .c=n_embd/n_head};

    // Potential optimization. Small buffer that can be easily broadcast.
    cmdbuf_mat_mul(device, encoder, nullptr, &p.p08_mm, m->inp[5], m->working_val_cache, keyBuf, 0, m->dimsUniforms[4]);

    //--------------------------------------------------------------------------------
    // Context-dependent encoding
    //--------------------------------------------------------------------------------
    


    


    m->inp[5].shape = {.b=n_head, .r=n_tokens, .c=n_tokens};
    keyBuf.shape = {.b=n_head, .r=n_tokens, .c=n_embd/n_head};



    std::swap(valueBuf.shape.b, valueBuf.shape.r);
    cmdbuf_transpose(device, encoder, nullptr, &p.p09_t, keyBuf, valueBuf, true, nullptr);

    valueBuf.shape = TensorShape{.l=0, .b=0, .r=n_tokens, .c=m->n_embd};
    m->inp[1].shape = TensorShape{.l=0, .b=0, .r=n_tokens, .c=m->n_embd};
    if (n_tokens == 1) {
        if (kSplitCompute) {
            m->splitScratch[0].shape = m->inp[1].shape;
            cmdbuf_vector_mat_mul_split_trans(
                device, encoder, nullptr, &p.p10_mm,
                valueBuf, l.wo, m->inp[1],
                {&m->splitScratch[0]},
                0,
                m->splitBuffers, false);
            cmdbuf_vector_reduce( device, encoder, nullptr, &p.p10_mm_reduce, m->inp[1], m->splitScratch[0], 16);
        } else {
            cmdbuf_vector_mat_mul_trans(device, encoder, nullptr, &p.p10_mm, valueBuf, l.wo, m->inp[1], 0);
        }
    } else {
        cmdbuf_mat_mul(device, encoder, nullptr, &p.p10_mm, valueBuf, l.wo, m->inp[1], 1);
    }

    m->inp[6].shape = m->inp[1].shape;
    m->inp[2].shape = m->inp[1].shape;
    cmdbuf_addition(device, encoder, nullptr, &p.p11_add, m->inp[1], m->inp[6], m->inp[2]);

    if (encoder) {
        wgpuCommandEncoderCopyBufferToBuffer(encoder, m->inp[2].gpu, 0, m->inp[3].gpu, 0, m->inp[3].get_size_bytes());
    }

    cmdbuf_rms_norm(device, encoder, nullptr, &p.p12_rms, m->inp[2]);
    cmdbuf_row_element_multiply(device, encoder, nullptr, &p.p13_norm, m->inp[2], l.ffn_norm);

    m->ffWorking[0].shape.r = n_tokens;
    m->ffWorking[1].shape.r = n_tokens;

    if (encoder) { pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr); }
    if (n_tokens == 1) {
        cmdbuf_vector_mat_mul_trans(device, encoder, pass, &p.p14_mm, m->inp[2], l.w1, m->ffWorking[0], 0);
        cmdbuf_vector_mat_mul_trans(device, encoder, pass, &p.p14_mm, m->inp[2], l.w3, m->ffWorking[1], 0);
    } else {
        std::swap(l.w1.shape.r, l.w1.shape.c);
        std::swap(l.w3.shape.r, l.w3.shape.c);
        cmdbuf_mat_mul(device, encoder, pass, &p.p14_mm, m->inp[2], l.w1, m->ffWorking[0], 1);
        cmdbuf_mat_mul(device, encoder, pass, &p.p14_mm, m->inp[2], l.w3, m->ffWorking[1], 1);
    }
    if (encoder) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    
    cmdbuf_silu(device, encoder, nullptr, &p.p15_silu, m->ffWorking[0]);

    cmdbuf_element_mult_in_place(device, encoder, nullptr, &p.p16_hadamard, /*out*/m->ffWorking[0], m->ffWorking[1]);

    if (n_tokens == 1) {
        cmdbuf_vector_mat_mul_trans(device, encoder, nullptr, &p.p17_mm, m->ffWorking[0], l.w2, m->inp[2], 0);
    } else {
        std::swap(l.w2.shape.r, l.w2.shape.c);
        cmdbuf_mat_mul(device, encoder, nullptr, &p.p17_mm, m->ffWorking[0], l.w2, m->inp[2], 1);
    }

    cmdbuf_addition(device, encoder, nullptr, &p.p18_add, m->inp[3], m->inp[2], m->inp[0]);

    if (encoder) {
        wgpuCommandEncoderCopyBufferToBuffer(encoder, m->inp[0].gpu, 0, m->inp[6].gpu, 0, m->inp[6].get_size_bytes());
    }
}

static std::vector<double> gHackTimingData{};

tk_llama_token sync_finish_compute(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, double tokenBeginTime);
void async_finish_compute(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, double tokenBeginTime);

static void webgpu_map_trampoline(WGPUBufferMapAsyncStatus status, void* context) {
    std::function<void(WGPUBufferMapAsyncStatus)>* callback = (std::function<void(WGPUBufferMapAsyncStatus)>*)context;
    (*callback)(status);
}

tk_llama_token th_eval_gpu(
        WGPUDevice device,
        WGPUQueue queue,
        std::shared_ptr<LlamaModel> m,
           const tk_llama_token*       tokens,
                         int        n_tokens,
                         int        n_past) {
    const int N = n_tokens;

    double tokenBeginTime = get_time_seconds(); // Note: we should be using queries on the GPU to get better timings.

    const int n_embd  = m->n_embd;
    const int n_head  = m->n_head;

    // Uniforms for RoPE
    {
        LlamaNetworkUniforms uniforms{
            .n_past = (uint32_t)n_past,
            .n_tokens = (uint32_t)n_tokens,
        };
        wgpuQueueWriteBuffer(queue, m->networkUniforms, 0, &uniforms, kLlamaUniformsSize);
    }

    // Uniform buffers for context transpose.
    {
        int64_t b = n_past + n_tokens;
        if (b == 0) { b = 1; }
        LlamaTensorDimsUniforms uniforms{
            .A_B = (uint32_t)b,
            .A_M = (uint32_t)n_head,
            .A_N = (uint32_t)n_embd/n_head,
        };
        wgpuQueueWriteBuffer(queue, m->dimsUniforms[0], 0, &uniforms, kLlamaUniformsSize);
    }

    {
        LlamaTensorDimsUniforms uniforms{
            .A_B = (uint32_t)n_tokens,
            .A_M = (uint32_t)n_head,
            .A_N = (uint32_t)n_embd/n_head,
        };
        wgpuQueueWriteBuffer(queue, m->dimsUniforms[1], 0, &uniforms, kLlamaUniformsSize);
    }
    
    // Uniform buffers for context matrix multiply.
    {
        int64_t head_dim  = n_embd/n_head;

        int64_t b = n_past + n_tokens;
        if (b == 0) { b = 1; }
        LlamaTensorDimsUniforms uniforms{
            .A_B = (uint32_t)n_head,
            .A_M = (uint32_t)n_tokens,
            .A_N = (uint32_t)n_embd/n_head,
            .scale = 1.0f/sqrtf(float(head_dim)),
            .B_B = (uint32_t)n_head,
            .B_M = (uint32_t)(n_embd/n_head),
            .B_N = (uint32_t)(n_past + n_tokens),
            .offset = 0.0,
        };
        wgpuQueueWriteBuffer(queue, m->dimsUniforms[2], 0, &uniforms, kLlamaUniformsSize);
    }

    // Uniform buffers for context softmax
    {
        LlamaTensorDimsUniforms uniforms{
            .A_B = (uint32_t)n_head,
            .A_M = (uint32_t)n_tokens,
            .A_N = (uint32_t)n_tokens + n_past,
        };
        wgpuQueueWriteBuffer(queue, m->dimsUniforms[3], 0, &uniforms, kLlamaUniformsSize);
    }

    // Uniform buffers for context softmax
    {
        LlamaTensorDimsUniforms uniforms{
            .A_B = (uint32_t)n_head,
            .A_M = (uint32_t)n_tokens,
            .A_N = (uint32_t)n_tokens + n_past,
            .scale = 1.0,
            .B_B = (uint32_t)n_head,
            .B_M = (uint32_t)(n_past + n_tokens),
            .B_N = (uint32_t)(n_embd/n_head),
            .offset = 0.0,
        };
        wgpuQueueWriteBuffer(queue, m->dimsUniforms[4], 0, &uniforms, kLlamaUniformsSize);
    }
    
    int64_t size = n_embd * N * sizeof(float);

    if (kUseGpuEmbeddingSelection) {
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        
        // Use GPU-accelerated f16->f32 conversion and row selection.
        for (int i = 0; i < N; i++) {
            int row = tokens[i];
            int64_t embeddingsStride = m->tok_embeddings.shape.c * get_TensorType_size(m->tok_embeddings.type);
            int64_t embeddingsOffset = embeddingsStride * row;
            int64_t inpStride = m->inp[0].shape.c * get_TensorType_size(m->inp[0].type);

            cmdbuf_f16_f32_conversion(
                device, encoder, nullptr, nullptr,
                m->inp[0], m->tok_embeddings, 4,
                i*inpStride,
                embeddingsOffset);
        }

        wgpuCommandEncoderCopyBufferToBuffer(encoder, m->inp[0].gpu, 0, m->inp[6].gpu, 0, m->inp[0].get_size_bytes());

        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);

        wgpuQueueSubmit(queue, 1, &commands);
        wgpuCommandBufferRelease(commands);
    } else {
        for (int i = 0; i < N; i++) {
            int row = tokens[i];
            int64_t stride = m->tok_embeddings.shape.c * get_TensorType_size(m->tok_embeddings.type);
            int64_t offset =  stride * row;
            
            wgpuQueueWriteBuffer(queue, m->inp[0].gpu, i*stride, &m->tok_embeddings.cpuBackup[offset], stride);
            wgpuQueueWriteBuffer(queue, m->inp[6].gpu, i*stride, &m->tok_embeddings.cpuBackup[offset], stride);
        }
    }

    m->inp[0].shape = {.b=0, .r=N, .c=n_embd};
    m->inp[6].shape = {.b=0, .r=N, .c=n_embd};
    m->inp[0].originalShape = m->inp[0].shape;
    m->inp[6].originalShape = m->inp[6].shape;

    {
        // Generate command buffers.
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        
        for (int i = 0; i < (int)m->layers.size(); ++i) {
            LlamaLayer& l = m->layers[i];

            LlamaLayerComputePipeline* pipeline{};
            if (n_tokens == kBatchTokenSize) {
                pipeline = &m->pb;
            } else if (n_tokens == 1) {
                pipeline = &m->ps;
            } else {
                printf("th_eval_gpu: n_tokens must be 1 or %" PRId64 ". Got: %" PRId32 "\n", kBatchTokenSize, n_tokens);
                assert(false);
                return 0;
            }

            build_layer_cmdbuf(
                device,
                encoder,
                m,
                l,
                *pipeline,
                n_tokens,
                n_past);
        }

        reset_working_memory_tensors(*m);

        LlamaFinalComputePipeline* pipeline{};
        if (n_tokens == kBatchTokenSize) {
            pipeline = &m->pfb;
        } else if (n_tokens == 1) {
            pipeline = &m->pfs;
        } else {
            assert(false);
            printf("th_eval_gpu: n_tokens must be 1 or %" PRId64 "\n", kBatchTokenSize);
            return 0;
        }
        build_final_compute_cmdbuf(device, encoder, m, *pipeline, n_tokens);

        WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);

        {
            //ScopedTimeMs scopedTime("Running layers");
            wgpuQueueSubmit(queue, 1, &cmdBuffer);
            wgpuCommandBufferRelease(cmdBuffer);
        }

    }

    //ScopedTimeMs scopedTime("Postprocessing");
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, m->out.gpu, 0, m->resultBuffer.gpu, 0, m->out.get_size_bytes());
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuCommandEncoderRelease(encoder);

    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);

    if (!kUseAsyncComputation) {
        return sync_finish_compute(device, queue, m, tokenBeginTime);
    } else {
        async_finish_compute(device, queue, m, tokenBeginTime);
        return 0;
    }
}

tk_llama_token sync_finish_compute(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, double tokenBeginTime) {
    std::vector<float> logits;
    logits.resize(m->out.shape.c);
    std::atomic<bool> waitingForQueue{true};
    std::function<void(WGPUBufferMapAsyncStatus)> callback =
        [&waitingForQueue,&logits,m](WGPUBufferMapAsyncStatus status) {
            std::shared_ptr<void> raiiCleanup( 0, [&waitingForQueue,m](void*) {
                waitingForQueue.store(false);
                wgpuBufferUnmap(m->resultBuffer.gpu);
            } );

            if (status != WGPUBufferMapAsyncStatus_Success) {
                printf("Result async failed Status: %d\n", status);
                return;
            }

            const uint8_t* mappedData = (const uint8_t*)
                wgpuBufferGetConstMappedRange(m->resultBuffer.gpu, 0, m->resultBuffer.get_size_bytes());
            assert(m->resultBuffer.get_size_bytes() == logits.size() * sizeof(float));

            memcpy(logits.data(), mappedData, m->resultBuffer.get_size_bytes());
        };
            
    // Testing
    wgpuBufferMapAsync(m->resultBuffer.gpu, WGPUMapMode_Read, 0, m->resultBuffer.get_size_bytes(), webgpu_map_trampoline, &callback);

#if defined(__EMSCRIPTEN__)
    printf("ERROR: Emscripten in synchronous path!\n");
    assert(false);
#endif

    // DANGER DANGER
    // TODO Total bug in the Web version of the implementation. There is no synchronization.
    // DANGER DANGER
#if !defined(__EMSCRIPTEN__)
    // Wait for callbacks. A hard synchronization to the GPU.
    while (waitingForQueue.load()) {
        wgpuDeviceTick(device);
        if (!waitingForQueue.load()) {
            break;
        }
        usleep(1);
    }
#endif

    if (kShowTiming) {
        double tokenEndTime = get_time_seconds(); // Note: we should be using queries on the GPU to get better timings.
        gHackTimingData.push_back((tokenEndTime - tokenBeginTime) * 1000.0);

        if (gHackTimingData.size() >= 50) {
            print_descriptive_stats(gHackTimingData, " (ms)");
            gHackTimingData.clear();
        }
    }

    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.10f;

    tk_llama_token token = llama_sample_top_p_top_k(m, {}, top_k, top_p, temp, repeat_penalty, logits);

    return token;
}


static std::vector<std::function<void(WGPUBufferMapAsyncStatus)>> gCallbacks;

static void async_calculate_next_token(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, std::vector<float> logits, double tokenBeginTime);
void async_finish_compute(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, double tokenBeginTime) {
    if (gCallbacks.size() == 0) {
        gCallbacks.reserve(/*n_ctx*/512);
    }
    // TEMPORARY! Static callback. Fix this. Place in a list shared with the model.
    //std::function<void(WGPUBufferMapAsyncStatus)> callback;
    gCallbacks.push_back(
        [m,tokenBeginTime,device,queue](WGPUBufferMapAsyncStatus status) {
            std::vector<float> logits;
            logits.resize(m->out.shape.c);

            if (status != WGPUBufferMapAsyncStatus_Success) {
                printf("Result async failed Status: %d\n", status);
                wgpuBufferUnmap(m->resultBuffer.gpu);
                return;
            }

            const uint8_t* mappedData = (const uint8_t*)
                wgpuBufferGetConstMappedRange(m->resultBuffer.gpu, 0, m->resultBuffer.get_size_bytes());
            assert(m->resultBuffer.get_size_bytes() == logits.size() * sizeof(float));

            memcpy(logits.data(), mappedData, m->resultBuffer.get_size_bytes());

            wgpuBufferUnmap(m->resultBuffer.gpu);

            async_calculate_next_token(device, queue, m, logits, tokenBeginTime);
        });

            
    wgpuBufferMapAsync(m->resultBuffer.gpu, WGPUMapMode_Read, 0, m->resultBuffer.get_size_bytes(), webgpu_map_trampoline, &gCallbacks.back());
}

static void async_calculate_next_token(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, std::vector<float> logits, double tokenBeginTime) {
    if (kShowTiming) {
        double tokenEndTime = get_time_seconds(); // Note: we should be using queries on the GPU to get better timings.
        gHackTimingData.push_back((tokenEndTime - tokenBeginTime) * 1000.0);

        if (gHackTimingData.size() >= 50) {
            print_descriptive_stats(gHackTimingData, " (ms)");
            gHackTimingData.clear();
        }
    }

    //if (temperature > 0.0) {
    //    // Softmax and perform top_p
    //}

    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.10f;

    tk_llama_token token = llama_sample_top_p_top_k(m, {}, top_k, top_p, temp, repeat_penalty, logits);

    m->lastGeneratedToken = token;

    async_finalize_inference(m);

    if (m->n_past < m->nTargetTokens && m->n_past < kMaxOutputTokens) {
        async_continue_inference(device, queue, m);
    } else {
        if (m->onInferenceComplete) {
            m->onInferenceComplete(m->generatedMessage);
            m->generatedMessage = "";
        }
    }
}


static void sample_top_k(std::vector<std::pair<float, tk_llama_token>> & logits_id, int top_k) {
    // find the top k tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<float, tk_llama_token> & a, const std::pair<float, tk_llama_token> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);
}

tk_llama_token llama_sample_top_p_top_k(
        std::shared_ptr<LlamaModel> m,
        const std::vector<tk_llama_token> & last_n_tokens,
        int top_k,
        float top_p,
        float temp,
        float repeat_penalty,
        std::vector<float>& logits) {
    const int n_logits = m->n_vocab;

    const auto * plogits = logits.data() + logits.size() - n_logits;

    if (temp <= 0) {
        // select the token with the highest logit directly
        float max_logit = plogits[0];
        tk_llama_token max_id = 0;

        for (int i = 1; i < n_logits; ++i) {
            if (plogits[i] > max_logit) {
                max_logit = plogits[i];
                max_id = i;
            }
        }
        return max_id;
    }

    std::vector<std::pair<float, tk_llama_token>> logits_id;
    logits_id.reserve(n_logits);

    {
        const float scale = 1.0f/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (plogits[i] < 0.0f) {
                    logits_id.push_back(std::make_pair(plogits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(plogits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(std::make_pair(plogits[i]*scale, i));
            }
        }
    }

    if (top_k > 0 && top_k < n_logits) {
        sample_top_k(logits_id, top_k);
    }

    float maxl = -std::numeric_limits<float>::infinity();
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top k tokens
    std::vector<float> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        const float p = expf(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0) {
        double cumsum = 0.0;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(m->rng);

    return logits_id[idx].second;
}


static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct tk_llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct tk_llama_sp_bigram {
    struct comparator {
        bool operator()(tk_llama_sp_bigram & l, tk_llama_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<tk_llama_sp_bigram>;
    using queue = std::priority_queue<tk_llama_sp_bigram, queue_storage, comparator>;
    tk_llama_sp_symbol::index left;
    tk_llama_sp_symbol::index right;
    float score;
    size_t size;
};

#define Min(X, Y) ((Y) > (X) ? (X) : (Y))

struct TkLlamaTokenizer {
    TkLlamaTokenizer(const LlamaVocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<tk_llama_token> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            tk_llama_sp_symbol sym;
            size_t char_len = Min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(std::move(sym));
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    tk_llama_token token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        tk_llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const LlamaVocab & vocab_;
    std::vector<tk_llama_sp_symbol> symbols_;
    tk_llama_sp_bigram::queue work_queue_;
};

static std::vector<tk_llama_token> tk_llama_tokenize(const LlamaVocab & vocab, const std::string & text, bool bos) {
    TkLlamaTokenizer tokenizer(vocab);
    std::vector<tk_llama_token> output;

    if (text.size() == 0) {
        return output;
    }

    if (bos) {
        output.push_back(1);
    }

    tokenizer.tokenize(text, output);
    return output;
}

int tk_llama_tokenize(
    std::shared_ptr<LlamaModel> m,
                  const char * text,
                 tk_llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = tk_llama_tokenize(m->vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

// TODO: not great allocating this every time
std::vector<tk_llama_token> tk_llama_tokenize(std::shared_ptr<LlamaModel> m, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<tk_llama_token> res(text.size() + (int)add_bos);
    int n = tk_llama_tokenize(m, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}

int tk_llama_n_vocab(std::shared_ptr<LlamaModel> m) {
    return m->vocab.id_to_token.size();
}

const char * tk_llama_token_to_str(std::shared_ptr<LlamaModel> m, tk_llama_token token) {
    if (token >= tk_llama_n_vocab(m)) {
        return nullptr;
    }

    return m->vocab.id_to_token[token].tok.c_str();
}

tk_llama_token tk_llama_token_bos() {
    return 1;
}

tk_llama_token tk_llama_token_eos() {
    return 2;
}



} // namespace th
