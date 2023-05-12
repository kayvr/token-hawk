// TokenHawk llama model.
#include "th-llama.hpp"

#include "llama-cpp/llama.h"
#include "llama-cpp/llama_utils.h"

namespace th {

static const int64_t kBatchTokenSize = 8;
static const int64_t kAllowedSubsequentBatchSize = 1;
static const bool kShowTiming = true;

llama_token th_eval_gpu(WGPUDevice device,
        WGPUQueue queue,
        std::shared_ptr<LlamaModel> m,
           const llama_token*       tokens,
                         int        n_tokens,
                         int        n_past);

llama_vocab::id llama_sample_top_p_top_k(
        std::shared_ptr<LlamaModel> m,
        const std::vector<llama_vocab::id> & last_n_tokens,
        int top_k,
        float top_p,
        float temp,
        float repeat_penalty,
        std::vector<float>& logits);

void build_layer_cmdbuf(
        WGPUDevice device,
        WGPUQueue queue,
        WGPUCommandEncoder encoder,
        std::shared_ptr<LlamaModel> m,
        LlamaLayer& l,
        LlamaLayerComputePipeline& p,
        int n_tokens,
        int n_past);

void build_final_compute_cmdbuf(
        WGPUDevice device,
        WGPUQueue queue,
        WGPUCommandEncoder encoder,
        std::shared_ptr<LlamaModel> m,
        LlamaFinalComputePipeline& p,
        int n_tokens,
        int n_past);

LlamaModel::LlamaModel(llama_context* ctx) {
    this->llamacpp_context = ctx;
}

LlamaModel::~LlamaModel() {
    llama_free(this->llamacpp_context);
    this->llamacpp_context = nullptr;
}

std::shared_ptr<LlamaModel> load_llama(WGPUDevice device, WGPUQueue queue, const std::string& filename, int32_t num_batch_tokens) {
    int32_t seed = 680658349;

    llama_context* ctx{};
    {
        llama_context_params lparams = llama_context_default_params();
        ctx = llama_init_from_file(filename.c_str(), lparams);
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, filename.c_str());
            return {};
        }
    }

    std::shared_ptr<LlamaModel> m = std::make_shared<LlamaModel>(ctx);

    m->rng = std::mt19937(seed);

    m->n_batch = num_batch_tokens;
    m->n_vocab = ctx->model.hparams.n_vocab;
    m->n_ctx   = ctx->model.hparams.n_ctx;
    m->n_embd  = ctx->model.hparams.n_embd;
    m->n_mult  = ctx->model.hparams.n_mult;
    m->n_head  = ctx->model.hparams.n_head;
    m->n_layer = ctx->model.hparams.n_layer;
    m->n_rot   = ctx->model.hparams.n_rot;
    m->f16     = ctx->model.hparams.f16;

    TensorShape kv_cache_shape = TensorShape{ .l=0, .b=m->n_ctx, .r=m->n_head, .c=m->n_embd/m->n_head};

    // Working buffers for input/output.
    m->working_key_cache = TensorBuffer(kv_cache_shape, TensorType_F32, device);
    m->working_val_cache = TensorBuffer(kv_cache_shape, TensorType_F32, device);

    TensorShape inp_shape = TensorShape{.l=0, .b=0, .r=m->n_batch, .c=m->n_embd};
    for (int i = 0; i < LlamaModel::nInpBuffers; ++i) {
        m->inp[i] = TensorBuffer(inp_shape, TensorType_F32, device);
    }

    int n_ff = ((2*(4*m->n_embd)/3 + m->n_mult - 1)/m->n_mult)*m->n_mult;
    assert(n_ff == 11008); // Presumably this value is only relevant for 7B.
    TensorShape ffshape = TensorShape{.l=0, .b=0, .r=m->n_batch, .c=n_ff};
    m->ffWorking[0] = TensorBuffer(ffshape, TensorType_F32, device);
    m->ffWorking[1] = TensorBuffer(ffshape, TensorType_F32, device);

    m->norm = TensorBuffer(ctx->model.norm, device, queue);
    m->outputMat = TensorBuffer(ctx->model.output, device, queue);

    m->out = TensorBuffer({.r=1, .c=m->outputMat.shape.r}, TensorType_F32, device);

    {
        LlamaNetworkUniforms uniforms{};
        size_t size = kLlamaUniformsSize;
        WGPUBufferDescriptor bufferDesc = {};
        bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
        bufferDesc.size  = size;
        m->networkUniforms = wgpuDeviceCreateBuffer(device, &bufferDesc);
        wgpuQueueWriteBuffer(queue, m->networkUniforms, 0, &uniforms, size);
    }

    {
        LlamaTensorDimsUniforms uniforms{};
        size_t size = kLlamaUniformsSize;
        WGPUBufferDescriptor bufferDesc = {};
        bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
        bufferDesc.size  = size;
        for (int i = 0; i < m->dimsUniforms.size(); ++i) {
            m->dimsUniforms[i] = wgpuDeviceCreateBuffer(device, &bufferDesc);
            wgpuQueueWriteBuffer(queue, m->dimsUniforms[i], 0, &uniforms, size);
        }
    }
    
    printf("Constructing compute pipelines.\n");

    //ScopedTimeMs scopedTime("Model upload and pipeline construction");
    //

    assert((TensorShape{.l=0, .b=0, .r=m->n_vocab, .c=m->n_embd} == get_TensorShape_from_ggml(ctx->model.tok_embeddings)));
    m->tok_embeddings = std::move(TensorBuffer(ctx->model.tok_embeddings, /*disable-gpu*/nullptr));
    for (int i = 0; i < m->n_layer; ++i) {
        m->layers.push_back(std::move(LlamaLayer{
          .index = i,

          .attention_norm = TensorBuffer(ctx->model.layers[i].attention_norm, device, queue),

          .wq = TensorBuffer(ctx->model.layers[i].wq, device, queue), // n_embd x n_embd
          .wk = TensorBuffer(ctx->model.layers[i].wk, device, queue), // n_embd x n_embd
          .wv = TensorBuffer(ctx->model.layers[i].wv, device, queue), // n_embd x n_embd
          .wo = TensorBuffer(ctx->model.layers[i].wo, device, queue), // n_embd x n_embd

          .ffn_norm = TensorBuffer(ctx->model.layers[i].ffn_norm, device, queue),

          .w1 = TensorBuffer(ctx->model.layers[i].w1, device, queue),
          .w2 = TensorBuffer(ctx->model.layers[i].w2, device, queue),
          .w3 = TensorBuffer(ctx->model.layers[i].w3, device, queue),

          .key_cache = TensorBuffer(kv_cache_shape, TensorType_F32, device), // n_ctx x n_head x (n_embd/n_head)
          .value_cache = TensorBuffer(kv_cache_shape, TensorType_F32, device), // 512x32x128
        }));
    }

    LlamaLayer& l = m->layers[0];

    // This doesn't build command buffers despite its name. It caches
    // pipelines for future use.
    build_layer_cmdbuf(device, queue, nullptr, m, l, m->ps, 1, 0);
    build_layer_cmdbuf(device, queue, nullptr, m, l, m->pb, kBatchTokenSize, 0);

    build_final_compute_cmdbuf(device, queue, nullptr, m, m->pfs, 1, 0);
    build_final_compute_cmdbuf(device, queue, nullptr, m, m->pfb, kBatchTokenSize, 0);

    printf("Finished constructing pipelines.\n");

    return m;
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

void do_inference(WGPUDevice device, WGPUQueue queue, std::shared_ptr<LlamaModel> m, const th::ThLlamaParameters& thParams) {
    std::string prompt = thParams.prompt;
    prompt.insert(0, 1, ' ');

    auto embd_inp = ::llama_tokenize(m->llamacpp_context, prompt.c_str(), true);
    const int n_ctx = llama_n_ctx(m->llamacpp_context);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return;
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(m->llamacpp_context, "\n", false);

    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    gpt_params params{}; // Default params.

    bool is_antiprompt = false;
    bool input_noecho  = false;

    int n_past     = 0;
    int n_remain   = params.n_predict;
    int n_consumed = 0;

    std::vector<llama_token> embd;

    // We only allow a certain number of tokens to be batch processed at the beginning.
    // This is due to a bug in our masked softmax implementation.
    int64_t allowedInitTokens = 1;
    if (embd_inp.size() >= kBatchTokenSize) {
        allowedInitTokens = kBatchTokenSize;
    }
    while ((int) embd_inp.size() > n_consumed && embd.size() < allowedInitTokens) {
        embd.push_back(embd_inp[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[n_consumed]);
        ++n_consumed;
    }

    if (!input_noecho) {
        for (auto id : embd) {
            printf("%s", llama_token_to_str(m->llamacpp_context, id));
        }
        fflush(stdout);
    }

    llama_token token{};

    for (int i = 0; i < 500; i++) {
        token = th_eval_gpu(device, queue, m, embd.data(), embd.size(), n_past);
        n_past += embd.size();
        embd.clear();

        while ((int) embd_inp.size() > n_consumed && embd.size() < kAllowedSubsequentBatchSize) {
            embd.push_back(embd_inp[n_consumed]);
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(embd_inp[n_consumed]);
            ++n_consumed;
        }

        if (embd.size() == 0) {
            embd.push_back(token);
        }

        for (auto id : embd) {
            printf("%s", llama_token_to_str(m->llamacpp_context, id));
        }
        fflush(stdout);

    }
}

void build_final_compute_cmdbuf(
        WGPUDevice device,
        WGPUQueue queue,
        WGPUCommandEncoder encoder,
        std::shared_ptr<LlamaModel> m,
        LlamaFinalComputePipeline& p,
        int n_tokens,
        int n_past) {
    reset_working_memory_tensors(*m);

    const int n_embd  = m->n_embd;
    const int n_layer = m->n_layer;
    const int n_ctx   = m->n_ctx;
    const int n_head  = m->n_head;
    const int n_vocab = m->n_vocab;
    const int n_rot   = m->n_embd/m->n_head;

    m->inp[0].shape = {.b=0, .r=n_tokens, .c=n_embd};
    m->out = TensorBuffer({.r=1, .c=m->outputMat.shape.r}, TensorType_F32, device);

    cmdbuf_rms_norm(            device, queue, encoder, nullptr, &p.p01, m->inp[0]);
    cmdbuf_row_element_multiply(device, queue, encoder, nullptr, &p.p02, m->inp[0], m->norm);

    cmdbuf_vector_mat_mul_trans(
        device, queue, encoder, nullptr, &p.p03, m->inp[0], m->outputMat, m->out,
        ((m->inp[0].shape.r-1)*m->inp[0].shape.c)*get_TensorType_size(m->inp[0].type)); // XXX What if less than 8 bits?
}

void build_layer_cmdbuf(
        WGPUDevice device,
        WGPUQueue queue,
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
    const int n_layer = m->n_layer;
    const int n_ctx   = m->n_ctx;
    const int n_head  = m->n_head;
    const int n_vocab = m->n_vocab;
    const int n_rot   = m->n_embd/m->n_head;

    int64_t head_dim  = n_embd/n_head;
    
    m->inp[0].shape.r = n_tokens;
    queryBuf.shape.r = n_tokens;
    keyBuf.shape.r = n_tokens;
    valueBuf.shape.r = n_tokens;
    queryTranspose.shape.r = n_tokens;

    WGPUComputePassEncoder pass{};
    
    bool constructingPipeline = p.p01.buildPipelineFlag;

    cmdbuf_rms_norm(            device, queue, encoder, nullptr, &p.p01, m->inp[0]);
    cmdbuf_row_element_multiply(device, queue, encoder, nullptr, &p.p02, m->inp[0], l.attention_norm);

    if (encoder) { pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr); };
    if (n_tokens == 1) {
        cmdbuf_vector_mat_mul_trans( device, queue, encoder, pass, &p.p03_mm, m->inp[0], l.wq, queryBuf, 0);
        cmdbuf_vector_mat_mul_trans( device, queue, encoder, pass, &p.p03_mm, m->inp[0], l.wk, keyBuf, 0);
        cmdbuf_vector_mat_mul_trans( device, queue, encoder, pass, &p.p03_mm, m->inp[0], l.wv, valueBuf, 0);
    } else {
        cmdbuf_mat_mul(             device, queue, encoder, pass, &p.p03_mm, m->inp[0], l.wq, queryBuf, 1);
        cmdbuf_mat_mul(             device, queue, encoder, pass, &p.p03_mm, m->inp[0], l.wk, keyBuf, 1);
        cmdbuf_mat_mul(             device, queue, encoder, pass, &p.p03_mm, m->inp[0], l.wv, valueBuf, 1);
    }
    if (encoder) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }

    queryBuf.shape = {.b=n_tokens, .r=n_head, .c=n_embd/n_head};
    keyBuf.shape = {.b=n_tokens, .r=n_head, .c=n_embd/n_head};
    valueBuf.shape = {.b=n_head, .r=n_tokens, .c=n_embd/n_head};
    if (encoder) { pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr); }
    cmdbuf_RoPE(                device, queue, encoder, pass, &p.p04_rope, queryBuf, m->networkUniforms);
    cmdbuf_RoPE(                device, queue, encoder, pass, &p.p04_rope, keyBuf, m->networkUniforms);
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
    cmdbuf_transpose(device, queue, encoder, pass, &p.p05_trans, l.key_cache, m->working_key_cache, true, m->dimsUniforms[0]);
    cmdbuf_transpose(device, queue, encoder, pass, &p.p05_trans, l.value_cache, m->working_val_cache, true, m->dimsUniforms[0]);
    cmdbuf_transpose(device, queue, encoder, pass, &p.p05_trans, queryBuf, queryTranspose, true, m->dimsUniforms[1]);
    if (encoder) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }

    std::swap(m->working_key_cache.shape.r, m->working_key_cache.shape.c);
    m->inp[5].shape = {.b=n_head, .r=n_tokens, .c=n_tokens + n_past};
    
    // Potential optimization: add uniform usage to vector mat_mul.
    cmdbuf_mat_mul(device, queue, encoder, nullptr, &p.p06_mm, queryTranspose, m->working_key_cache, m->inp[5], /*transpose*/1, m->dimsUniforms[2]);

    if (n_tokens == kBatchTokenSize) {
        cmdbuf_masked_softmax(device, queue, encoder, nullptr, &p.p07_softmax, m->inp[5], m->dimsUniforms[3]);
    } else if (n_tokens > 1) {
        printf("Masking unimplemented.\n");
        assert(false);
    } else {
        cmdbuf_row_softmax(device, queue, encoder, nullptr, &p.p07_softmax, m->inp[5], m->dimsUniforms[3]);
    }


    keyBuf.shape = {.b=n_head, .r=n_tokens, .c=n_embd/n_head};

    // Potential optimization. Small buffer that can be easily broadcast.
    cmdbuf_mat_mul(device, queue, encoder, nullptr, &p.p08_mm, m->inp[5], m->working_val_cache, keyBuf, 0, m->dimsUniforms[4]);

    //--------------------------------------------------------------------------------
    // Context-dependent encoding
    //--------------------------------------------------------------------------------
    


    


    m->inp[5].shape = {.b=n_head, .r=n_tokens, .c=n_tokens};
    keyBuf.shape = {.b=n_head, .r=n_tokens, .c=n_embd/n_head};



    std::swap(valueBuf.shape.b, valueBuf.shape.r);
    cmdbuf_transpose(device, queue, encoder, nullptr, &p.p09_t, keyBuf, valueBuf, true, nullptr);

    valueBuf.shape = TensorShape{.l=0, .b=0, .r=n_tokens, .c=m->n_embd};
    m->inp[1].shape = TensorShape{.l=0, .b=0, .r=n_tokens, .c=m->n_embd};
    if (n_tokens == 1) {
        cmdbuf_vector_mat_mul_trans(device, queue, encoder, nullptr, &p.p10_mm, valueBuf, l.wo, m->inp[1], 0);
    } else {
        cmdbuf_mat_mul(device, queue, encoder, nullptr, &p.p10_mm, valueBuf, l.wo, m->inp[1], 1);
    }

    m->inp[6].shape = m->inp[1].shape;
    m->inp[2].shape = m->inp[1].shape;
    cmdbuf_addition(device, queue, encoder, nullptr, &p.p11_add, m->inp[1], m->inp[6], m->inp[2]);

    if (encoder) {
        wgpuCommandEncoderCopyBufferToBuffer(encoder, m->inp[2].gpu, 0, m->inp[3].gpu, 0, m->inp[3].get_size_bytes());
    }

    cmdbuf_rms_norm(device, queue, encoder, nullptr, &p.p12_rms, m->inp[2]);
    cmdbuf_row_element_multiply(device, queue, encoder, nullptr, &p.p13_norm, m->inp[2], l.ffn_norm);

    m->ffWorking[0].shape.r = n_tokens;
    m->ffWorking[1].shape.r = n_tokens;

    if (encoder) { pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr); }
    if (n_tokens == 1) {
        cmdbuf_vector_mat_mul_trans(device, queue, encoder, pass, &p.p14_mm, m->inp[2], l.w1, m->ffWorking[0], 0);
        cmdbuf_vector_mat_mul_trans(device, queue, encoder, pass, &p.p14_mm, m->inp[2], l.w3, m->ffWorking[1], 0);
    } else {
        std::swap(l.w1.shape.r, l.w1.shape.c);
        std::swap(l.w3.shape.r, l.w3.shape.c);
        cmdbuf_mat_mul(device, queue, encoder, pass, &p.p14_mm, m->inp[2], l.w1, m->ffWorking[0], 1);
        cmdbuf_mat_mul(device, queue, encoder, pass, &p.p14_mm, m->inp[2], l.w3, m->ffWorking[1], 1);
    }
    if (encoder) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    
    cmdbuf_silu(device, queue, encoder, nullptr, &p.p15_silu, m->ffWorking[0]);

    cmdbuf_element_mult_in_place(device, queue, encoder, nullptr, &p.p16_hadamard, /*out*/m->ffWorking[0], m->ffWorking[1]);

    if (n_tokens == 1) {
        cmdbuf_vector_mat_mul_trans(device, queue, encoder, nullptr, &p.p17_mm, m->ffWorking[0], l.w2, m->inp[2], 0);
    } else {
        std::swap(l.w2.shape.r, l.w2.shape.c);
        cmdbuf_mat_mul(device, queue, encoder, nullptr, &p.p17_mm, m->ffWorking[0], l.w2, m->inp[2], 1);
    }

    cmdbuf_addition(device, queue, encoder, nullptr, &p.p18_add, m->inp[3], m->inp[2], m->inp[0]);

    if (encoder) {
        wgpuCommandEncoderCopyBufferToBuffer(encoder, m->inp[0].gpu, 0, m->inp[6].gpu, 0, m->inp[6].get_size_bytes());
    }
}


static void webgpu_map_trampoline(WGPUBufferMapAsyncStatus status, void* context) {
    std::function<void(WGPUBufferMapAsyncStatus, const void*)>* callback = (std::function<void(WGPUBufferMapAsyncStatus, const void*)>*)context;
    (*callback)(status, nullptr);
}

llama_token th_eval_gpu(
        WGPUDevice device,
        WGPUQueue queue,
        std::shared_ptr<LlamaModel> m,
           const llama_token*       tokens,
                         int        n_tokens,
                         int        n_past) {
    TensorShape saveShape{};

    const int N = n_tokens;
    llama_context* lctx = m->llamacpp_context;

    const int n_embd  = m->n_embd;
    const int n_layer = m->n_layer;
    const int n_ctx   = m->n_ctx;
    const int n_head  = m->n_head;
    const int n_vocab = m->n_vocab;
    const int n_rot   = m->n_embd/m->n_head;

    auto & buf_compute   = lctx->buf_compute;
    auto & lmodel        = lctx->model;

    auto & kv_self = lmodel.kv_self;

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
    


    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size(),
        /*.mem_buffer =*/ buf_compute.data(),
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    ggml_cgraph gf = {};
    gf.n_threads = 1;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, tokens, N*ggml_element_size(embd));

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, m->tok_embeddings.ram, embd);

    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    ggml_free(ctx0);
    
    int64_t size = n_embd * N * sizeof(float);

    static std::vector<double> hackTimingData{};
    double tokenBeginTime = get_time_seconds(); // Note: we should be using queries on the GPU to get better timings.
    
    wgpuQueueWriteBuffer(queue, m->inp[0].gpu, 0, ggml_get_data(inpL), size);
    wgpuQueueWriteBuffer(queue, m->inp[6].gpu, 0, ggml_get_data(inpL), size);

    m->inp[0].shape = {.b=0, .r=N, .c=n_embd};
    m->inp[6].shape = {.b=0, .r=N, .c=n_embd};
    m->inp[0].originalShape = m->inp[0].shape;
    m->inp[6].originalShape = m->inp[6].shape;
    
    {
        // Generate command buffers.
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        
        for (int i = 0; i < m->layers.size(); ++i) {
            LlamaLayer& l = m->layers[i];

            LlamaLayerComputePipeline* pipeline{};
            if (n_tokens == kBatchTokenSize) {
                pipeline = &m->pb;
            } else if (n_tokens == 1) {
                pipeline = &m->ps;
            } else {
                printf("th_eval_gpu: n_tokens must be 1 or %ld. Got: %d\n", kBatchTokenSize, n_tokens);
                assert(false);
                return 0;
            }

            build_layer_cmdbuf(
                device,
                queue,
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
            printf("th_eval_gpu: n_tokens must be 1 or %ld\n", kBatchTokenSize);
            return 0;
        }
        build_final_compute_cmdbuf(device, queue, encoder, m, *pipeline, n_tokens, n_past);

        WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);

        {
            //ScopedTimeMs scopedTime("Running layers");
            wgpuQueueSubmit(queue, 1, &cmdBuffer);
            wgpuCommandBufferRelease(cmdBuffer);
        }

    }

    //ScopedTimeMs scopedTime("Postprocessing");

    TensorBuffer resultBuffer(m->out.shape, m->out.type, device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, m->out.gpu, 0, resultBuffer.gpu, 0, m->out.get_size_bytes());
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuCommandEncoderRelease(encoder);

    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);

    std::vector<float> logits;
    logits.resize(m->out.shape.c);
    std::atomic<bool> waitingForQueue{true};
    std::function<void(WGPUBufferMapAsyncStatus)> callback =
        [&resultBuffer,&waitingForQueue,&logits](WGPUBufferMapAsyncStatus status) {
            std::shared_ptr<void> raiiCleanup( 0, [&waitingForQueue,&resultBuffer](void*) {
                waitingForQueue.store(false);
                wgpuBufferUnmap(resultBuffer.gpu);
            } );

            if (status != WGPUBufferMapAsyncStatus_Success) {
                printf("Result async failed Status: %d\n", status);
                return;
            }

            const uint8_t* mappedData = (const uint8_t*)
                wgpuBufferGetConstMappedRange(resultBuffer.gpu, 0, resultBuffer.get_size_bytes());
            assert(resultBuffer.get_size_bytes() == logits.size() * sizeof(float));

            memcpy(logits.data(), mappedData, resultBuffer.get_size_bytes());
        };
            
    // Testing
    wgpuBufferMapAsync(resultBuffer.gpu, WGPUMapMode_Read, 0, resultBuffer.get_size_bytes(), webgpu_map_trampoline, &callback);

    // Wait for callbacks. A hard synchronization to the GPU.
    while (waitingForQueue.load()) {
        wgpuDeviceTick(device);
        if (!waitingForQueue.load()) {
            break;
        }
        usleep(1);
    }

    if (kShowTiming) {
        double tokenEndTime = get_time_seconds(); // Note: we should be using queries on the GPU to get better timings.
        hackTimingData.push_back((tokenEndTime - tokenBeginTime) * 1000.0);

        if (hackTimingData.size() >= 50) {
            print_descriptive_stats(hackTimingData, " (ms)");
            hackTimingData.clear();
        }
    }

    //if (temperature > 0.0) {
    //    // Softmax and perform top_p
    //}

    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.10f;

    // Apparently llama_vocab::id and llama_token are the same thing.
    llama_vocab::id token = llama_sample_top_p_top_k(m, {}, top_k, top_p, temp, repeat_penalty, logits);

    return token;
}

static void sample_top_k(std::vector<std::pair<float, llama_vocab::id>> & logits_id, int top_k) {
    // find the top k tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<float, llama_vocab::id> & a, const std::pair<float, llama_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);
}

llama_vocab::id llama_sample_top_p_top_k(
        std::shared_ptr<LlamaModel> m,
        const std::vector<llama_vocab::id> & last_n_tokens,
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
        llama_vocab::id max_id = 0;

        for (int i = 1; i < n_logits; ++i) {
            if (plogits[i] > max_logit) {
                max_logit = plogits[i];
                max_id = i;
            }
        }
        return max_id;
    }

    std::vector<std::pair<float, llama_vocab::id>> logits_id;
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

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    //printf("Range: %ld\n", logits_id.size());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(m->rng);

    //llama_context* lctx = m->llamacpp_context;
    //for (std::pair<float, llama_vocab::id> i : logits_id) {
    //    printf("Word: %s (%d) (%f)\n", llama_token_to_str(lctx, i.second), i.second, i.first);
    //}

    return logits_id[idx].second;
}


} // namespace th
