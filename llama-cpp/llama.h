#ifndef LLAMA_H
#define LLAMA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <vector>
#include <regex>
#include <cassert>
#include <cstring>
#include <memory>

#include "ggml.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#define LLAMA_FILE_VERSION 1
#define LLAMA_FILE_MAGIC 0x67676a74 // 'ggjt' in hex
#define LLAMA_FILE_MAGIC_UNVERSIONED 0x67676d6c // pre-versioned files


#define LLAMA_USE_SCRATCH
#define LLAMA_MAX_SCRATCH_BUFFERS 16

#define JhMaximillion(X, Y) ((Y) < (X) ? (X) : (Y))

// NOTE: The following is outside of extern 'c'.

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_7B,
    MODEL_13B,
    MODEL_30B,
    MODEL_65B,
};


    // default hparams (LLaMA 7B)
    struct llama_hparams {
        int32_t n_vocab = 32000;
        int32_t n_ctx   = 512;   // this is provided as user input?
        int32_t n_embd  = 4096;
        int32_t n_mult  = 256;
        int32_t n_head  = 32;
        int32_t n_layer = 32;
        int32_t n_rot   = 64;
        int32_t f16     = 1;
    };
    
    struct llama_layer {
        // normalization
        struct ggml_tensor * attention_norm;
    
        // attention
        struct ggml_tensor * wq;
        struct ggml_tensor * wk;
        struct ggml_tensor * wv;
        struct ggml_tensor * wo;
    
        // normalization
        struct ggml_tensor * ffn_norm;
    
        // ff
        struct ggml_tensor * w1;
        struct ggml_tensor * w2;
        struct ggml_tensor * w3;
    };
    
    struct llama_kv_cache {
        struct ggml_tensor * k;
        struct ggml_tensor * v;
    
        struct ggml_context * ctx;
    
        std::vector<uint8_t> buf;
    
        int n; // number of tokens currently in the cache
    };

    struct llama_model {
        e_model type = MODEL_UNKNOWN;
    
        llama_hparams hparams;
    
        struct ggml_tensor * tok_embeddings;
    
        struct ggml_tensor * norm;
        struct ggml_tensor * output;
    
        std::vector<llama_layer> layers;
    
        // context
        struct ggml_context * ctx;
    
        // key + value cache for the self attention
        // TODO: move to llama_state
        struct llama_kv_cache kv_self;
    
        // the model memory buffer
        std::vector<uint8_t> buf;
    
        // model memory mapped file
        void * mm_addr = NULL;
        uint64_t mm_length = 0;
    
        // tensors
        int n_loaded;
        std::unordered_map<std::string, struct ggml_tensor *> tensors;
    };

    struct llama_vocab {
        using id    = int32_t;
        using token = std::string;
    
        struct token_score {
            token tok;
            float score;
        };
    
        std::unordered_map<token, id> token_to_id;
        std::vector<token_score> id_to_token;
    };

struct llama_context {
    std::mt19937 rng;

    // JH_START_CHANGE
    int64_t n_tokens = 0; // MUST BE A MULTIPLE OF 8. See compute shaders.
    int64_t n_past = 0;
    // JH_END_CHANGE

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;
    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    llama_model model;
    llama_vocab vocab;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    std::vector<uint8_t> buf_compute;
    std::vector<uint8_t> buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size(), buf.data(), });
        }

        if (buf_last >= 0) {

            buf_max_size[buf_last] = JhMaximillion(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};


#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_context;

    typedef int llama_token;

    typedef struct llama_token_data {
        llama_token id;  // token id

        float p;     // probability of the token
        float plog;  // log probability of the token

    } llama_token_data;

    typedef void (*llama_progress_callback)(float progress, void *ctx);

    struct llama_context_params {
        int n_ctx;   // text context
        int n_parts; // -1 for default
        int seed;    // RNG seed, 0 for random

        bool f16_kv;     // use fp16 for KV cache
        bool logits_all; // the llama_eval() call computes all logits, not just the last one
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mlock;  // force system to keep model in RAM
        bool embedding;  // embedding mode only

        // called with a progress value between 0 and 1, pass NULL to disable
        llama_progress_callback progress_callback;
        // context pointer passed to the progress callback
        void * progress_callback_user_data;
    };

    LLAMA_API struct llama_context_params llama_context_default_params();

    // Various functions for loading a ggml llama model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    LLAMA_API struct llama_context * llama_init_from_file(
                             const char * path_model,
            struct llama_context_params   params);

    // Frees all allocated memory
    LLAMA_API void llama_free(struct llama_context * ctx);

    // TODO: not great API - very likely to change
    // Returns 0 on success
    LLAMA_API int llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
                   int   itype);

    // Returns the KV cache that will contain the context for the
    // ongoing prediction with the model.
    LLAMA_API const uint8_t * llama_get_kv_cache(struct llama_context * ctx);

    // Returns the size of the KV cache
    LLAMA_API size_t llama_get_kv_cache_size(struct llama_context * ctx);

    // Returns the number of tokens in the KV cache
    LLAMA_API int llama_get_kv_cache_token_count(struct llama_context * ctx);

    // Sets the KV cache containing the current context for the model
    LLAMA_API void llama_set_kv_cache(
            struct llama_context * ctx,
                   const uint8_t * kv_cache,
                          size_t   n_size,
                             int   n_token_count);

    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    LLAMA_API int llama_eval(
            struct llama_context * ctx,
               const llama_token * tokens,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    LLAMA_API int llama_tokenize(
            struct llama_context * ctx,
                      const char * text,
                     llama_token * tokens,
                             int   n_max_tokens,
                            bool   add_bos);

    LLAMA_API int llama_n_vocab(struct llama_context * ctx);
    LLAMA_API int llama_n_ctx  (struct llama_context * ctx);
    LLAMA_API int llama_n_embd (struct llama_context * ctx);

    // Token logits obtained from the last call to llama_eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // Rows: n_tokens
    // Cols: n_vocab
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Token Id -> String. Uses the vocabulary in the provided context
    LLAMA_API const char * llama_token_to_str(struct llama_context * ctx, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_token_bos();
    LLAMA_API llama_token llama_token_eos();

    // TODO: improve the last_n_tokens interface ?
    LLAMA_API llama_token llama_sample_top_p_top_k(
       struct llama_context * ctx,
          const llama_token * last_n_tokens_data,
                        int   last_n_tokens_size,
                        int   top_k,
                      float   top_p,
                      float   temp,
                      float   repeat_penalty);

    // Performance information
    LLAMA_API void llama_print_timings(struct llama_context * ctx);
    LLAMA_API void llama_reset_timings(struct llama_context * ctx);

    // JH_START_CHANGE
    LLAMA_API void llama_print_model_info(struct llama_context * ctx);
    // JH_END_CHANGE

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);


#ifdef __cplusplus
}
#endif

#endif
