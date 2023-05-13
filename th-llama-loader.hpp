#pragma once

#include "th.hpp"
#include "th-llama.hpp"

namespace th {

std::shared_ptr<LlamaModel> load_llama_chunked(WGPUDevice device, WGPUQueue queue, const std::string& dir);
std::shared_ptr<LlamaModel> load_llama_file(
        WGPUDevice device, WGPUQueue queue, const std::string& filename);

void post_load_init_model(WGPUDevice device, WGPUQueue queue, std::shared_ptr<th::LlamaModel> m);
void load_model_chunk(th::LlamaModel* m, WGPUDevice device, WGPUQueue queue, void* data, int64_t dataSize);
bool load_footer(th::LlamaModel* m, void* data, int64_t dataSize);
bool load_weights(th::LlamaModel* m, WGPUDevice device, WGPUQueue queue, void* data, int64_t dataSize, int64_t numElementsInFile, int64_t originalFileOffset);
bool load_header(th::LlamaModel* m, void* data, int64_t dataSize, int64_t /*vocabSize*/);

} // namespace th
