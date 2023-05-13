#include "model-load.hpp"

#include "../th.hpp"
#include "../th-llama.hpp"
#include "../th-llama-loader.hpp"

#include "capi.h"

#include <emscripten/em_js.h>
#include <emscripten/emscripten.h>

#include <span>
#include <cstdint>
#include <iostream>

namespace th {

std::shared_ptr<th::LlamaModel> gModel;
static WGPUDevice gDevice;
static WGPUQueue gQueue;

static const int64_t kFileType_Header = 0;
static const int64_t kFileType_Weights = 1;
static const int64_t kFileType_Footer = 2;

static const int64_t kftype_f32 = 0;
static const int64_t kftype_f16 = 1;
static const int64_t kftype_q40 = 2;
static const int64_t kftype_q41 = 3;

void loader_set_gpu_device_queue(WGPUDevice device, WGPUQueue queue) {
    gDevice = device;
    gQueue = queue;
}

EMSCRIPTEN_KEEPALIVE
extern "C" void capi_model_begin_load() {
    gModel = std::make_shared<th::LlamaModel>();
    gModel->numFilesLoaded = 0;
}

EMSCRIPTEN_KEEPALIVE
extern "C" void capi_load_model_chunk(void* data, int64_t dataSize) {
    load_model_chunk(gModel.get(), gDevice, gQueue, data, dataSize);
}

EMSCRIPTEN_KEEPALIVE
extern "C" bool capi_model_end_load() {
    if (gModel->targetFilesLoaded != gModel->numFilesLoaded) {
        printf("Error: Failed to load the proper number of files.\n");
        printf("Files loaded: %lld. Files needed: %lld.\n", gModel->numFilesLoaded, gModel->targetFilesLoaded);
        gModel->loadFailed = true;
        return false;
    }
    printf("Successfully loaded model!\n");

    printf("Initializing and reprocessing...\n");
    post_load_init_model(gDevice, gQueue, gModel);

    printf("Performing inference\n");
    th::ThLlamaParameters params{};
    th::do_inference(gDevice, gQueue, gModel, params);
    printf("After inference..\n");
    return true;
}


} // namespace th
