#include <emscripten/em_js.h>
#include <emscripten/emscripten.h>

#include <emscripten/html5_webgpu.h>


#include <stdio.h>

#include "../th.hpp"
#include "../th-llama.hpp"
#include "../th-llama-loader.hpp"

//#include "model-load.hpp"

#ifndef KEEP_IN_MODULE
#define KEEP_IN_MODULE extern "C" __attribute__((used, visibility("default")))
#endif

extern "C" int __main__(int /*argc*/, char* /*argv*/[]);

static WGPUDevice gDevice{};
static WGPUQueue gQueue{};
static std::shared_ptr<th::LlamaModel> gModel;

void initialize_webgpu(WGPUDevice device);

namespace impl {
EM_JS(void, glue_preint, (), {
    var entry = __glue_main_;
    if (entry) {
        // Ensure that webgpu is supported.
        if (navigator["gpu"]) {
            navigator["gpu"]["requestAdapter"]().then(function (adapter) {
                adapter["requestDevice"]({requiredFeatures:[]}).then( function (device) {
                    console.log("Requested device successfully.");
                    Module["preinitializedWebGPUDevice"] = device;
                    entry();
                });
            }, function () {
                console.error("No WebGPU adapter; not starting");
            });
        } else {
            console.error("No support for WebGPU; not starting");
        }
    } else {
        console.error("Entry point not found; unable to start");
    }
});
}

void initialize_webgpu(WGPUDevice device) {
    gDevice = device;
    gQueue = wgpuDeviceGetQueue(gDevice);
}

KEEP_IN_MODULE void _glue_main_() {
    __main__(0, nullptr);
}

int main(int /*argc*/, char* /*argv*/[]) {
    impl::glue_preint();
    return 0;
}


EMSCRIPTEN_KEEPALIVE
extern "C" void capi_test_capi() {
    printf("CAPI Test\n");
}

extern "C" int __main__(int /*argc*/, char* /*argv*/[]) {
    initialize_webgpu(emscripten_webgpu_get_device());
    return 0;
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

