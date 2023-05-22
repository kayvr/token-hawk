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
static std::vector<std::string> gHumanMessages{};
//static int64_t gLastProcessedHumanMessage = -1;
static bool gInferenceComplete = false;
static int64_t gBotMessageSequence = 0;
static std::string gLastBotId = "";

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
extern "C" void capi_load_model_header(void* data, double dataSizeD) {
    int64_t dataSize = (int64_t)dataSizeD;
    load_header(gModel.get(), data, dataSize, 0);
}

EMSCRIPTEN_KEEPALIVE
extern "C" void capi_load_model_weights(void* data, double weightsBeginOffsetD, double dataSizeD) {
    int64_t weightsBeginOffset = (int64_t)weightsBeginOffsetD;
    int64_t dataSize = (int64_t)dataSizeD;
    load_weights(gModel.get(), gDevice, gQueue, data, dataSize, 1, weightsBeginOffset);
}

EM_JS(void, updateMessageText, (const char* messageId, const char* str), {
    var jsMessageStr = UTF8ToString(str);
    var jsMessageId = UTF8ToString(messageId);
    updateMessageText(jsMessageId, jsMessageStr);
});

static void on_new_token(std::string /*token*/, std::string message) {
    updateMessageText(gLastBotId.c_str(), message.c_str());
}

EM_JS(void, sendNewBotMessage, (const char* str, const char* messageId), {
    var jsMessageStr = UTF8ToString(str);
    var jsMessageId = UTF8ToString(messageId);
    sendChatMessage(1, jsMessageStr, jsMessageId);
});

static void on_inference_complete(std::string message) {
    printf("Inference complete! %s\n", message.c_str());
    gInferenceComplete = true;

    //std::string botId = "bot-done-msg-" + std::to_string(gBotMessageSequence);
    //sendNewBotMessage(message.c_str(), botId.c_str());
}

static void on_error(std::string message) {
    gBotMessageSequence = gBotMessageSequence + 1;
    std::string botId = "bot-error-msg-" + std::to_string(gBotMessageSequence);
    sendNewBotMessage(message.c_str(), botId.c_str());
    gInferenceComplete = true;
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

    gModel->onNewToken = &on_new_token;
    gModel->onInferenceComplete = &on_inference_complete;
    gModel->onError = &on_error;

    gInferenceComplete = true;

    return true;
}

EMSCRIPTEN_KEEPALIVE
extern "C" void capi_on_human_message(const char* str) {
    gHumanMessages.push_back(str);
    std::string input(str);

    if (input == "[cmd] reset") {
        gModel->n_past = 0;
        gBotMessageSequence = gBotMessageSequence + 1;
        gLastBotId = "bot-msg-" + std::to_string(gBotMessageSequence);
        sendNewBotMessage("LLM context reset.", gLastBotId.c_str());
        return;
    }

    if (gInferenceComplete) {
        gInferenceComplete = false;
        gBotMessageSequence = gBotMessageSequence + 1;
        gLastBotId = "bot-msg-" + std::to_string(gBotMessageSequence);
        sendNewBotMessage("--", gLastBotId.c_str());
        th::do_inference(gDevice, gQueue, gModel, str);
    }
}

