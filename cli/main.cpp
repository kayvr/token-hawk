#include <dawn/native/DawnNative.h>
#include <dawn/dawn_proc.h>

#include <iostream>
#include <memory>
#include <string.h>
#include <filesystem>

#include "th.hpp"
#include "th-llama.hpp"
#include "th-llama-loader.hpp"


static void printDeviceError(WGPUErrorType errorType, const char *message, void *);
static void printDeviceLost(WGPUDeviceLostReason reason, char const *message, void *);
static void printDeviceLog(WGPULoggingType type, char const *message, void *);
static void request_adapter_callback(WGPURequestAdapterStatus, WGPUAdapter received, const char *, void *userdata);
static void print_usage();

static bool run_inference(WGPUDevice device, WGPUQueue queue, const th::ThLlamaParameters& params);

int main(int argc, char* argv[]) {
    th::ThLlamaParameters params{};
    for (int i = 1; i < argc; i++)
    {
        const char* arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            print_usage();
            return EXIT_SUCCESS;
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            if (i == argc - 1) {
                fprintf(stderr, "'-m' requires a filename.\n");
                return EXIT_FAILURE;
            }

            params.modelFile = argv[++i];
            continue;
        } else if (!strcmp(arg, "-d") || !strcmp(arg, "--dir")) {
            if (i == argc - 1) {
                fprintf(stderr, "'-d' requires a directory.\n");
                return EXIT_FAILURE;
            }

            params.directory = argv[++i];
            continue;
        }

        // Positional parameters are interpreted as part of the prompt.
        if (arg[0] == '-') {
            fprintf(stderr, "Unrecognized argument: %s\n", arg);
            return EXIT_FAILURE;
        }

        if (params.prompt.size() > 0) {
            params.prompt += " ";
        }
        params.prompt += argv[i];
    }


    WGPUDevice device;
    WGPUQueue queue;

    DawnProcTable procs = dawn::native::GetProcs();
    dawnProcSetProcs(&procs);

    WGPUInstanceDescriptor instanceDesc{};
    WGPUInstance instance = wgpuCreateInstance(&instanceDesc);

    WGPURequestAdapterOptions adapterOpts{
        .powerPreference = WGPUPowerPreference_HighPerformance,
        .forceFallbackAdapter = false,
    };

    WGPUAdapter adapter = nullptr;
    wgpuInstanceRequestAdapter(instance, &adapterOpts, request_adapter_callback, reinterpret_cast<void *>(&adapter));

    std::vector<WGPUFeatureName> requiredFeatures = {
        //WGPUFeatureName::WGPUFeatureName_DawnShaderFloat16,
        //WGPUFeatureName::WGPUFeatureName_ShaderF16,
        //WGPUFeatureName::WGPUFeatureName_TimestampQuery,
    };

    WGPUDeviceDescriptor deviceDesc{
        .requiredFeaturesCount = requiredFeatures.size(),
        .requiredFeatures = requiredFeatures.data(),
    };
    device = wgpuAdapterCreateDevice(adapter, &deviceDesc);

    wgpuDeviceSetUncapturedErrorCallback(device, printDeviceError, nullptr);
    wgpuDeviceSetDeviceLostCallback(device, printDeviceLost, nullptr);
    wgpuDeviceSetLoggingCallback(device, printDeviceLog, nullptr);

    queue = wgpuDeviceGetQueue(device);

    run_inference(device, queue, params);

    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);

    printf("\n");

    return 0;
}

static bool run_inference(WGPUDevice device, WGPUQueue queue, const th::ThLlamaParameters& params) {
    std::shared_ptr<th::LlamaModel> model;
    if (!params.directory.empty()) {
        model = th::load_llama_chunked(device, queue, params.directory);
    } else {
        model = th::load_llama_file(device, queue, params.modelFile);
    }
    th::do_inference(device, queue, model, params.prompt);
    model.reset();
}

static void printDeviceError(WGPUErrorType errorType, const char *message, void *) {
    switch (errorType) {
        case WGPUErrorType_Validation:
            printf("[ERROR] DEVICE::VALIDATION: %s\n", message);
            break;
        case WGPUErrorType_OutOfMemory:
            std::cerr << "[ERROR] DEVICE::OUT_OF_MEMORY: " << message << std::endl;
            break;
        case WGPUErrorType_DeviceLost:
            std::cerr << "[ERROR] DEVICE::LOST: " << message << std::endl;
            break;
        default:
            std::cerr << "[ERROR] DEVICE::UNKNOWN: " << message << std::endl;
            return;
    }
    fflush(stdout);
}

static void printDeviceLost(WGPUDeviceLostReason reason, char const *message, void *) {
    switch (reason) {
        case WGPUDeviceLostReason_Undefined:
            std::cerr << "[VERBOSE] DEVICE::LOST::UNDEFINED: " << message << std::endl;
            break;
        case WGPUDeviceLostReason_Force32:
            std::cerr << "[VERBOSE] DEVICE::LOST::FORCE32: " << message << std::endl;
            break;
        case WGPUDeviceLostReason_Destroyed:
            break;
    }
}

static void printDeviceLog(WGPULoggingType type, char const *message, void *) {
    switch (type) {
        case WGPULoggingType_Verbose:
            std::cout << "[VERBOSE] DEVICE: " << message << std::endl;
            break;
        case WGPULoggingType_Warning:
            std::cout << "[WARNING] DEVICE: " << message << std::endl;
            break;
        case WGPULoggingType_Error:
            std::cerr << "[ERROR] DEVICE: " << message << std::endl;
            break;
        default:
            std::cout << "[INFO] DEVICE: " << message << std::endl;
            break;
    }
}

static void request_adapter_callback(WGPURequestAdapterStatus, WGPUAdapter received, const char *, void *userdata) {
    *reinterpret_cast<WGPUAdapter *>(userdata) = received;
}

static void print_usage()
{
  printf("USAGE:\n"
         "    th [FLAGS] [OPTIONS] [prompt]...\n"
         "\n"
         "FLAGS:\n"
         "    -m, --model         Model to load (default: models/7B/ggml-model-f16.bin).\n"
         "    -d, --dir           Load web-chunked model data from this directory.\n"
         "    -h, --help          Print this help.\n"
         "\n"
         "ARGS:\n"
         "    [prompt]...     The prompt to pass to the LLM.\n"
         "\n"
         "EXAMPLES:\n"
         "    th -m models/llama-7B/ggml-model-f16.bin \"Hello, how are you?\"\n"
         );
}
