#include <emscripten/em_js.h>
#include <emscripten/emscripten.h>

#include <emscripten/html5_webgpu.h>


#include <stdio.h>

#include "model-load.hpp"

#ifndef KEEP_IN_MODULE
#define KEEP_IN_MODULE extern "C" __attribute__((used, visibility("default")))
#endif

extern "C" int __main__(int /*argc*/, char* /*argv*/[]);

static WGPUDevice gDevice{};
static WGPUQueue gQueue{};

void initialize_webgpu(WGPUDevice device);

//****************************************************************************/

namespace impl {
EM_JS(void, glue_preint, (), {
    var entry = __glue_main_;
    if (entry) {
        /*
         * None of the WebGPU properties appear to survive Closure, including
         * Emscripten's own `preinitializedWebGPUDevice` (which from looking at
         *`library_html5` is probably designed to be inited in script before
         * loading the Wasm).
         */

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
    th::loader_set_gpu_device_queue(gDevice, gQueue);
}

//****************************************************************************/

/**
 * Redirector to call \c __main__() (exposed to Emscripten's \c Module).
 *
 * \todo pass URL query string for args
 */
KEEP_IN_MODULE void _glue_main_() {
    printf("In glue_main\n");
    __main__(0, nullptr);
}


/**
 * Entry point. Workaround for Emscripten needing an \c async start.
 */
int main(int /*argc*/, char* /*argv*/[]) {
    printf("In main\n");
    impl::glue_preint();
    return 0;
}


EMSCRIPTEN_KEEPALIVE
extern "C" void capi_test_capi() {
    printf("CAPI Test\n");
}

extern "C" int __main__(int /*argc*/, char* /*argv*/[]) {
    printf("IN THE __MAIN__ FUNCTION!\n");
    initialize_webgpu(emscripten_webgpu_get_device());
    return 0;
}
