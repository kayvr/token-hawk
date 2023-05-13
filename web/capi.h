#pragma once

#include <stdint.h>

// Bidirectional CAPI between high-level (web, electron, dawn, what have you).

// WebASM to 'high-level' api.
#ifdef __cplusplus
extern "C" {
#endif
  extern void js_print(const char* logText);
  extern void js_wprint(const char* logText);
  extern void js_eprint(const char* logText);
  extern void js_async_server_request(const char* url, const char* jsonRequest, void* funPtr);
  extern void js_set_inner_html(const char* elementId, const char* html);
  extern void js_javascript_eval(const char* js);
  extern void js_get_stack_trace(char* str, double stringBufferMax);
  extern void js_focus_element(const char* text);
  extern void js_toggle_visibility(const char* element, bool visibility);
  extern void js_upload_local_file(double sequence);
#ifdef __cplusplus
}
#endif

// Definitions needed to mix C and C++.
#ifdef __cplusplus
#define EXPORT_TO_C extern "C"
#else
#define EXPORT_TO_C
#endif

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

// 'high-level' api to WebASM.
EXPORT_TO_C void capi_test_capi();
EXPORT_TO_C void capi_model_begin_load();
EXPORT_TO_C void capi_model_load_hyperparams(const char* data, int64_t dataSize);
//EXPORT_TO_C bool capi_load_model_chunk(int32_t dataSize, int32_t fileType, int32_t numElementsInFile, int32_t vocabSize, void* data);
//EXPORT_TO_C void capi_model_load_weights(const char* name);
//EXPORT_TO_C void capi_model_load_weights(const char* name, int64_t ndims, int64_t ftype, int64_t dim0, int64_t dim1, int64_t dim2, void* tensorData);

