// Javascript API. See: https://github.com/kripken/emscripten/tree/master/src
mergeInto(LibraryManager.library, {
  js_print: function(text_ptr) {
    var text = UTF8ToString(text_ptr);
    console.log(text);
  },
  js_wprint: function(text_ptr) {
    var text = UTF8ToString(text_ptr);
    console.log(text);
  },
  js_eprint: function(text_ptr) {
    var text = UTF8ToString(text_ptr);
    console.log(text);
  },
  js_async_server_request: function(request_url_ptr, request_json_ptr, fun_ptr) {
    var request_json = UTF8ToString(request_json_ptr);
    var request_url = UTF8ToString(request_url_ptr);

    var request_url_print = "Request URL: " + request_url;
    console.log(request_url_print);

    // https://stackoverflow.com/questions/19959072/sending-binary-data-in-javascript-over-http
    // https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest/response
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
      // See: https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest/readyState
      const READY_STATE_DONE = 4;
      if (xhr.readyState === READY_STATE_DONE) {
        var response = xhr.response;

        var ptr = allocate(intArrayFromString(response), 'i8', ALLOC_NORMAL);
        _jstoc_HandleAsyncServerResponse(ptr, fun_ptr);
        _free(ptr);
      }
    }
    xhr.open('POST', request_url);
    // https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(request_json);
  },
  js_set_inner_html: function(element_name_ptr, html_ptr) {
    var html = UTF8ToString(html_ptr);
    var element_name = UTF8ToString(element_name_ptr);

    var element = document.getElementById(element_name);

    if (element)
    {
      // var text_to_print = "Element found: " + element_name + " Text to print: " + html;
      // console.log(text_to_print);
      element.innerHTML = `
      ${html}
      `;
    }
    else
    {
      //alert("Element '" + element_name + "' not found.");
    }
  },
  js_javascript_eval: function(js_ptr) {
    var js = UTF8ToString(js_ptr);
    eval(js);
  },
  js_get_stack_trace : function(cStringPtr, cStringMaxLength) {
    //var stackTraceString = new Error().stack;
    var stackTraceString = stackTrace(); 
    stringToUTF8(stackTraceString, cStringPtr, cStringMaxLength);
  },
  js_focus_element: function(cStringPtr) {
    var element_to_focus = UTF8ToString(cStringPtr);
    var element = document.getElementById(element_to_focus);
    if (element) {
      element.focus();
    }
  },
  js_toggle_visibility: function(cStringPtr, visible) {
    var element_to_focus = UTF8ToString(cStringPtr);
    var element = document.getElementById(element_to_focus);
    if (element) {
      if (visible) {
        element.style.display = "block";
      } else {
        element.style.display = "none";
      }
    }
  },
  js_upload_local_file: function(sequence) {
    let requestUpload = async () => {
      try {
        [fileHandle]    = await window.showOpenFilePicker();
        const file      = await fileHandle.getFile();

        //// Works when expecting text file.
        // const contents  = await file.text();
        // var ptr = allocate(intArrayFromString(contents), ALLOC_NORMAL);
        // _capi_handle_async_file_upload("", ptr, contents.byteLength, sequence);
        // _free(ptr);

        // 
        const contents = new Uint8Array(await file.arrayBuffer());
        var ptr = Module._malloc(contents.byteLength);
        writeArrayToMemory(contents, ptr);
        var namePtr = allocate(intArrayFromString(file.name), ALLOC_NORMAL);
        _capi_handle_async_file_upload(namePtr, ptr, contents.byteLength, sequence);
        _free(namePtr)
        Module._free(ptr);
      } catch (e) {
        let error = " " + e
        var ptr = allocate(intArrayFromString(error), ALLOC_NORMAL);
        _capi_handle_async_file_upload_failure(ptr, sequence);
        _free(ptr)
      }
    }
    requestUpload();
  },
});
