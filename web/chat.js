const kHumanId = 0;
const kBotId = 1;
const kSystemId = 2;

const kFileType_Header = 0;
const kFileType_Weights = 1;
const kFileType_Footer = 2;

const u32_s = 4; // Size of uint32
const i32_s = 4; // Size of uint32
const f32_s = 4;
const u16_s = 2;
const u64_s = 8;

const th_magic = 0x1737
const th_version = 1

let gFilesProcessed = 0;

function sendChatMessage(botId, message) {
    if (message.trim()) { // trim to prevent sending empty messages
        var chatlogs = document.getElementById('chatlogs');
        var lines = message.split('\n');
        var messageContainer = document.createElement('div');
        messageContainer.className = 'message';

        var icon = document.createElement('img');
        if (botId === kBotId || botId == kSystemId) {
            icon.src = 'bot.png';
        } else {
            icon.src = 'human.png';
        }
        icon.className = 'message-icon';
        messageContainer.appendChild(icon);

        // Create message text container
        var messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        for (var i = 0; i < lines.length; i++) {
            if (i === 0 && botId == kSystemId) {
                lines[i] =  "[System] " + lines[i]
            }
            var newMessage = document.createElement('p');
            if (lines[i].trim()) { // only append non-empty lines
                if (botId == kSystemId) {
                    newMessage.innerHTML = lines[i];
                } else {
                    newMessage.textContent = lines[i];
                }
                messageContainer.appendChild(newMessage);
            } else { // Append <br> in place .
                newMessage = document.createElement('br');
            }
            messageText.appendChild(newMessage);
        }

        messageContainer.appendChild(messageText);
        chatlogs.appendChild(messageContainer);

        chatlogs.scrollTop = chatlogs.scrollHeight;
    }
}

function submitMessage() {
    var messagebox = document.getElementById('message');
    var message = messagebox.value;

    sendChatMessage(kHumanId, message);

    messagebox.value = '';
}

document.getElementById('send-btn').addEventListener('click', function() {
    submitMessage()
});

document.getElementById('message').addEventListener('keydown', function(event) {
    if (event.key == 'Enter') {
        event.preventDefault();
        if (event.shiftKey || event.ctrlKey) {
            this.value += '\n'; // Add a newline if shift or ctrl is pressed.
        } else {
            submitMessage()
        }
    }
});

let gCurrentFile = 0;
let gFiles = [];

document.getElementById('load-model').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
    fileInput.click();
});

document.getElementById('fileInput').addEventListener('change', async (e) => {
    const progressBar = document.getElementById("progress-bar");
    progressBar.style.display = "block";
    progressBar.value = 0;

    gFilesProcessed = 0;

    Module._capi_model_begin_load()

    var files = e.target.files; // This is a FileList object
    console.log("Files to process: " + files.length);
    for(var i = 0; i < files.length; i++) {
        progressBar.value = (i / files.length) * 100;
        if (!await loadChunkedFile(files[i])) {
            return;
        }
        gFilesProcessed++;
    }
    console.log("Files processed: " + gFilesProcessed);

    if (Module._capi_model_end_load()) {
        sendChatMessage(kSystemId, "Successfully loaded model.");
    }

    progressBar.style.display = "none";
});

document.getElementById('convert-model').addEventListener('click', function() {
    const convertInput = document.getElementById('convert-model-input');
    convertInput.click();
});

document.getElementById('convert-model-input').addEventListener('change', async (e) => {
    //var files = e.target.files; // This is a FileList object
    //for(var i = 0; i < files.length; i++) {
    //    console.log(files[i]); // You can handle each file here
    //}

    const file = e.target.files[0];
    if (file) {
        convertGGMLFile(file, true);
    }
});

function timeoutSleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function downloadGGMLChunk(filename, buffer, fileType, numElementsInFile, vocabSize, originalFileOffset) {
    // TODO Await promise to download file.

    let prependedBuffer = new ArrayBuffer(u16_s * 2 + u32_s*3 + u64_s*2);
    let view = new DataView(prependedBuffer);
    view.setUint16(0, th_magic, true);
    view.setUint16(u16_s, th_version, true);
    view.setUint32(u32_s*1, fileType, true);
    view.setUint32(u32_s*2, numElementsInFile, true);
    view.setUint32(u32_s*3, vocabSize, true);

    // Offset of the original file in bytes. 
    // Used for calculating the 32-byte alignment offset
    view.setBigInt64(u32_s*4, BigInt(originalFileOffset), true);
    view.setBigInt64(u32_s*6, BigInt(0), true); // padding

    let blob = new Blob([prependedBuffer, buffer], {type: 'application/octet-stream'});
    let url = URL.createObjectURL(blob);

    let link = document.createElement('a');
    link.href = url;
    link.download = filename;
    
    document.body.appendChild(link);
    link.click();
    
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

async function loadChunkedFile(file, convert) {
    const fileSize = file.size;

    const fileData = await readData(file, 0, fileSize);

    const dataPtr = Module._malloc(fileSize);
    const dataView = new Uint8Array(fileData);
    Module.HEAPU8.set(dataView, dataPtr);
    Module._capi_load_model_chunk(dataPtr, fileSize);
    Module._free(dataPtr);

    return true;
}


async function convertGGMLFile(file, convert) {

    const progressBar = document.getElementById("progress-bar");
    progressBar.style.display = "block";
    progressBar.value = 0;

    const fileSize = file.size;

    Module._capi_model_begin_load(fileSize)

    cur = 0;
    originalFileOffset = 0;

    // Ensure we have a GGML file and it is of the correct version.
    {
        size = u32_s * 2;
        const magicData = await readData(file, cur, size);
        cur += size;

        const ggmlMagicUnversioned = 0x67676d6c;
        const ggmlMagicValue = 0x67676a74;
        const magicView = new DataView(magicData);

        const fileMagic = magicView.getUint32(0, true);

        if (fileMagic === ggmlMagicUnversioned) {
            console.log("Model file is too old: " + file)
            return;
        }

        if (ggmlMagicValue != magicView.getUint32(0, true)) {
            console.log("Failed to read magic from GGML file. Is this a valid GGML file?")
            return;
        }

        const llamaFileVersion = 1;
        ggmlFormatVersion = magicView.getUint32(u32_s, true);

        if (ggmlFormatVersion != llamaFileVersion) {
            console.log("Failed to read magic from GGML file. Is this a valid GGML file?")
            return;
        }
    }

    // Load hyper-parameters.
    n_vocab = 0;
    n_ctx = 0;
    n_embd = 0;
    n_mult = 0;
    n_head = 0;
    n_layer = 0;
    n_rot = 0;
    f16 = 0;
    {
        size = i32_s * 7;
        const hpData = await readData(file, cur, size);
        cur += size;

        const hpView = new DataView(hpData);

        n_vocab = hpView.getUint32(0*u32_s, true);
        n_embd  = hpView.getUint32(1*u32_s, true);
        n_mult  = hpView.getUint32(2*u32_s, true);
        n_head  = hpView.getUint32(3*u32_s, true);
        n_layer = hpView.getUint32(4*u32_s, true);
        n_rot   = hpView.getUint32(5*u32_s, true);
        f16     = hpView.getUint32(6*u32_s, true);
    }

    // Load vocabulary (wish we could read in one chunk).
    // For loop over n_vocab
    vocabBegin = cur
    for (let i = 0; i < n_vocab; i++) {
        const sizeData = await readData(file, cur, u32_s);
        cur += u32_s;
        const sizeView = new DataView(sizeData);
        size = sizeView.getUint32(0,true);
        cur += size + f32_s;
    }
    vocabEnd = cur
    //console.log("Typical header size: " + cur);
    
    let ggmlChunk = 0;

    if (convert) {
        const data = await readData(file, 0, vocabEnd);
        await downloadGGMLChunk(file.name + "-chunk-" + ggmlChunk.toString().padStart(4, '0'), data, kFileType_Header, 0, vocabEnd - vocabBegin, originalFileOffset);
        // TODO Await file selection box.
        ++ggmlChunk;
        originalFileOffset = cur;
    }

    let decoder = new TextDecoder('utf-8');

    const ftype_f32 = 0;
    const ftype_f16 = 1;
    const ftype_q40 = 2;
    const ftype_q41 = 3;

    const maxFileSize = 550000000; // Limit for chrome.

    // Unfortunately, we don't know how many tensors are coming.
    // So we don't accurately gauge progress.
    // We can detect 7B, 13B, etc... by the number of layers
    // that are specified in the hyper parameters.
    // So we hardcode 7B values (32) for now.
    totalSteps = 32 * 9;

    lastWeightPtr = cur
    lastChunkWrittenPtr = cur

    // Load each of the tensors into memory.
    currentLayerIndex = 0
    numWeightsSoFar = 0
    while (cur != fileSize) {
        progressBar.value = (currentLayerIndex / totalSteps) * 100;
        currentLayerIndex++;
        n_dims = 0;
        stringLen = 0;
        ftype = 0;
        dims = []
        {
            size = i32_s*3;
            
            if (cur + size > fileSize) {
                break;
            }

            const headerData = await readData(file, cur, size);
            cur += size;

            const hpView = new DataView(headerData);
            n_dims = hpView.getInt32(0*i32_s, true);
            stringLen  = hpView.getInt32(1*i32_s, true);
            ftype  = hpView.getInt32(2*i32_s, true);

            if (n_dims < 0 || stringLen < 0 || ftype < 0) {
                console.log("Detected an error");
                return;
            }

            size = i32_s*n_dims + stringLen;
            const restOfHeader = await readData(file, cur, size);
            const restOfHeaderView = new DataView(restOfHeader);
            cur += size;
            tensorSizeBytes = 1;
            if (n_dims === 0) {
                console.log("ERROR: Zero size tensors not permitted");
                return;
            }

            for (let i = 0; i < n_dims; ++i) {
                dim_size = restOfHeaderView.getInt32(i*i32_s, true)
                dims.push(dim_size);
                tensorSizeBytes *= dim_size;
            }

            // Ensure we have enough space.
            for (i = n_dims; i < 3; ++i) {
                dims.push(0);
            }

            // TODO Double check tensor dims when using quantization.

            if (ftype === ftype_f32) {
                tensorSizeBytes = tensorSizeBytes * 4;
            } else if (ftype === ftype_f16) {
                tensorSizeBytes = tensorSizeBytes * 2;
            } else if (ftype === ftype_q40 || ftype === ftype_q41) {
                console.log("ERROR: Quantized formats not supported yet");
                tensorSizeBytes = tensorSizeBytes / 2;
            }

            offset = n_dims*i32_s;

            let uint8Array = new Uint8Array(restOfHeader);
            let weightName = decoder.decode(uint8Array.subarray(offset, offset + stringLen));

            // Skip to next tensor (we will send data to C++).
            cur = Math.ceil(cur / 32) * 32;  // 32-byte alignment.
            cur += tensorSizeBytes;

            if (cur - lastChunkWrittenPtr > maxFileSize) {
                data = await readData(file, lastChunkWrittenPtr, lastWeightPtr - lastChunkWrittenPtr);
                lastChunkWrittenPtr = lastWeightPtr
                filename = file.name + "-chunk-" + ggmlChunk.toString().padStart(4, '0');
                await downloadGGMLChunk(filename, data, kFileType_Weights, numWeightsSoFar, 0, originalFileOffset);
                ++ggmlChunk;
                originalFileOffset = cur;
                numWeightsSoFar = 0;
            }

            ++numWeightsSoFar;
            
            lastWeightPtr = cur
        }
    }

    if (cur != lastChunkWrittenPtr) {
        data = await readData(file, lastChunkWrittenPtr, cur - lastChunkWrittenPtr);
        lastChunkWrittenPtr = lastWeightPtr
        await downloadGGMLChunk(file.name + "-chunk-" + ggmlChunk.toString().padStart(4, '0'), data, kFileType_Weights, numWeightsSoFar, 0, originalFileOffset);
        ++ggmlChunk;
        originalFileOffset = cur;
    }

    {
        const u32_s = 4;
        let data = new ArrayBuffer(u32_s);
        let view = new DataView(data);
        view.setUint32(u32_s*0, ggmlChunk + 1, true);

        // Now write a footer containing the number of files that should have been written.
        // This is to sanity check file loading.
        await downloadGGMLChunk(file.name + "-chunk-" + ggmlChunk.toString().padStart(4, '0'), data, kFileType_Footer, ggmlChunk + 1, 0, originalFileOffset);
        originalFileOffset = cur;
    }

    sendChatMessage(kSystemId, "Successfully chunked GGML file.");

    progressBar.style.display = "none";
}

function readData(file, start, size) {
  return new Promise((resolve, reject) => {
    const fileReader = new FileReader();
    
    fileReader.onload = (event) => {
      resolve(event.target.result);
    };
    
    fileReader.onerror = (error) => {
        console.log(error);
        reject('Error reading file:', error);
    };
    
    const blobSlice = file.slice(start, start + size);
    fileReader.readAsArrayBuffer(blobSlice);
  });
}


Module.onRuntimeInitialized = async _ => {
    //// The wasm module has been compiled and the runtime is ready.
    //// Now you can list the exported functions.
    //console.log("C/C++ FUNCTIONS")
    //for (let name in Module.asm) {
    //    if (typeof Module.asm[name] === 'function') {
    //        console.log(name);
    //    }
    //}
};

sendChatMessage(kSystemId, "Welcome to TokenHawk.\nPower your local LLms using WebGPU.\nCurrently, TokenHawk is in testing and only 7B-f16 llama models are supported.\nDue to file size limits in Chrome, use the 'Convert Model' button below in Firefox to split a 7B <i>lamma.cpp</i> GGML file into chunks. Then, in Chrome, you can load the resulting chunks using the 'Load Model' button ('Load Model' allows multiple file selection).\nMore details can be found <a href='https://github.com/kayvr/token-hawk'>here</a>.")
