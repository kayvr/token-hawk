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


function sendChatMessage(botId, message, messageId) {
    if (message.trim()) { // trim to prevent sending empty messages
        var chatlogs = document.getElementById('chatlogs');
        var lines = message.split('\n');
        var messageContainer = document.createElement('div');
        messageContainer.className = 'message';

        if (!(messageId === null || messageId === undefined || messageId.trim().length === 0)) {
            messageContainer.setAttribute('id', messageId);
        }

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

function updateMessageText(messageId, newMessage) {
    var messageElement = document.getElementById(messageId);

    if (messageElement) {
        var chatlogs = document.getElementById('chatlogs');
        // Find the message-text child div and update its contents
        var messageTextDiv = messageElement.getElementsByClassName('message-text')[0];
        if (messageTextDiv) {
            messageTextDiv.innerHTML = ''; // Clear the existing message text

            // Split the new message by line
            var lines = newMessage.split('\n');
            for (var i = 0; i < lines.length; i++) {
                var newMessageLine = document.createElement('p');
                if (lines[i].trim()) { // only append non-empty lines
                    newMessageLine.textContent = lines[i];
                } else { // Append <br> in place of empty lines
                    newMessageLine = document.createElement('br');
                }
                messageTextDiv.appendChild(newMessageLine);
            }
        }

        chatlogs.scrollTop = chatlogs.scrollHeight;
    } else {
        console.log(`Message with id ${messageId} not found.`);
    }
}

function submitMessage() {
    var messagebox = document.getElementById('message');
    var message = messagebox.value;

    sendChatMessage(kHumanId, message, null);

    Module.ccall('capi_on_human_message',
                 null,                      // return type
                 ['string'],                // argument type
                 [message]);

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
    var files = e.target.files; // This is a FileList object
    await loadFile(files[0]);
});

function timeoutSleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


async function loadFile(file) {
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
    
    {
        const size = vocabEnd;
        const data = await readData(file, 0, size);

        const dataPtr = Module._malloc(size);
        const dataView = new Uint8Array(data);
        Module.HEAPU8.set(dataView, dataPtr);
        Module._capi_load_model_header(dataPtr, size)
        Module._free(dataPtr);

        originalFileOffset = cur;
    }

    let decoder = new TextDecoder('utf-8');

    const ftype_f32 = 0;
    const ftype_f16 = 1;
    const ftype_q40 = 2;
    const ftype_q41 = 3;

    const maxFileSize = 550000000; // Limit for chrome.

    // Unfortunately, we don't know how many tensors are coming.
    // So we can't accurately gauge progress.
    // We can detect 7B, 13B, etc... by the number of layers
    // that are specified in the hyper parameters.
    // So we hardcode 7B values (32) for now.
    totalSteps = 32 * 9;

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
            console.log("weightName: " + weightName);

            // Skip to next tensor (we will send data to C++).
            cur = Math.ceil(cur / 32) * 32;  // 32-byte alignment.
            cur += tensorSizeBytes;

            {
                size = cur - originalFileOffset;
                data = await readData(file, originalFileOffset, size);

                const dataPtr = Module._malloc(size);
                const dataView = new Uint8Array(data);
                Module.HEAPU8.set(dataView, dataPtr);

                Module._capi_load_model_weights(dataPtr, originalFileOffset, size);
                Module._free(dataPtr);

                originalFileOffset = cur;
                numWeightsSoFar = 0;
            }

            ++numWeightsSoFar;
        }
    }

    if (Module._capi_model_end_load()) {
        sendChatMessage(kSystemId, "Successfully loaded model.", null);
    } else {
        sendChatMessage(kSystemId, "Failed to load GGML file.", null);
    }

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

sendChatMessage(kSystemId, "TokenHawk WebUI\nRequires Chrome Canary v115.0.5786.0 (or higher)\nMore details <a href='https://github.com/kayvr/token-hawk'>here</a>.", null)
