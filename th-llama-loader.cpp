#include "th-llama-loader.hpp"

#include <span>
#include <cstdint>
#include <memory>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cinttypes>

namespace th {

static const int64_t kFileType_Header = 0;
static const int64_t kFileType_Weights = 1;
static const int64_t kFileType_Footer = 2;

static const int64_t kftype_f32 = 0;
static const int64_t kftype_f16 = 1;
static const int64_t kftype_q40 = 2;
static const int64_t kftype_q41 = 3;

static const uint32_t ggmlMagicUnversioned = 0x67676d6c;
static const uint32_t ggmlMagicValue = 0x67676a74;
static const uint32_t llamaFileVersion = 1;


static void readDataRaw(const std::span<const char>& buffer, size_t& offset, void* dst, size_t size) {
    std::memcpy(dst, buffer.data() + offset, size);
    offset += size;
    if (offset > buffer.size()) {
        fprintf(stderr, "ERROR: readData: Exceeded size of std::span!\n");
    }
}

template <typename T>
T readData(const std::span<const char>& buffer, size_t& offset) {
    T value;
    std::memcpy(&value, buffer.data() + offset, sizeof(T));
    offset += sizeof(T);
    if (offset > buffer.size()) {
        fprintf(stderr, "ERROR: readData: Exceeded size of std::span!\n");
    }
    return value;
}

bool load_header(th::LlamaModel* m, void* data, int64_t dataSize, int64_t /*vocabSize*/) {
    const std::span<const char> buffer((const char*)data, dataSize);
    size_t offset = 0;

    printf("Processing header!\n");

    uint32_t magic = readData<uint32_t>(buffer, offset);
    if (magic == ggmlMagicUnversioned) {
        printf("ERROR: load_header: Old version of magic.\n");
        return false;
    }

    if (magic != ggmlMagicValue) {
        printf("ERROR: load_header: Invalid magic value.\n");
        return false;
    }

    uint32_t formatVersion = readData<uint32_t>(buffer, offset);
    if (formatVersion != llamaFileVersion) {
        printf("ERROR: load_header: Invalid file version.\n");
        return false;
    }

    m->n_vocab = readData<uint32_t>(buffer,offset);
    m->n_embd  = readData<uint32_t>(buffer,offset);
    m->n_mult  = readData<uint32_t>(buffer,offset);
    m->n_head  = readData<uint32_t>(buffer,offset);
    m->n_layer = readData<uint32_t>(buffer,offset);
    m->n_rot   = readData<uint32_t>(buffer,offset);
    m->f16     = readData<uint32_t>(buffer,offset);

    const size_t tmpBufferSize = 128;
    std::vector<char> tmp(tmpBufferSize);

    th::LlamaVocab* v = &m->vocab;

    v->id_to_token.resize(m->n_vocab);

    std::string word;
    for (int i = 0; i < m->n_vocab; ++i) {
        uint32_t len = readData<int32_t>(buffer, offset);
        if (len > 8096) {
            printf("ERROR: load_header: Vocabulary element should not be larger than 8096.\n");
            return false;
        }

        word.resize(len);
        if (len > 0) {
            tmp.resize(len);
            readDataRaw(buffer, offset, tmp.data(), len);
            word.resize(len);
            word.assign(tmp.data(), len);
            //printf("New vocab word: %s\n", word.data());
        } else {
            word.clear();
        }

        float score = readData<float>(buffer, offset);

        v->token_to_id[word] = i;

        auto &tok_score = v->id_to_token[i];
        tok_score.tok = word;
        tok_score.score = score;
    }
    printf("Finished processing header!\n");

    if (offset != dataSize) {
        printf("Unknown left-over data\n");
    }

    return true;
}

bool load_weights(th::LlamaModel* m, WGPUDevice device, WGPUQueue queue, void* data, int64_t dataSize, int64_t numElementsInFile, int64_t originalFileOffset) {
    const std::span<const char> buffer((const char*)data, dataSize);
    size_t offset = 0;

    const size_t tmpBufferSize = 128;
    std::vector<char> tmp(tmpBufferSize);
    std::string tname;

    for (int i = 0; i < numElementsInFile; i++) {
        int32_t ndims       = readData<int32_t>(buffer, offset);
        int32_t stringLen   = readData<int32_t>(buffer, offset);
        int32_t ftype       = readData<int32_t>(buffer, offset);

        if (ndims < 0 || stringLen < 0 || ftype < 0) {
            printf("Detected an error\n");
            return false;
        }

        int64_t tensorSizeBytes = 1;
        TensorShape shape{};
        int ni = 0;
        if (ni < ndims) { shape.c = (int64_t)readData<int32_t>(buffer,offset); } ++ni;
        if (ni < ndims) { shape.r = (int64_t)readData<int32_t>(buffer,offset); } ++ni;
        if (ni < ndims) { shape.b = (int64_t)readData<int32_t>(buffer,offset); } ++ni;
        int64_t c = shape.c; if (c == 0) { c = 1; }
        int64_t r = shape.r; if (r == 0) { r = 1; }
        int64_t b = shape.b; if (b == 0) { b = 1; }
        tensorSizeBytes = c*r*b;

        TensorType tensorType = TensorType_Unknown;
        if (ftype == kftype_f32) {
            tensorSizeBytes = tensorSizeBytes * 4;
            tensorType = TensorType_F32;
        } else if (ftype == kftype_f16) {
            tensorSizeBytes = tensorSizeBytes * 2;
            tensorType = TensorType_F16;
        } else if (ftype == kftype_q40 || ftype == kftype_q41) {
            printf("ERROR: Quantized formats not supported yet\n");
            tensorSizeBytes = tensorSizeBytes / 2;
        }

        tname.resize(stringLen);
        if (stringLen > 0) {
            tmp.resize(stringLen);
            readDataRaw(buffer, offset, tmp.data(), stringLen);
            tname.resize(stringLen);
            tname.assign(tmp.data(), stringLen);
        } else {
            tname.clear();
        }

        // Skip to next tensor (we will send data to C++).
        int64_t alignmentOffset = originalFileOffset + offset;
        int64_t alignmentOffsetDelta = (alignmentOffset + 31) & -32;
        offset += alignmentOffsetDelta - alignmentOffset;

        shape.canonicalize();
        
        // Don't place tok_embeddings on the GPU.
        if (tname == "tok_embeddings.weight") {
            if (kUseGpuEmbeddingSelection) {
                // We now copy and convert embeddings on the GPU.
                m->loadedMapping[tname] = th::TensorBuffer((const void*)(buffer.data() + offset), shape, tensorType, false, device, queue, th::TensorBuffer::k_default_usage);
            } else {
                printf("Found tok_embeddings.weight. Reprocessing tokens into f32.\n");
                // We convert tok_embeddings to float32 from float16. Somewhat expensive.
                std::vector<float> newBuff(shape.get_total_num_elements());
                const std::span<const char> f16tof32_conv_buff((const char*)buffer.data() + offset, shape.get_total_num_elements()*sizeof(uint16_t));
                size_t convBuffOffset = 0;
                for (int i = 0; i < shape.get_total_num_elements(); ++i) {
                    newBuff[i] = ggml_compute_fp16_to_fp32(readData<uint16_t>(f16tof32_conv_buff, convBuffOffset));
                }
                //m->loadedMapping[tname] = std::move(th::TensorBuffer((const void*)(buffer.data() + offset), shape, tensorType, true));
                m->loadedMapping[tname] = th::TensorBuffer(newBuff.data(), shape, TensorType_F32, true);
                printf("Finished reprocessing.\n");
            }
        } else if (tname == "output.weight") {
            m->loadedMapping[tname] = th::TensorBuffer((const void*)(buffer.data() + offset), shape, tensorType, false, device, queue, th::TensorBuffer::k_default_usage);
            //if (!kSplitFinalMultiply) {
            //} else {
            {
                // Split apart the buffer before storing the contents.

                int64_t origStride = shape.c * get_TensorType_size(tensorType);
                int64_t newStride = origStride / 2;

                int64_t newBufferSize = newStride * shape.r;

                TensorShape newShape = shape;
                newShape.c = newShape.c / 2;

                uint8_t* origData = (uint8_t*)(buffer.data() + offset);

                printf("Reprocessing output matrix...\n");
                // To avoid allocating too much memory (a problem in WASM)
                // we perform one buffer at a time then release the memory
                // backing of m->outputMat afterwards.
                {
                    std::vector<uint8_t> buffer1;
                    buffer1.resize(newBufferSize);

                    for (int r = 0; r < shape.r; ++r) {
                        memcpy(&buffer1[r*newStride], &origData[r*origStride], newStride);
                    }

                    m->loadedMapping[tname + "-split1"] = th::TensorBuffer(buffer1.data(), newShape, tensorType, false, device, queue);
                }

                {
                    std::vector<uint8_t> buffer2;
                    buffer2.resize(newBufferSize);

                    for (int r = 0; r < shape.r; ++r) {
                        memcpy(&buffer2[r*newStride], &origData[r*origStride + newStride], newStride);
                    }

                    m->loadedMapping[tname + "-split2"] = th::TensorBuffer(buffer2.data(), newShape, tensorType, false, device, queue);
                    //m->outputMatSplit2 = std::move(th::TensorBuffer(buffer2.data(), newShape, m->outputMat.type, true, device, queue));
                }

                printf("Finished...\n");
            }
        } else {
            m->loadedMapping[tname] = th::TensorBuffer((const void*)(buffer.data() + offset), shape, tensorType, false, device, queue, th::TensorBuffer::k_default_usage);

            // Our tensor is at 'offset' of tensorSizeBytes.
            if (m->loadedMapping[tname].get_size_bytes() != tensorSizeBytes) {
                printf("ERROR: Unexpected tensor size!\n");
                return false;
            }
            if (!m->loadedMapping[tname].gpu) {
                printf("ERROR: Failed to upload to GPU.\n");
                return false;
            }
        }

        offset += tensorSizeBytes;
    }

    if (offset != dataSize) {
        printf("Unknown left-over data %zu %lld\n", offset, dataSize);
    }

    return true;
}

bool load_footer(th::LlamaModel* m, void* data, int64_t dataSize) {
    const std::span<const char> buffer((const char*)data, dataSize);
    size_t offset = 0;

    m->targetFilesLoaded = readData<uint32_t>(buffer,offset);
    return true;
}

void load_model_chunk(th::LlamaModel* m, WGPUDevice device, WGPUQueue queue, void* data, int64_t dataSize) {
    const uint16_t th_magic = 0x1737;
    const uint16_t th_version = 1;

    const std::span<const char> buffer((const char*)data, dataSize);
    size_t offset = 0;

    // Load the file header.
    uint16_t magic = readData<uint16_t>(buffer,offset);
    if (magic != th_magic) {
        printf("Load failure: Invalid magic value.\n");
        return;
    }

    uint16_t version = readData<uint16_t>(buffer,offset);
    if (version != th_version) {
        printf("Load failure: Invalid model version.\n");
        return;
    }

    uint32_t fileType = readData<uint32_t>(buffer,offset);
    uint32_t numElementsInFile = readData<uint32_t>(buffer,offset);
    uint32_t vocabSize = readData<uint32_t>(buffer,offset);

    int64_t originalFileOffset = readData<int64_t>(buffer,offset);
    int64_t padding = readData<int64_t>(buffer,offset);
    (void)padding;

    char* payloadData = (char*)data;
    payloadData += offset;

    int64_t payloadDataSize = dataSize - offset;

    bool success = true;
    if (fileType == kFileType_Header) {
        success = load_header(m, payloadData, payloadDataSize, vocabSize);
    } else if (fileType == kFileType_Weights) {
        success = load_weights(m, device, queue, payloadData, payloadDataSize, numElementsInFile, originalFileOffset);
    } else if (fileType == kFileType_Footer) {
        success = load_footer(m, payloadData, payloadDataSize);
    } else {
        printf("FAILURE\n");
        success = false;
    }

    if (success) {
        m->numFilesLoaded++;
    } else {
        m->loadFailed = true;
        printf("Unsuccessful load!\n");
    }

    return;
}

void post_load_init_model(WGPUDevice device, WGPUQueue queue, std::shared_ptr<th::LlamaModel> m) {
    // Initialize the model.
    int32_t seed = 780658349;
    m->rng = std::mt19937(seed);

    TensorShape kv_cache_shape = TensorShape{ .l=0, .b=m->n_ctx, .r=m->n_head, .c=m->n_embd/m->n_head};

    // Working buffers for input/output.
    m->working_key_cache = TensorBuffer(kv_cache_shape, TensorType_F32, device);
    m->working_val_cache = TensorBuffer(kv_cache_shape, TensorType_F32, device);

    TensorShape inp_shape = TensorShape{.l=0, .b=0, .r=m->n_batch, .c=m->n_embd};
    for (int i = 0; i < LlamaModel::nInpBuffers; ++i) {
        m->inp[i] = TensorBuffer(inp_shape, TensorType_F32, device);
    }
    for (int i = 0; i < LlamaModel::nSplitScratch; ++i) {
        m->splitScratch[i] = TensorBuffer(inp_shape, TensorType_F32, device);
    }

    int n_ff = ((2*(4*m->n_embd)/3 + m->n_mult - 1)/m->n_mult)*m->n_mult;
    assert(n_ff == 11008); // Presumably this value is only relevant for 7B.
    TensorShape ffshape = TensorShape{.l=0, .b=0, .r=m->n_batch, .c=n_ff};
    m->ffWorking[0] = TensorBuffer(ffshape, TensorType_F32, device);
    m->ffWorking[1] = TensorBuffer(ffshape, TensorType_F32, device);
    
    m->norm = std::move(m->loadedMapping["norm.weight"]);
    m->outputMat = std::move(m->loadedMapping["output.weight"]);
    m->outputMatSplit1 = std::move(m->loadedMapping["output.weight-split1"]);
    m->outputMatSplit2 = std::move(m->loadedMapping["output.weight-split2"]);

    m->out = TensorBuffer({.r=1, .c=m->outputMat.shape.r}, TensorType_F32, device);
    m->outScratch = TensorBuffer({.r=1, .c=m->outputMat.shape.r}, TensorType_F32, device);

    m->resultBuffer = TensorBuffer(m->out.shape, m->out.type, device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    {
        LlamaNetworkUniforms uniforms{};
        size_t size = kLlamaUniformsSize;
        WGPUBufferDescriptor bufferDesc = {};
        bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
        bufferDesc.size  = size;
        m->networkUniforms = wgpuDeviceCreateBuffer(device, &bufferDesc);
        wgpuQueueWriteBuffer(queue, m->networkUniforms, 0, &uniforms, size);
    }

    {
        LlamaTensorDimsUniforms uniforms{};
        size_t size = kLlamaUniformsSize;
        WGPUBufferDescriptor bufferDesc = {};
        bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
        bufferDesc.size  = size;
        for (int i = 0; i < (int)m->dimsUniforms.size(); ++i) {
            m->dimsUniforms[i] = wgpuDeviceCreateBuffer(device, &bufferDesc);
            wgpuQueueWriteBuffer(queue, m->dimsUniforms[i], 0, &uniforms, size);
        }
    }

    {
        LlamaVecMatSplitDimsUniforms uniforms{
            .A_B = 0,
            .A_M = 0,
            .A_N = 0,
            .split = 0,
            .B_B = 0,
            .B_M = 0,
            .B_N = 0,
            .totalSplits = LlamaModel::nSplits,
        };
        size_t size = kLlamaUniformsSize;
        WGPUBufferDescriptor bufferDesc = {};
        bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
        bufferDesc.size  = size;
        for (int i = 0; i < LlamaModel::nSplits; ++i) {
            m->splitBuffers.push_back(wgpuDeviceCreateBuffer(device, &bufferDesc));
            uniforms.split = i;
            wgpuQueueWriteBuffer(queue, m->splitBuffers[i], 0, &uniforms, size);
            //m->splitTensors.push_back(std::move(TensorBuffer(inp_shape, TensorType_F32, device)));
        }
    }
    
    m->tok_embeddings = std::move(m->loadedMapping["tok_embeddings.weight"]);
    for (int i = 0; i < m->n_layer; ++i) {
        th::LlamaLayer newLayer{};
        newLayer.index = i;

        newLayer.attention_norm = std::move(m->loadedMapping["layers." + std::to_string(i) + ".attention_norm.weight"]);
        
        newLayer.wq = std::move(m->loadedMapping["layers." + std::to_string(i) + ".attention.wq.weight"]);
        newLayer.wk = std::move(m->loadedMapping["layers." + std::to_string(i) + ".attention.wk.weight"]);
        newLayer.wv = std::move(m->loadedMapping["layers." + std::to_string(i) + ".attention.wv.weight"]);
        newLayer.wo = std::move(m->loadedMapping["layers." + std::to_string(i) + ".attention.wo.weight"]);

        newLayer.ffn_norm = std::move(m->loadedMapping["layers." + std::to_string(i) + ".ffn_norm.weight"]);

        newLayer.w1 = std::move(m->loadedMapping["layers." + std::to_string(i) + ".feed_forward.w1.weight"]);
        newLayer.w2 = std::move(m->loadedMapping["layers." + std::to_string(i) + ".feed_forward.w2.weight"]);
        newLayer.w3 = std::move(m->loadedMapping["layers." + std::to_string(i) + ".feed_forward.w3.weight"]);

        newLayer.key_cache = TensorBuffer(kv_cache_shape, TensorType_F32, device);
        newLayer.value_cache = TensorBuffer(kv_cache_shape, TensorType_F32, device);

        m->layers.push_back(std::move(newLayer));
    }

    build_pipelines_llama(device, queue, m);
}

std::shared_ptr<LlamaModel> load_llama_chunked(
        WGPUDevice device, WGPUQueue queue, const std::string& dir) {
    std::shared_ptr<th::LlamaModel> model = std::make_shared<th::LlamaModel>();

    try {
        std::filesystem::directory_iterator dir_iter(dir);

        for (const auto& entry : dir_iter) {
            std::cout << entry.path() << std::endl;

            std::string filename = entry.path();

            std::ifstream file(filename, std::ios::binary | std::ios::ate);

            if (!file) {
                fprintf(stderr, "Unable to open file: %s\n", filename.c_str());
                return {};
            }

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            // Create a vector to hold the data
            std::vector<char> buffer(size);

            if (!file.read(buffer.data(), size)) {
                fprintf(stderr, "Unable to read file %s\n", filename.c_str());
                return {};
            }

            load_model_chunk(model.get(), device, queue, buffer.data(), size);
        }
    } catch(std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }

    if (model->targetFilesLoaded != model->numFilesLoaded) {
        fprintf(stderr, "ERROR: Not all files loaded!\n");
        return {};
    }

    post_load_init_model(device, queue, model);

    model->loadedMapping.clear();
    
    return model;
}

std::shared_ptr<LlamaModel> load_llama_file(
        WGPUDevice device, WGPUQueue queue, const std::string& filename) {
    std::shared_ptr<th::LlamaModel> m = std::make_shared<th::LlamaModel>();

    std::ifstream fin(filename, std::ios::binary | std::ios::ate);

    if (!fin) {
        fprintf(stderr, "Unable to open file: %s\n", filename.c_str());
        return {};
    }

    fin.seekg(0, std::ios::beg);

    // Most of the code below is copied from llama.cpp.
    // There is serious code duplication with the above loading functions and
    // the javascript version. Major rethinking and refactoring is needed.
    // Specifically 'load_header' is duplicated.
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic == ggmlMagicUnversioned) {
            fprintf(stderr, "%s: invalid model file '%s' (too old, regenerate your model files or convert them with convert-unversioned-ggml-to-ggml.py!)\n",
                    __func__, filename.c_str());
            return {};
        }
        if (magic != ggmlMagicValue) {
            printf("%s: Bad magic\n", __func__);
            return {};
        }

        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != llamaFileVersion) {
            fprintf(stderr, "%s: invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n",
                    __func__, filename.c_str(), format_version, llamaFileVersion);
            return {};
        }
    }

    // load hparams
    {
        fin.read((char *) &m->n_vocab, sizeof(m->n_vocab));
        fin.read((char *) &m->n_embd,  sizeof(m->n_embd));
        fin.read((char *) &m->n_mult,  sizeof(m->n_mult));
        fin.read((char *) &m->n_head,  sizeof(m->n_head));
        fin.read((char *) &m->n_layer, sizeof(m->n_layer));
        fin.read((char *) &m->n_rot,   sizeof(m->n_rot));
        fin.read((char *) &m->f16,     sizeof(m->f16));
    }

    // load vocab
    {
        const size_t tmpBufferSize = 128;
        std::vector<char> tmp(tmpBufferSize);

        th::LlamaVocab& vocab = m->vocab;

        std::string word;
        vocab.id_to_token.resize(m->n_vocab);

        for (int i = 0; i < m->n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            if (len > 0) {
                tmp.resize(len);
                fin.read(tmp.data(), len);
                word.assign(tmp.data(), len);
            } else {
                word.clear();
            }

            float score;
            fin.read((char *) &score, sizeof(score));

            vocab.token_to_id[word] = i;

            auto &tok_score = vocab.id_to_token[i];
            tok_score.tok = word;
            tok_score.score = score;
        }
    }

    // Get the size of each set of weights. Then call load_weights
    std::vector<char> tensorData(128*1024*1024);
    while (true) {
        std::streamsize weightsBegin = fin.tellg();

        int32_t n_dims;
        int32_t length;
        int32_t ftype;

        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
        fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

        if (fin.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        size_t tensorSizeBytes = nelements;
        if (ftype == kftype_f32) {
            tensorSizeBytes = tensorSizeBytes * 4;
        } else if (ftype == kftype_f16) {
            tensorSizeBytes = tensorSizeBytes * 2;
        } else if (ftype == kftype_q40 || ftype == kftype_q41) {
            printf("ERROR: Quantized formats not supported yet\n");
            tensorSizeBytes = tensorSizeBytes / 2;
        }


        std::string name(length, 0);
        fin.read(&name[0], length);

        std::streamsize currentOffset = fin.tellg();
        // Skip to next tensor (we will send data to C++).
        int64_t alignmentOffsetDelta = (currentOffset + 31) & -32;
        currentOffset += alignmentOffsetDelta - currentOffset;
        
        currentOffset += tensorSizeBytes;

        size_t sizeToRead = currentOffset - weightsBegin;
        fin.seekg(weightsBegin, std::ios::beg);
        tensorData.resize(sizeToRead);
        fin.read(tensorData.data(), sizeToRead);

        load_weights(m.get(), device, queue, tensorData.data(), tensorData.size(), 1, weightsBegin);
    }

    fin.close();

    post_load_init_model(device, queue, m);

    for (const auto& pair : m->loadedMapping) {
        if (pair.second.is_valid()) {
            printf("FOUND A VALID TENSOR: %s\n", pair.first.c_str());
        }
    }
    m->loadedMapping.clear();
    
    return m;
}

} // namespace th
