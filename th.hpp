#pragma once
// Conventions
// * Types are PascalCase.
//   - All members must be default initialized.
// * Functions are lower_snake_case
//   - If a type appears in a function name, use PascalCase.
// * Variables are camelCase (you'll see lower_snake_case is some locations);
// * Only one namespace: 'th'.
// * Member functions should use 'this->' to access member variables. Functions that transform
//   types are preferred. Not functions that transform data within a type.
// * Use modern memory management. Strongly prefer standard library memory management.
//
// Values
//  TokenHawk highly values GPU performance.
//
//  TokenHawk values parallelism on the GPU.
//
//  TokenHawk values GPU efficiency over simplicity.
//
//  TokenHawk values CPU simplicity over efficiency.
//
//  TokenHawk values purpose built optimizations for GPU hardware.

#include <webgpu/webgpu.h>
#include <assert.h>
#include <vector>
#include <string>

struct ggml_tensor;

namespace th {

enum TensorType {
    TensorType_Unknown,
    TensorType_F16,
    TensorType_F32,
};

TensorType get_TensorType_from_ggml(ggml_tensor* tensor);
std::string get_TensorType_name(TensorType dt);
void print_ggml_tensor_info(struct ggml_tensor * tensor, const char* tensorName);

inline size_t get_TensorType_size(TensorType dt) {
  // Note: We need to be careful with 4-bit or 3-bit types.
  if      (dt == TensorType_F16)  { return 2; }
  else if (dt == TensorType_F32)  { return 4; }
  else                            { assert(false); }
  return 0;
}

// This should remain an aggregate type.
struct TensorShape {
    int64_t l{};    // Layers.
    int64_t b{};    // Batches.
    int64_t r{};    // Rows.
    int64_t c{};    // Columns.

    int64_t get_total_num_elements() const {
        if (this->l == 0 && this->b == 0 && this->r == 0 && this->c == 0) {
          return 0;
        }
        int64_t size = 1;
        if (this->l > 0)   { size *= this->l; }
        if (this->b > 0)  { size *= this->b; }
        if (this->r > 0)     { size *= this->r;}
        if (this->c > 0)     { size *= this->c; }
        return size;
    }

    std::string to_string() const {
        std::string out = "";
        out += "L:" + std::to_string(this->l);
        out += " B:" + std::to_string(this->b);
        out += " R:" + std::to_string(this->r);
        out += " C:" + std::to_string(this->c);
        return out;
    }
    
    void print() const {
        printf("Shape: %s\n", this->to_string().c_str());
    }

    void canonicalize() {
        if (this->l == 1)    { this->l = 0; }
        if (this->b == 1)    { this->b = 0; }
        if (this->r == 0)    { this->r = 1; }
    }
};

TensorShape get_TensorShape_from_ggml(ggml_tensor* tensor);

inline bool operator==(const TensorShape& lhs, const TensorShape& rhs) {
    return (lhs.l == rhs.l && lhs.b == rhs.b && lhs.r == rhs.r && lhs.c == rhs.c);
}

//==========================
// GPU no-copy RAII wrapper
//==========================

struct TensorBuffer {
    static const WGPUBufferUsageFlags k_default_usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Storage;

    TensorBuffer() = default;
    TensorBuffer(TensorShape shape, TensorType type, WGPUDevice device = nullptr, WGPUBufferUsageFlags usage = k_default_usage);
    TensorBuffer(ggml_tensor* tensor, WGPUDevice device = nullptr, WGPUQueue queue = nullptr, WGPUBufferUsageFlags usage = k_default_usage);
    TensorBuffer(const void* data, TensorShape shape, TensorType type, bool backup, WGPUDevice device = nullptr, WGPUQueue queue = nullptr, WGPUBufferUsageFlags usage = k_default_usage);
    ~TensorBuffer() {
        free_buffers();
    }

    // No copying.
    TensorBuffer(const TensorBuffer& other) = delete;
    TensorBuffer& operator=(const TensorBuffer& other) = delete;
    
    // Only moving.
    TensorBuffer& operator=(TensorBuffer&& other) noexcept;
    TensorBuffer(TensorBuffer&& other) noexcept;

    size_t get_size_bytes() const;
    void allocate_gpu_memory(WGPUDevice device, WGPUBufferUsageFlags usage);
    void upload_ggml_to_gpu(WGPUQueue queue, ggml_tensor* tensor);
    void upload_data_to_gpu(WGPUQueue queue, const void* data);

    void reset_shape();

    int64_t get_num_dims() const {
        int64_t num_dimensions = 0;
        if (this->shape.l != 0) { num_dimensions++; }
        if (this->shape.b != 0) { num_dimensions++; }
        if (this->shape.r != 0) { num_dimensions++; }
        if (this->shape.c != 0) { num_dimensions++; }
        return num_dimensions;
    }

    void free_buffers() {
        if (this->gpu != nullptr) {
            wgpuBufferRelease(this->gpu);
            this->gpu = nullptr;
        }

        this->ram = nullptr;
    }

    size_t get_size() const {
        // Assert for types that have been tested.
        assert(this->type == TensorType_F32);
        assert(this->type == TensorType_F16);
        size_t type_size = get_TensorType_size(this->type);
        return this->shape.get_total_num_elements() * type_size;
    }

    bool is_valid() const {
        return (gpu != nullptr) || (cpuBackup.size() > 0);
    }

    // Before adding or removing any values here, make sure you properly
    // update the move constructor and move assignment operator.
    TensorShape           shape{};
    TensorType            type = TensorType_Unknown;

    //int64_t               disk{}; // Unused.
    bool                  cpuOnly = false;
    ggml_tensor*          ram{};
    std::vector<uint8_t>  cpuBackup{};
    WGPUBuffer            gpu{};

    TensorShape           originalShape{};
    
    std::string           name{};
};

struct ComputePipeline {
    ComputePipeline() = default;
    ComputePipeline(bool buildPipelineFlagIn) {
        buildPipelineFlag = buildPipelineFlagIn;
    }
    ~ComputePipeline() {
        free_buffers();
    }

    // No copying.
    ComputePipeline(const ComputePipeline& other) = delete;
    ComputePipeline& operator=(const ComputePipeline& other) = delete;
    
    // Only moving.
    ComputePipeline& operator=(ComputePipeline&& other) noexcept {
        if (this != &other) {
            free_buffers();
            this->bindGroupLayout = other.bindGroupLayout;
            this->pipeline = other.pipeline;
            this->bindGroup = other.bindGroup;
            this->buildPipelineFlag = other.buildPipelineFlag;
            this->sa = other.sa;
            this->sb = other.sb;
            this->sc = other.sc;
            this->ta = other.ta;
            this->tb = other.tb;
            this->tc = other.tc;
            other.bindGroupLayout = nullptr;
            other.pipeline = nullptr;
            other.bindGroup = nullptr;
        }
        return *this;
    }

    ComputePipeline(ComputePipeline&& other) noexcept
        : bindGroupLayout(other.bindGroupLayout)
        , pipeline(other.pipeline)
        , bindGroup(other.bindGroup)
        , buildPipelineFlag(other.buildPipelineFlag)
        , sa(other.sa)
        , ta(other.ta)
        , sb(other.sb)
        , tb(other.tb)
        , sc(other.sc)
        , tc(other.tc)
    {
        other.bindGroupLayout = nullptr;
        other.pipeline = nullptr;
        other.bindGroup = nullptr;
    }

    void free_buffers() {
        if (this->bindGroupLayout != nullptr) {
            wgpuBindGroupLayoutReference(this->bindGroupLayout);
            this->bindGroupLayout = nullptr;
        }
        
        if (this->pipeline != nullptr) {
            wgpuComputePipelineRelease(this->pipeline);
            this->pipeline = nullptr;            
        }

        if (this->bindGroup != nullptr) {
            wgpuBindGroupRelease(this->bindGroup);
            this->bindGroup = nullptr;            
        }
    }
    
    bool is_valid() {
        // bindGroup is intentionally left out of this check.
        return (this->bindGroupLayout != nullptr && this->pipeline != nullptr);
    }

    WGPUBindGroupLayout     bindGroupLayout{};
    WGPUComputePipeline     pipeline{};
    WGPUBindGroup           bindGroup{};      // Can be changed. Maybe sizes for each layer?
    
    bool buildPipelineFlag = false;
    
    // Validation variables. Keeps track of the sizes of input Tensors.
    TensorShape sa{}; TensorType ta = TensorType_Unknown;
    TensorShape sb{}; TensorType tb = TensorType_Unknown;
    TensorShape sc{}; TensorType tc = TensorType_Unknown;
};

struct CommandBuffer {
    CommandBuffer() = default;
    CommandBuffer(WGPUCommandBuffer buffer) {
        this->cmdBuffer = buffer;
    }
    ~CommandBuffer() {
        free_buffers();
    }

    // No copying.
    CommandBuffer(const CommandBuffer& other) = delete;
    CommandBuffer& operator=(const CommandBuffer& other) = delete;
    
    // Only moving.
    CommandBuffer& operator=(CommandBuffer&& other) noexcept {
        if (this != &other) {
            free_buffers();
            this->cmdBuffer = other.cmdBuffer;
            other.cmdBuffer = nullptr;
        }
        return *this;
    }

    CommandBuffer(CommandBuffer&& other) noexcept
        : cmdBuffer(other.cmdBuffer)
    {
        other.cmdBuffer = nullptr;
    }

    void free_buffers() {
        if (this->cmdBuffer != nullptr) {
            wgpuCommandBufferRelease(this->cmdBuffer);
            this->cmdBuffer = nullptr;
        }
    }
    
    bool is_valid() const {
        return (this->cmdBuffer != nullptr);
    }

    // Include timing information here?.
    WGPUCommandBuffer       cmdBuffer{};
};

void print_TensorBuffer(TensorBuffer* buffer, const char* bufferName);

bool are_pipelines_similar(const ComputePipeline& a, const ComputePipeline& b);

bool are_mat_mul_pipelines_the_same(const ComputePipeline& a, float scaleA, bool transposeA,
                                    const ComputePipeline& b, float scaleB, bool transposeB);

double get_time_seconds();

void print_descriptive_stats(std::vector<double> data, const std::string& dataType);

ComputePipeline create_compute_pipeline(
        WGPUDevice device,
        const char* wgslSource,
        std::vector<WGPUBindGroupLayoutEntry> layoutEntries,
        const char* label);

CommandBuffer cmdbuf_mat_mul(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        int transposeB = 0,
        WGPUBuffer uniforms = nullptr); // Note: we should just use a uniform buffer here.

CommandBuffer cmdbuf_transpose(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& inputF32,
        const TensorBuffer& outputF32,
        bool zy,
        WGPUBuffer dimBuffer);

CommandBuffer cmdbuf_rms_norm(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& A);

CommandBuffer cmdbuf_row_element_multiply(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& inputOutputBuffer,
        const TensorBuffer& rowMultBuffer);

CommandBuffer cmdbuf_RoPE(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& A,
        WGPUBuffer networkUniforms);    // Buffer should include n_past as the first u32 element

CommandBuffer cmdbuf_masked_softmax(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& inputOutputBuffer,
        WGPUBuffer dimBuffer);

CommandBuffer cmdbuf_row_softmax(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& inputOutputBuffer,
        WGPUBuffer dimBuffer);

CommandBuffer cmdbuf_addition(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& a,
        const TensorBuffer& b,
        const TensorBuffer& c);

CommandBuffer create_element_mult(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        const TensorBuffer& a, // out
        const TensorBuffer& b,
        const TensorBuffer& c);

CommandBuffer cmdbuf_element_mult_in_place(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& a, // out
        const TensorBuffer& b);

// TODO: Fused element multiply + silu.

CommandBuffer cmdbuf_silu(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& a);

CommandBuffer cmdbuf_vector_mat_mul_trans(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        int64_t aOffset);

CommandBuffer cmdbuf_vector_mat_mul_split_trans(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,      // Optional
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        const std::vector<TensorBuffer*>& scratchBuffers,
        // int32_t numSplits, // By default the number of splits is two.
        int64_t aOffset,
        const std::vector<WGPUBuffer>& splitBuffers,
        bool useDimsFromUniforms);

CommandBuffer cmdbuf_vector_multi_mat_mul_split_trans(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr)
        ComputePipeline* pipeline,
        const TensorBuffer& A,
        const std::vector<TensorBuffer*> B,
        const TensorBuffer& C,
        const std::vector<TensorBuffer*>& scratchBuffers,
        int64_t aOffset,
        const std::vector<WGPUBuffer>& splitBuffers,
        bool useDimsFromUniforms);

CommandBuffer cmdbuf_vector_reduce(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional
        WGPUComputePassEncoder pass,    // Optional
        ComputePipeline* pipeline,
        const TensorBuffer& A, // OUT
        const TensorBuffer& B,
        int numSplits);

// Only converts one row in B.
CommandBuffer cmdbuf_f16_f32_conversion(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr)
        ComputePipeline* pipeline,
        const TensorBuffer& A, // OUT
        const TensorBuffer& B,
        int numSplits,
        const int aOffset,
        const int bOffset);

} // namespace th
