#include "th.hpp"
#include "th-llama.hpp" // For kUniformsSize

#include "math.h"

#include <vector>
#include <chrono>
#include <cstring>
#include <cinttypes>
#include <numeric>

namespace th {

std::string get_TensorType_name(TensorType dt) {
    switch (dt) {
        case TensorType_F16: return "TensorTypeF16";
        case TensorType_F32: return "TensorTypeF32";
        default: assert(false);
    }
    return "TensorType_Unknown";
}

double get_time_seconds() {
  auto t0 = std::chrono::high_resolution_clock::now();        
  auto nanosec = t0.time_since_epoch();

  return (nanosec.count() / 1000000000.0);
}

static double percentile(const std::vector<double>& sortedValues, double percent) {
    if(sortedValues.empty()) return 0;
    
    if (percent <= 0) return sortedValues[0];
    else if (percent >= 1) return sortedValues.back();
    
    double index = (sortedValues.size() - 1) * percent;
    size_t lower = std::floor(index);
    size_t upper = std::ceil(index);
    double weight = index - lower;
    
    if(upper >= sortedValues.size()) return sortedValues[lower];
    return (1 - weight) * sortedValues[lower] + weight * sortedValues[upper];
}

void print_descriptive_stats(std::vector<double> runTimes, const std::string& dataType) {
    // Calculate mean
    double sum = std::accumulate(runTimes.begin(), runTimes.end(), 0.0);
    double mean = sum / runTimes.size();

    // Sort the runtimes for median and percentile calculations
    std::sort(runTimes.begin(), runTimes.end());

    // Calculate median
    size_t size = runTimes.size();
    double median = (size % 2 == 0) ? 
                    (runTimes[size / 2 - 1] + runTimes[size / 2]) / 2.0 : 
                    runTimes[size / 2];

    // Calculate mode
    double mode = *std::max_element(runTimes.begin(), runTimes.end(), 
                    [&runTimes](double a, double b){
                        return std::count(runTimes.begin(), runTimes.end(), a) < 
                               std::count(runTimes.begin(), runTimes.end(), b);
                    });

    // Calculate standard deviation
    double sq_sum = std::inner_product(runTimes.begin(), runTimes.end(), runTimes.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / runTimes.size() - mean * mean);

    // Output results
    printf("\n");
    printf("Mean: %8.2f%s\n", mean, dataType.c_str());
    printf("Median: %8.2f%s\n", median, dataType.c_str());
    printf("Mode: %8.2f%s\n", mode, dataType.c_str());
    printf("StdDev: %8.2f%s\n", std_dev, dataType.c_str());

    // Calculate percentiles
    double percentile99 = percentile(runTimes, 0.99);
    double percentile95 = percentile(runTimes, 0.95);
    double percentile5 = percentile(runTimes, 0.05);
    double percentile1 = percentile(runTimes, 0.01);
    
    printf("99th Percentile: %8.2f%s\n", percentile99, dataType.c_str());
    printf("95th Percentile: %8.2f%s\n", percentile95, dataType.c_str());
    printf("5th Percentile: %8.2f%s\n", percentile5, dataType.c_str());
    printf("1st Percentile: %8.2f%s\n", percentile1, dataType.c_str());
}

static void store_pipeline_validation(
        ComputePipeline& pipeline,
        const TensorBuffer* A,
        const TensorBuffer* B = nullptr,
        const TensorBuffer* C = nullptr) {
    if (A) { pipeline.sa = A->shape; pipeline.ta = A->type; }
    if (B) { pipeline.sb = B->shape; pipeline.tb = B->type; }
    if (C) { pipeline.sc = C->shape; pipeline.tc = C->type; }
}

static bool validate_pipeline(
        const ComputePipeline& pipeline,
        const TensorBuffer* A,
        const TensorBuffer* B = nullptr,
        const TensorBuffer* C = nullptr) {
    if (A) {
        if (!(pipeline.sa == A->shape) || pipeline.ta != A->type) {
            assert(false);
            return false;
        }
    }
    if (B) {
        if (pipeline.sb != B->shape || pipeline.tb != B->type) {
            assert(false);
            return false;
        }
    }
    if (C) {
        if (pipeline.sc != C->shape || pipeline.tc != C->type) {
            assert(false);
            return false;
        }
    }
    
    return true;
}

bool are_pipelines_similar(const ComputePipeline& a, const ComputePipeline& b) {
    if (a.sa != b.sa || a.sb != b.sb || a.sc != b.sc) {
        return false;
    }
    if (a.ta != b.ta || a.tb != b.tb || a.tc != b.tc) {
        return false;
    }
    return true;
}

bool are_mat_mul_pipelines_the_same(const ComputePipeline& a, float scaleA, bool transposeA,
                                    const ComputePipeline& b, float scaleB, bool transposeB) {
    if (!are_pipelines_similar(a, b)) {
        return false;
    }
    if (scaleA != scaleB) {
        return false;
    }
    if (transposeA != transposeB) {
        return false;
    }
    return true; 
}

TensorBuffer::TensorBuffer(TensorShape shapeIn, TensorType typeIn, WGPUDevice device, WGPUBufferUsageFlags usage) {
    this->shape = shapeIn;
    this->type = typeIn;
    this->originalShape = shape;

    if (device) { allocate_gpu_memory(device, usage); }
}

TensorBuffer::TensorBuffer(const void* data, TensorShape shapeIn, TensorType type, bool backup, WGPUDevice device, WGPUQueue queue, WGPUBufferUsageFlags usage) {
    this->shape = shapeIn;
    this->type = type;
    if (backup) {
        size_t size = this->get_size_bytes();
        this->cpuBackup.resize(size);
        memcpy(this->cpuBackup.data(), data, size);
    }
    this->originalShape = shape;
  
    if (device) { allocate_gpu_memory(device, usage); }
    if (queue)  { upload_data_to_gpu(queue, data); }
}

TensorBuffer& TensorBuffer::operator=(TensorBuffer&& other) noexcept {
    if (this != &other) {
        free_buffers();
        this->shape = other.shape;
        this->type = other.type;
        this->gpu = other.gpu;
        this->cpuOnly = other.cpuOnly;
        this->gpu = other.gpu;
        this->originalShape = other.originalShape;
        this->cpuBackup = other.cpuBackup;
        this->name = other.name;
        other.gpu = nullptr;
        other.cpuBackup.clear();
    }
    return *this;
}

TensorBuffer::TensorBuffer(TensorBuffer&& other) noexcept
    : shape(other.shape)
    , type(other.type)
    , cpuOnly(other.cpuOnly)
    , cpuBackup(other.cpuBackup)
    , gpu(other.gpu)
    , originalShape(other.originalShape)
    , name(other.name)
{
    other.gpu = nullptr;
    other.cpuBackup.clear();
}

size_t TensorBuffer::get_size_bytes() const {
    assert(this->type != TensorType_Unknown);
    int64_t elementSize = get_TensorType_size(this->type);
    int64_t size = shape.get_total_num_elements();
    return size * elementSize;
}

void TensorBuffer::upload_data_to_gpu(WGPUQueue queue, const void* data) {
    assert(this->gpu);
    wgpuQueueWriteBuffer(queue, this->gpu, 0, data, this->get_size_bytes());
}

void TensorBuffer::allocate_gpu_memory(WGPUDevice device, WGPUBufferUsageFlags usage) {
    // TODO Adjust size of buffer if we have a format that is less than 8 bits.
    // get_TensorType_size
    assert(!this->gpu);
    assert(this->type != TensorType_Unknown);
    size_t size = get_size_bytes();

    WGPUBufferDescriptor bufferDesc = {};
    bufferDesc.usage = usage;
    bufferDesc.size  = size;
    this->gpu = wgpuDeviceCreateBuffer(device, &bufferDesc);
}

void TensorBuffer::reset_shape() {
    this->shape = this->originalShape;
}

ComputePipeline create_compute_pipeline(
        WGPUDevice device,
        const char* wgslSource,
        std::vector<WGPUBindGroupLayoutEntry> layoutEntries,
        const char* label)
{
    ComputePipeline out{};

    WGPUShaderModule cs;
    {
      WGPUShaderModuleWGSLDescriptor wgsl = {};
      wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
      wgsl.source = wgslSource;
      WGPUShaderModuleDescriptor desc = {};
      desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl);
      desc.label = label;
      cs = wgpuDeviceCreateShaderModule(device, &desc);
    }

    WGPUBindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = layoutEntries.size();
    bglDesc.entries = layoutEntries.data();
    out.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    // pipeline layout (used by the render pipeline, released after its creation)
    WGPUPipelineLayoutDescriptor layoutDesc = {};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &out.bindGroupLayout;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);

    WGPUProgrammableStageDescriptor stageDesc{};
    stageDesc.module = cs;
    stageDesc.entryPoint = "main";
    stageDesc.constantCount = 0;
    stageDesc.constants = nullptr;

    WGPUComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.label = label;
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.compute = stageDesc;

    out.pipeline = wgpuDeviceCreateComputePipeline(device, &pipelineDesc);

    wgpuShaderModuleRelease(cs);
    wgpuPipelineLayoutRelease(pipelineLayout);

    return out;
}

void print_TensorBuffer(TensorBuffer* buffer, const char* bufferName) {
    const auto & shape = buffer->shape;
    int64_t size = buffer->get_size_bytes();

    fprintf(stdout, "\n");
    fprintf(stdout, "GPU buffer info (%s):\n", bufferName);
    fprintf(stdout, "    size = %" PRId64 "\n", size);
    fprintf(stdout, "    shape = l:%" PRId64 " b:%" PRId64 " r:%" PRId64 " c:%" PRId64 "\n", shape.l, shape.b, shape.r, shape.c);
    fprintf(stdout, "    type = %s\n", get_TensorType_name(buffer->type).c_str());
}

// Originally from llama.cpp
// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16
float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

uint32_t fp32_to_bits(float f) {
	union {
		float as_value;
		uint32_t as_bits;
	} fp32;
	fp32.as_value = f;
	return fp32.as_bits;
}

float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}



static const char* wgsl_fp16_to_fp32 = R"(
// Takes a u32 that contains two f16's, and an index to the f16
// in the u32.
fn compute_fp16_to_fp32(h_in : u32, f16_part : u32) -> f32 {
    var h : u32 = h_in >> (16 * f16_part);  // Select correct 16 bits.
    h = h & u32(0xFFFF);                  // Mask off upper 16 bits.

    let w : u32 = h << 16;
    let sign : u32 = w & u32(0x80000000);
    let two_w : u32 = w + w;

    let exp_offset : u32 = u32(0xE0) << 23;
    //const float exp_scale = 0x1.0p-112f;
    let exp_scale : f32 = bitcast<f32>(u32(0x7800000));
    let normalized_value : f32 = bitcast<f32>((two_w >> 4) + exp_offset) * exp_scale;

    let magic_mask : u32 = u32(126) << 23;
    //let magic_mask : u32 = u32(126) << 20;
    let magic_bias : f32 = 0.5f;
    let denormalized_value : f32 = bitcast<f32>((two_w >> 17) | magic_mask) - magic_bias;

    let denormalized_cutoff : u32 = u32(1) << 27;
    var result : u32;
    if (two_w < denormalized_cutoff) {
        result = bitcast<u32>(denormalized_value);
    } else {
        result = bitcast<u32>(normalized_value);
    }
    result = sign | result;
    return bitcast<f32>(result);
}
)";

static const char* wgsl_gemm_header = R"(
// Note that bindings similar to the below exist:
//@group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
//@group(0) @binding(1) var<storage,read> b : array<f32>; // [K, N]
//@group(0) @binding(2) var<storage,read_write> c : array<f32>;

const kSharedMemDimX : u32 = kWorkgroupX * kTileSizeX;
const kSharedMemDimY : u32 = kWorkgroupY * kTileSizeY;

var<workgroup> aShared : array<f32, kSharedMemDimX*kSharedMemDimY>;
var<workgroup> bShared : array<f32, kSharedMemDimX*kSharedMemDimY>;

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupX, kWorkgroupY, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
)";

static const char* wgsl_gemm_body = R"(
  let matStrideA : u32 = M * K * workGroupID.z;
  let matStrideB : u32 = K * N * workGroupID.z;
  let matStrideOut : u32 = M * N * workGroupID.z;

  // Each thread loads 4x4 tile into the shared workgroup memory.
  let outputIndexA = vec2<u32>(
    workGroupID.xy * vec2<u32>(kSharedMemDimX, kSharedMemDimY) +
    localInvocationId.xy * vec2<u32>(kTileSizeX, kTileSizeY));

  var suma : array<array<f32,kTileSizeY>, kTileSizeX>;

  for (var j : u32 = 0; j < kTileSizeY; j = j + 1) {
    for (var i : u32 = 0; i < kTileSizeX; i = i + 1) {
      suma[i][j] = 0.0;
    }
  }

  var numLoopsNeeded : u32 = K / kSharedMemDimX;
  if (K % kSharedMemDimX != 0) {
    numLoopsNeeded = numLoopsNeeded + 1;
  }

  var bRows : u32 = K;
  var bCols : u32 = N;
  if (kTransposeB == 1) {
    bRows = N;
    bCols = K;
  }

  for (var highi : u32 = 0; highi < numLoopsNeeded; highi = highi + 1) {
    // The idea here: workGroupID specifies the 'block' within the matrix
    // that we are working on (will receive our results). highi represents
    // how far we are along the 'x' axis of 'A' and 'y' axis of 'B'.
    let a_row = workGroupID.y * kSharedMemDimY + localInvocationId.y * kTileSizeY;
    let a_col = highi * kSharedMemDimX + localInvocationId.x * kTileSizeX;

    var b_row = highi * kSharedMemDimY + localInvocationId.y * kTileSizeY;
    var b_col = workGroupID.x * kSharedMemDimX + localInvocationId.x * kTileSizeX;

    if (kTransposeB == 1) {
        b_row = workGroupID.x * kSharedMemDimX + localInvocationId.x * kTileSizeX;
        b_col = highi * kSharedMemDimY + localInvocationId.y * kTileSizeY;
    }

    for (var row : u32 = 0; row < kTileSizeY; row = row + 1) {
      let rowPos = (localInvocationId.y * kTileSizeY + row) * kSharedMemDimX + localInvocationId.x * kTileSizeX;
      for (var col : u32 = 0; col < kTileSizeX; col = col + 1) {
        let ax = a_col + col;
        let ay = a_row + row;
        let bx = b_col + col;
        let by = b_row + row;
        if (!isOutOfBounds(M, K, ax, ay))
        {
          aShared[rowPos + col] = a[matStrideA + ay * K + ax];
        }
        else
        {
          aShared[rowPos + col] = 0.0; // MUST 
        }
        if (!isOutOfBounds(bRows, bCols, bx, by))
        {
          // TODO Transpose b when placing in shared memory. Leads to better smem locality.
          if (is_b_f16) {
              let b_index : u32 = by * (bCols/2) + bx / 2;
              let f16_part = bx % 2;
              bShared[rowPos + col] = compute_fp16_to_fp32(u32(b[matStrideB + b_index]), f16_part); // f16
          } else {
              bShared[rowPos + col] = f32(b[matStrideB + by * bCols + bx]);
          }
        }
        else
        {
          bShared[rowPos + col] = 0.0; // We must set this to 0, otherwise we need conditionals below.
        }
      }
    }

    // Required to ensure shared memory is correctly populated before proceeding.
    workgroupBarrier();

    // Calculate all values in our 4x4 tile.
    for (var j : u32 = 0; j < kTileSizeY; j = j + 1) {
      for (var i : u32 = 0; i < kTileSizeX; i = i + 1) {
        var row = localInvocationId.y * kTileSizeY + j;
        var col = localInvocationId.x * kTileSizeX + i;
        var sum : f32 = 0.0;
        for (var k : u32 = 0; k < kSharedMemDimX; k = k + 1) {
            // Note: Even if we transpose the matrix, we still need to
            // maintain the same row/col multiplication order in the
            // sub-blocks to ensure we 'transpose' sub-blocks.
            sum = sum + aShared[row * kSharedMemDimX + k] * bShared[k * kSharedMemDimX + col];
        }

        // It's 'okay' to do a non-atomic update of C. We are guaranteed
        // to be the only thread touching it.
        suma[i][j] = suma[i][j] + sum;
      }
    }

    // This barrier is absolutely necessary to avoid other threads updating the shared
    // memory before we are done with it.
    workgroupBarrier();
  }

  for (var j : u32 = 0; j < kTileSizeY; j = j + 1) {
    for (var i : u32 = 0; i < kTileSizeX; i = i + 1) {
      if (do_scale) {
        suma[i][j] = suma[i][j] * scale;
      }

      if (!isOutOfBounds(M, N, outputIndexA.x + i, outputIndexA.y + j))
      {
        let outIndex : u32 = (outputIndexA.y + j) * N + outputIndexA.x + i;
        c[matStrideOut + outIndex] = suma[i][j];
      }
    }
  }
}
)";

static bool validate_mat_mul(
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C) {
    if (A.type != TensorType_F32) {
        fprintf(stderr, "cmdbuf_mat_mul: A.type != TensorType_F32\n");
        assert(false);
        return false;
    }

    if (!(B.type == TensorType_F32 || B.type == TensorType_F16)) {
        fprintf(stderr, "cmdbuf_mat_mul: B.type != TensorType_F32 or TensorType_F16\n");
        assert(false);
        return false;
    }

    if (C.type != TensorType_F32) {
        fprintf(stderr, "cmdbuf_mat_mul: C.type != TensorType_F32\n");
        assert(false);
        return false;
    }

    if (A.shape.c != B.shape.r) {
        fprintf(stderr, "cmdbuf_mat_mul: Num rows in A not equal to columns in B\n");
        assert(false);
        return false;
    }
    
    if (A.shape.r == 0 || A.shape.c == 0) {
        fprintf(stderr, "cmdbuf_mat_mul: one of the dimensions of A is zero.\n");
        assert(false);
        return false;
    }

    if (B.shape.r == 0 || B.shape.c == 0) {
        printf("cmdbuf_mat_mul: one of the dimensions of B is zero.\n");
        assert(false);
        return false;
    }

    if (A.shape.b > 1) {
        if (A.shape.b != B.shape.b) {
            printf("cmdbuf_mat_mul: A.shape.b != B.shape.b\n");
            assert(false);
            return false;
        }

        if (A.shape.b != C.shape.b) {
            printf("cmdbuf_mat_mul: A.shape.b != C.shape.b\n");
            assert(false);
            return false;
        }
    }

    if (C.shape.r != A.shape.r) {
        printf("cmdbuf_mat_mul: C's number of rows do not match A's\n");
        assert(false);
        return false;
    }

    if (C.shape.c != B.shape.c) {
        printf("cmdbuf_mat_mul: C's number of columns do not match B's\n");
        assert(false);
        return false;
    }

    return true;
}

// Note workgroup size (gWorkgroupX*gWorkgroupY) shouldn't exceed 256.
static const int64_t kMatMulWorkgroupX = 8;
static const int64_t kMatMulWorkgroupY = 8;

static const int64_t kMatMulTileSizeX = 1; // 1
static const int64_t kMatMulTileSizeY = 1; // 1

ComputePipeline pipeline_mat_mul(
        WGPUDevice device,
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        int transposeB,
        bool useUniforms) {
    if (!validate_mat_mul(A,B,C)) {
      return {};
    }

    const char* label = "mat_mul";
    
    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        // Binding 0: Input matrix
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        // Binding 1: Weight matrix.
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        // Binding 2: Output matrix
        .binding = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
    };
    
    if (useUniforms) {
        bglEntries.push_back(
            WGPUBindGroupLayoutEntry{
              // Binding 3: 
              .binding = 3,
              .visibility = WGPUShaderStage_Compute,
              .buffer = WGPUBufferBindingLayout{
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
              },
            }); 
    }
    
    int64_t M = A.shape.r;
    int64_t N = B.shape.c;
    int64_t K = A.shape.c;

    bool is_f16 = (B.type == TensorType_F16);
    std::string is_f16_str = "false";
    if (is_f16) { is_f16_str = "true"; }
    
    // Add embedding dims.
    std::string code = "\nconst kEmbeddingSize = " + std::to_string(K) + ";\n";
    code += "\nconst kNumTokens = " + std::to_string(M) + ";\n";
    if (!useUniforms) {
        code += "\nconst M = " + std::to_string(M) + ";\n";
        code += "\nconst N = " + std::to_string(N) + ";\n";
        code += "\nconst K = " + std::to_string(K) + ";\n";
    } else {
        code += R"ARR(
          struct Uniforms {
            A_B : u32,
            A_M : u32,
            A_N : u32,
            scale : f32,
            B_B : u32,
            B_M : u32,
            B_N : u32,
            offset : f32
          };
          @group(0) @binding(3) var<uniform> uniforms : Uniforms;
          )ARR";
    }
    code += "\nconst kWorkgroupX : u32 = " + std::to_string(kMatMulWorkgroupX) + ";\n";
    code += "\nconst kWorkgroupY : u32 = " + std::to_string(kMatMulWorkgroupY) + ";\n";
    code += "\nconst kTileSizeX : u32 = " + std::to_string(kMatMulTileSizeX) + ";\n";
    code += "\nconst kTileSizeY : u32 = " + std::to_string(kMatMulTileSizeY) + ";\n";
    code += "\nconst kTransposeB : u32 = " + std::to_string(transposeB) + ";\n";
    code += "\nconst is_b_f16 : bool = " + is_f16_str + ";\n";
    if (is_f16) {
      code += R"(
      @group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
      @group(0) @binding(1) var<storage,read> b : array<u32>; // [K, N] (f16)
      @group(0) @binding(2) var<storage,read_write> c : array<f32>;
      )";
    } else {
      code += R"(
      @group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
      @group(0) @binding(1) var<storage,read> b : array<f32>; // [K, N]
      @group(0) @binding(2) var<storage,read_write> c : array<f32>;
      )";
    }
    code += std::string(wgsl_fp16_to_fp32);
    code += std::string(wgsl_gemm_header);

    if (useUniforms) {
        // This code is the same as:
        // int64_t M = A.shape.r;
        // int64_t N = B.shape.c;
        // int64_t K = A.shape.c;
        code += R"ARR(
          let M : u32 = uniforms.A_M;
          let N : u32 = uniforms.B_N;
          let K : u32 = uniforms.A_N;
          let scale : f32 = uniforms.scale;
          let do_scale : bool = true;
          )ARR";
    } else {
        code += "\nlet scale : f32 = 1.0;\n";
        code += "\nlet do_scale : bool = false;\n";
    }
    code += std::string(wgsl_gemm_body);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &A, &B, &C);
    
    return pipeline;
}

CommandBuffer cmdbuf_mat_mul(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        WGPUComputePassEncoder pass,
        ComputePipeline* pipeline,
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        int transposeB,
        WGPUBuffer uniforms)
{
    if (!validate_mat_mul(A,B,C)) {
      return {};
    }

    bool useUniforms = (uniforms != nullptr);

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_mat_mul(device, A, B, C, transposeB, uniforms);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else if (!useUniforms) {
        if (!validate_pipeline(*pipeline, &A, &B, &C)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = A.gpu,
        .offset = 0,
        .size = (uint32_t)A.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = B.gpu,
        .offset = 0,
        .size = (uint32_t)B.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 2,
        .buffer = C.gpu,
        .offset = 0,
        .size = (uint32_t)C.get_size_bytes(),
      },
    };

    if (useUniforms) {
      bgEntries.push_back(
          WGPUBindGroupEntry{
            .binding = 3,
            .buffer = uniforms,
            .offset = 0,
            .size = kLlamaUniformsSize
          });
    }

    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t M = A.shape.r;
    int64_t N = B.shape.c;

    int64_t divisorX = (kMatMulWorkgroupX * kMatMulTileSizeX);
    int64_t divisorY = (kMatMulWorkgroupY * kMatMulTileSizeY);
    int64_t workgroupsX = N / divisorX;
    int64_t workgroupsY = M / divisorY;
    int64_t workgroupsZ = A.shape.b == 0 ? 1: A.shape.b;

    if (N % divisorX != 0) { workgroupsX = workgroupsX + 1; }
    if (M % divisorY != 0) { workgroupsY = workgroupsY + 1; }

    CommandBuffer out{};

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

int64_t kTransposeWorkgroupX = 8;
int64_t kTransposeWorkgroupY = 8;

// I only need two transpose functions:
// 1) A basic 2D matrix transpose.
// 2) A 3D matrix transpose where we transpose the batches and rows. This can be used in combination with 1) to transpose batches and columns.

// B - Number of batches.
// M - Number of rows.
// N - Number of columns.
// kWorkgroupX - Size of the workgroup along X dimension.
// kWorkgroupY - Size of the workgroup along Y dimension.
// let do_zy : bool = False;
static const char* wgsl_transpose_header_f32 = R"(
@group(0) @binding(0) var<storage,read> a : array<f32>;         // f32 [B, M, N]
@group(0) @binding(1) var<storage,read_write> c : array<f32>;   // f32 [B, M, N]

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupX, kWorkgroupY, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>)
{
)";

static const char* wgsl_transpose_body_f32 = R"(
  let zb : u32 = M*N;
  let yb : u32 = N;
  let index_z = workGroupID.z + localInvocationId.z;
  let index_y = (workGroupID.y*kWorkgroupY) + localInvocationId.y;
  let index_x = (workGroupID.x*kWorkgroupX) + localInvocationId.x;

  if (isOutOfBounds(M, N, index_x, index_y)) {
    return;
  }

  if (do_zy) {
    let zbt : u32 = B*N;
    let ybt : u32 = N;
    c[index_y*zbt + index_z*ybt + index_x] = a[index_z*zb + index_y*yb + index_x];
  } else {
    let zbt : u32 = M*N;
    let ybt : u32 = M;
    c[index_z*zbt + index_x*ybt + index_y] = a[index_z*zb + index_y*yb + index_x];
  }
}
)";

static bool validate_transpose(
        const TensorBuffer& A,
        const TensorBuffer& B,
        bool zy) {
    if (zy) {
        if (   A.shape.r != B.shape.b
            || A.shape.c != B.shape.c
            || A.shape.b != B.shape.r) {
            printf("cmdbuf_transpose: zy: input shape doesn't match expected transpose of output.\n");
            assert(false);
            return false;
        }
    } else {
        if (   A.shape.r != B.shape.c
            || A.shape.c != B.shape.r
            || A.shape.b != B.shape.b) {
            printf("cmdbuf_transpose: yx: input shape doesn't match expected transpose of output.\n");
            assert(false);
            return false;
        }
    }

    return true;
}

static ComputePipeline pipeline_transpose(
        WGPUDevice device,
        const TensorBuffer& A,
        const TensorBuffer& B,
        bool zy,
        bool useUniforms,
        int64_t M, int64_t N) {
    const char* label = "transpose";

    int64_t numBatches = A.shape.b;
    if (numBatches <= 0) { numBatches = 1; }

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
    };
    
    if (useUniforms) {
        bglEntries.push_back(
            WGPUBindGroupLayoutEntry{
              .binding = 2,
              .visibility = WGPUShaderStage_Compute,
              .buffer = WGPUBufferBindingLayout{
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
              },
            }); 
    }

    int64_t kWorkgroupX = kTransposeWorkgroupX;
    int64_t kWorkgroupY = kTransposeWorkgroupY;

    if (M == 1 && numBatches == 1) {
        kWorkgroupX = 256;
        kWorkgroupY = 1;
    }

    std::string code = "";
    if (zy) {
        code += "\nconst do_zy = true;\n";
    } else {
        code += "\nconst do_zy = false;\n";
    }
    if (!useUniforms) {
        code += "\nconst B : u32 = " + std::to_string(numBatches) + ";\n";
        code += "\nconst M : u32 = " + std::to_string(M) + ";\n";
        code += "\nconst N : u32 = " + std::to_string(N) + ";\n";
    } else {
        code += R"ARR(
          struct Uniforms {
            A_B : u32,
            A_M : u32,
            A_N : u32,
            pad1 : f32,
            B_B : u32,
            B_M : u32,
            B_N : u32,
            pad2 : f32
          };
          @group(0) @binding(2) var<uniform> uniforms : Uniforms;
          )ARR";
    }
    code += "\nconst kWorkgroupX = " + std::to_string(kWorkgroupX) + ";\n";
    code += "\nconst kWorkgroupY = " + std::to_string(kWorkgroupY) + ";\n";
    code += std::string(wgsl_transpose_header_f32);
    if (useUniforms) {
        code += R"ARR(
          let B : u32 = uniforms.A_B;
          let M : u32 = uniforms.A_M;
          let N : u32 = uniforms.A_N;
          )ARR";
    }
    code += std::string(wgsl_transpose_body_f32);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &A, &B);
      
    return pipeline;
}

// TODO: Handle f16 and other quantizations within this command buffer.
// zy parameter.
CommandBuffer cmdbuf_transpose(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        WGPUComputePassEncoder pass,
        ComputePipeline* pipeline,
        const TensorBuffer& A,
        const TensorBuffer& B,
        bool zy,
        WGPUBuffer dimBuffer)
{
    const int64_t M = A.shape.r;
    const int64_t N = A.shape.c;
    
    bool useUniforms = (dimBuffer != nullptr);

    if (!validate_transpose(A,B,zy)) {
        return {};
    }

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_transpose(device, A, B, zy, useUniforms, M, N);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else if (!useUniforms) {
        if (!validate_pipeline(*pipeline, &A, &B)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
        WGPUBindGroupEntry{
          .binding = 0,
          .buffer = A.gpu,
          .offset = 0,
          .size = A.get_size_bytes()
        },
        WGPUBindGroupEntry{
          .binding = 1,
          .buffer = B.gpu,
          .offset = 0,
          .size = B.get_size_bytes()
        },
    };
    
    if (useUniforms) {
      bgEntries.push_back(
          WGPUBindGroupEntry{
            .binding = 2,
            .buffer = dimBuffer,
            .offset = 0,
            .size = kLlamaUniformsSize
          });
    }

    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    // 32-bit to 16-bit conversion requires 2 32-bit input values for every 32-bit
    // output value.
    // CODE DUPLICATION WITH PIPELINE
    int64_t numBatches = A.shape.b;
    if (numBatches <= 0) { numBatches = 1; }
    int64_t kWorkgroupX = kTransposeWorkgroupX;
    int64_t kWorkgroupY = kTransposeWorkgroupY;
    if (M == 1 && numBatches == 1) {
      kWorkgroupX = 256;
      kWorkgroupY = 1;
    }
    // CODE DUPLICATION WITH PIPELINE
    
    int64_t workgroupsX = N / (kWorkgroupX);
    int64_t workgroupsY = M / (kWorkgroupY);
    int64_t workgroupsZ = numBatches;
    
    if (N % (kWorkgroupX) != 0) { workgroupsX++; }
    if (M % (kWorkgroupY) != 0) { workgroupsY++; }

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    CommandBuffer out{};

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

static const char* wgsl_rms_norm = R"(
const kWorkgroupSize : u32 = 256;   // Maximum workgroup size (multiple_of).
const kColsPerThread : u32 = N / kWorkgroupSize; // WARNING: Expects N to be a multiple of kWorkgroupSize.

const kEpsilon : f32 = 1e-6f;

@group(0) @binding(0) var<storage,read_write> inpL : array<f32>;

var<workgroup> aShared : array<f32, kWorkgroupSize>;

@compute @workgroup_size(kWorkgroupSize, 1u, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
    let baseIndex = workGroupID.y * N + localInvocationId.x * kColsPerThread;
    
    var sum : f32 = 0.0;
    for (var i : u32 = 0; i < kColsPerThread; i = i + 1) {
        sum = sum + f32(inpL[baseIndex + i]) * f32(inpL[baseIndex + i]);
    }

    aShared[localInvocationId.x] = sum;
    workgroupBarrier();
    
    sum = 0.0;
    for (var stride : u32 = kWorkgroupSize / 2; stride > 0; stride = stride / 2) {
        if (localInvocationId.x < stride) {
            aShared[localInvocationId.x] = aShared[localInvocationId.x] + aShared[localInvocationId.x + stride];
        }
        workgroupBarrier();
    }
    
    if (localInvocationId.x == 0) {
        aShared[0] = 1.0 / sqrt(aShared[0] / f32(N) + kEpsilon);
    }

    workgroupBarrier();

    // Scale all elements by the inverse of the RMS norm.
    let invNorm : f32 = aShared[0];

    for (var i : u32 = 0; i < kColsPerThread; i = i + 1) {
        inpL[baseIndex + i] = inpL[baseIndex + i] * invNorm;
    }
}
)";

static ComputePipeline pipeline_rms_norm(
        WGPUDevice device,
        const TensorBuffer& A) {
    const char* label = "rms_norm";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
        WGPUBindGroupLayoutEntry{
            // Binding 0: Input/output matrix
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Storage,
              .hasDynamicOffset = false,
            },
        },
    };

    std::string code = "\nconst N = " + std::to_string(A.shape.c) + ";\n";
    code += std::string(wgsl_rms_norm);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &A);

    return pipeline;
}

CommandBuffer cmdbuf_rms_norm(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        WGPUComputePassEncoder pass,
        ComputePipeline* pipeline,
        const TensorBuffer& A)
{
    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_rms_norm(device, A);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
        if (!validate_pipeline(*pipeline, &A)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
        WGPUBindGroupEntry{
            .binding = 0,
            .buffer = A.gpu,
            .offset = 0,
            .size = (uint32_t)A.get_size_bytes(),
        },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc); // Do we need to release what we are overwriting?
    

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
        encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
        pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, A.shape.r, 1);

    CommandBuffer out{};

    if (!hasPass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    if (!hasEncoder && !hasPass) {
        out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

static const char* wgsl_row_element_multiply = R"(
const kWorkgroupSize : u32 = 256;   // Maximum workgroup size.

@group(0) @binding(0) var<storage,read_write> inpL : array<f32>;
@group(0) @binding(1) var<storage,read> colMult : array<f32>;

@compute @workgroup_size(kWorkgroupSize, 1u, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
  // Multiplication works based on 'rows'. We replicate incoming colMult.
  let colIndex = workGroupID.x*kWorkgroupSize + localInvocationId.x;
  let baseIndex = workGroupID.y*N + colIndex;
  inpL[baseIndex] = inpL[baseIndex]*colMult[colIndex];
}
)";

static bool validate_row_element_multiply(
        const TensorBuffer& /*A*/,
        const TensorBuffer& B) {
    if (B.shape.r != 1) {
        printf("create_row_element_multiply: Expected B to have 1 row, got %" PRId64 "\n", B.shape.r);
        assert(false);
        return false;
    }
    
    return true;
}

static ComputePipeline pipeline_row_element_multiply(
        WGPUDevice device,
        const TensorBuffer& A,
        const TensorBuffer& B) {
    if (!validate_row_element_multiply(A,B)) {
      return {};
    }

    const char* label = "row_element_multiply";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
    };

    std::string code = "\nconst N = " + std::to_string(A.shape.c) + ";\n";
    code += std::string(wgsl_row_element_multiply);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &A, &B);
      
    return pipeline;
}

CommandBuffer cmdbuf_row_element_multiply(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        WGPUComputePassEncoder pass,
        ComputePipeline* pipeline,
        const TensorBuffer& A,
        const TensorBuffer& B)
{
    if (!validate_row_element_multiply(A,B)) {
      return {};
    }

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_row_element_multiply(device, A, B);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
        if (!validate_pipeline(*pipeline, &A, &B)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = A.gpu,
        .offset = 0,
        .size = (uint32_t)A.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = B.gpu,
        .offset = 0,
        .size = (uint32_t)B.get_size_bytes(),
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);


    const int kWorkgroupSize = 256; // In the shader.
    const int kColsPerWorkgroup = A.shape.c / kWorkgroupSize;

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, kColsPerWorkgroup, A.shape.r, 1);

    CommandBuffer out{};

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}


static const int64_t kRoPEXTileSize = 2;
static const int64_t kRoPEWorkgroupSizeX = 8;
static const int64_t kRoPEWorkgroupSizeY = 8;

// Rotary positional embedding.
static const char* wgsl_gpu_rope = R"(
// Multiple constants are passed into the shader.

struct NetworkUniforms {
  n_past : u32,
  n_tokens : u32,
  pad1 : f32,
  pad2 : f32,
};

@group(0) @binding(0) var<storage,read_write> input : array<f32>;
@group(0) @binding(1) var<uniform> uniforms : NetworkUniforms;

@compute @workgroup_size(kWorkgroupSizeX, kWorkgroupSizeY, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
  let x : u32 = workGroupID.x*kWorkgroupSizeX*kTileSizeX + localInvocationId.x*kTileSizeX;
  let y : u32 = workGroupID.y*kWorkgroupSizeY + localInvocationId.y;
  let index : u32 = workGroupID.z*kZB + y*kYB + x*kXB;

  let p : f32 = f32(uniforms.n_past + workGroupID.z);
  let theta : f32 = pow(10000.0, (-f32(x))/f32(kDims));

  let cos_theta : f32 = cos(p*theta);
  let sin_theta : f32 = sin(p*theta);

  let x0 = input[index + 0];
  let x1 = input[index + 1];

  input[index + 0] = x0*cos_theta - x1*sin_theta;
  input[index + 1] = x0*sin_theta + x1*cos_theta;
}
)";

ComputePipeline pipeline_RoPE(
        WGPUDevice device,
        const TensorBuffer& A) {
    const char* label = "RoPE";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
        },
      },
    };

    std::string code = "";

    code += "\nconst kTileSizeX : u32 = " + std::to_string(kRoPEXTileSize) + ";\n";
    code += "\nconst kXB : u32 = 1;\n";
    code += "\nconst kYB : u32 = " + std::to_string(A.shape.c) + ";\n";
    code += "\nconst kZB : u32 = " + std::to_string(A.shape.c*A.shape.r) + ";\n";
    code += "\nconst kDims : u32 = " + std::to_string(A.shape.c) + ";\n";
    code += "\nconst kWorkgroupSizeX : u32 = " + std::to_string(kRoPEWorkgroupSizeX) + ";\n";
    code += "\nconst kWorkgroupSizeY : u32 = " + std::to_string(kRoPEWorkgroupSizeY) + ";\n";
    code += std::string(wgsl_gpu_rope);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &A);

    return pipeline;
}

CommandBuffer cmdbuf_RoPE(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        WGPUComputePassEncoder pass,
        ComputePipeline* pipeline,
        const TensorBuffer& A,
        WGPUBuffer networkUniforms) // Buffer should include n_past as the first u32 element
{
    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_RoPE(device, A);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
        if (!validate_pipeline(*pipeline, &A)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = A.gpu,
        .offset = 0,
        .size = (uint32_t)A.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = networkUniforms,
        .offset = 0,
        .size = (uint32_t)kLlamaUniformsSize,
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t kWorkgroupsX = A.shape.c / (kRoPEWorkgroupSizeX * kRoPEXTileSize);
    int64_t kWorkgroupsY = A.shape.r / (kRoPEWorkgroupSizeY);
    int64_t kWorkgroupsZ = A.shape.b;
    if (kWorkgroupsZ == 0) {
      kWorkgroupsZ = 1;
    }

    CommandBuffer out{};

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, kWorkgroupsX, kWorkgroupsY, kWorkgroupsZ);

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}


static const char* wgsl_masked_softmax_header = R"(
const kEpsilon : f32 = 1e-6f;
const kNegativeInf : f32 = -1e14f;

@group(0) @binding(0) var<storage,read_write> a : array<f32>;

var<workgroup> aShared : array<f32, kWorkgroupSizeX*kWorkgroupSizeY>;

@compute @workgroup_size(kWorkgroupSizeX, kWorkgroupSizeY, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
)";

static const char* wgsl_masked_softmax_body = R"(
  let matStride : u32 = M * N * workGroupID.z;

  let kColsPerThread : u32 = N / kWorkgroupSizeX; // WARNING: Expects N to be a multiple of kWorkgroupSizeX

  let sharedIndexY = localInvocationId.y*kWorkgroupSizeX;
  let sharedIndex = sharedIndexY + localInvocationId.x;
  let index = vec2<u32>(
         workGroupID.xy * vec2<u32>(kWorkgroupSizeX*kColsPerThread, kWorkgroupSizeY)
       + localInvocationId.xy * vec2<u32>(kColsPerThread, 1));

  let firstIndex : u32 = matStride + index.y*N + index.x;
  var row_max : f32 = a[firstIndex];
  if (index.x + localInvocationId.x*kColsPerThread > index.y + localInvocationId.y) {
    row_max = kNegativeInf;    
    a[firstIndex] = 0.0;
  }
  for (var i : u32 = /*START_AT_ONE*/1; i < kColsPerThread; i = i + 1) {
    if (index.x + localInvocationId.x*kColsPerThread <= index.y + localInvocationId.y) {
      row_max = max(row_max, a[firstIndex + i]);
    } else {
      a[firstIndex + i] = 0.0;
    }
  }

  aShared[sharedIndex] = row_max;

  workgroupBarrier();
  
  for (var stride : u32 = kWorkgroupSizeX / 2; stride > 0; stride = stride / 2) {
    if (localInvocationId.x < stride) {
      aShared[sharedIndex] = max(aShared[sharedIndex], aShared[sharedIndex + stride]);
    }
    workgroupBarrier();
  }

  row_max = aShared[sharedIndexY];


  var row_exp_sum : f32 = 0.0;
  for (var i : u32 = 0; i < kColsPerThread; i = i + 1) {
    if (index.x + localInvocationId.x*kColsPerThread <= index.y + localInvocationId.y) {
      let val_exp = exp(a[firstIndex + i] - row_max);
      a[firstIndex + i] = val_exp;
      row_exp_sum = row_exp_sum + val_exp;
    }
  }

  aShared[sharedIndex] = row_exp_sum;
  workgroupBarrier();

  for (var stride : u32 = kWorkgroupSizeX / 2; stride > 0; stride = stride / 2) {
    if (localInvocationId.x < stride) {
      aShared[sharedIndex] = aShared[sharedIndex] + aShared[sharedIndex + stride];
    }
    workgroupBarrier();
  }
  
  row_exp_sum = aShared[sharedIndexY];

  // Remember, we've already set the appropriate matrix elements to 0.
  for (var i : u32 = 0; i < kColsPerThread; i = i + 1) {
    a[firstIndex + i] = a[firstIndex + i] / row_exp_sum;
  }
}
)";

static const int64_t kMaskedSoftmaxWorkgroupX = 8;
static const int64_t kMaskedSoftmaxWorkgroupY = 8;

ComputePipeline pipeline_masked_softmax(
        WGPUDevice device,
        const TensorBuffer& inputOutputBuffer,
        bool useUniforms,
        int64_t /*B*/, int64_t M, int64_t N) {
    const char* label = "masked_softmax";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        // Binding 0: Input/output matrix
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
    };

    if (useUniforms) {
        bglEntries.push_back(
            WGPUBindGroupLayoutEntry{
              .binding = 1,
              .visibility = WGPUShaderStage_Compute,
              .buffer = WGPUBufferBindingLayout{
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
              },
            }); 
    }

    std::string code = "";
    if (!useUniforms) {
        code += "\nconst M : u32 = " + std::to_string(M) + ";\n";
        code += "\nconst N : u32 = " + std::to_string(N) + ";\n";
    } else {
        code += R"ARR(
          struct Uniforms {
            A_B : u32,
            A_M : u32,
            A_N : u32,
            pad1 : f32,
            B_B : u32,
            B_M : u32,
            B_N : u32,
            pad2 : f32
          };
          @group(0) @binding(1) var<uniform> uniforms : Uniforms;
          )ARR";
    }
    code += "\nconst kWorkgroupSizeX = " + std::to_string(kMaskedSoftmaxWorkgroupX) + ";\n";
    code += "\nconst kWorkgroupSizeY = " + std::to_string(kMaskedSoftmaxWorkgroupY) + ";\n";
    code += std::string(wgsl_masked_softmax_header);
    if (useUniforms) {
        code += R"ARR(
          let B : u32 = uniforms.A_B;
          let M : u32 = uniforms.A_M;
          let N : u32 = uniforms.A_N;
          )ARR";
    }
    code += std::string(wgsl_masked_softmax_body);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &inputOutputBuffer);
    
    return pipeline;
}

CommandBuffer cmdbuf_masked_softmax(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        ComputePipeline* pipeline,
        const TensorBuffer& inputOutputBuffer,
        WGPUBuffer dimBuffer)
{
    int64_t B = inputOutputBuffer.shape.b;
    const int64_t M = inputOutputBuffer.shape.r;
    const int64_t N = inputOutputBuffer.shape.c;
    if (B == 0) { B = 1; }

    bool useUniforms = (dimBuffer != nullptr);

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_masked_softmax(device, inputOutputBuffer, useUniforms, B, M, N);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else if (!useUniforms) {
        if (!validate_pipeline(*pipeline, &inputOutputBuffer)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = inputOutputBuffer.gpu,
        .offset = 0,
        .size = (uint32_t)inputOutputBuffer.get_size_bytes(),
      },
    };

    if (useUniforms) {
      bgEntries.push_back(
          WGPUBindGroupEntry{
            .binding = 1,
            .buffer = dimBuffer,
            .offset = 0,
            .size = kLlamaUniformsSize
          });
    }

    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc); // Do we need to release what we are overwriting?

    int64_t workgroupsX = N / (kMaskedSoftmaxWorkgroupX);
    int64_t workgroupsY = M / (kMaskedSoftmaxWorkgroupY);
    int64_t workgroupsZ = B;

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    CommandBuffer out{};

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

static const char* wgsl_row_softmax_header = R"(
const kEpsilon : f32 = 1e-6f;
const kNegativeInf : f32 = -1e14f;

@group(0) @binding(0) var<storage,read_write> a : array<f32>;

var<workgroup> aShared : array<f32, kWorkgroupSize>;

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupSize, 1u, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
)";

static const char* wgsl_row_softmax_body = R"(
  let matStride : u32 = M * N * workGroupID.z;

  // Potential source of errors for large contexts.
  var kColsPerThread : u32 = N / kWorkgroupSize;
  if (N % kWorkgroupSize != 0) {
    kColsPerThread = kColsPerThread + 1;
  }

  let indexX = localInvocationId.x * kColsPerThread;
  let indexY = workGroupID.y;
  let offsetY = indexY * N;

  let baseIndex = matStride + offsetY + indexX;
  let sharedIndex = localInvocationId.x;

  var row_max : f32 = kNegativeInf;
  for (var i : u32 = 0; i < kColsPerThread; i = i + 1) {
    if (!isOutOfBounds(M,N, indexX + i, indexY))
    {
      row_max = max(row_max, a[baseIndex + i]);
    }
  }
  
  aShared[sharedIndex] = row_max;
  workgroupBarrier();
  
  for (var stride : u32 = kWorkgroupSize / 2; stride > 0; stride = stride / 2) {
    if (localInvocationId.x < stride) {
      aShared[sharedIndex] = max(aShared[sharedIndex], aShared[sharedIndex + stride]);
    }
    workgroupBarrier();
  }

  row_max = aShared[0];

      //for (var i : u32 = 0; i < kColsPerThread; i = i + 1) {
      //  if (!isOutOfBounds(M,N, indexX + i, indexY)) {
      //    a[baseIndex + i] = row_max;
      //  }
      //}
      //return;

  var row_exp_sum : f32 = 0.0;
  for (var i : u32 = 0; i < kColsPerThread; i = i + 1) {
    if (!isOutOfBounds(M,N, indexX + i, indexY))
    {
      let val_exp = exp(a[baseIndex + i] - row_max);
      a[baseIndex + i] = val_exp;
      row_exp_sum = row_exp_sum + val_exp;
    }
  }

  aShared[sharedIndex] = row_exp_sum;
  workgroupBarrier();

  // Binary parallel reduction requires a workgroup that is a power of 2.
  for (var stride : u32 = kWorkgroupSize / 2; stride > 0; stride = stride / 2) {
    if (localInvocationId.x < stride) {
      aShared[sharedIndex] = aShared[sharedIndex] + aShared[sharedIndex + stride];
    }
    workgroupBarrier();
  }
  
  row_exp_sum = aShared[0];

  // Remember, we've already set the appropriate matrix elements to 0.
  for (var i : u32 = 0; i < kColsPerThread; i = i + 1) {
    if (!isOutOfBounds(M,N, indexX + i, indexY))
    {
      a[baseIndex + i] = a[baseIndex + i] / row_exp_sum;
    }
  }

}

)";

ComputePipeline pipeline_row_softmax(
        WGPUDevice device,
        const TensorBuffer& inputOutputBuffer,
        bool useUniforms,
        int64_t /*B*/, int64_t M, int64_t N) {
    const char* label = "row_softmax";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
    };

    if (useUniforms) {
        bglEntries.push_back(
            WGPUBindGroupLayoutEntry{
              .binding = 1,
              .visibility = WGPUShaderStage_Compute,
              .buffer = WGPUBufferBindingLayout{
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
              },
            }); 
    }
    
    // WorkgroupSize must be a power of 2 due to parallel reduction.
    int64_t kWorkgroupSize = 256;

    std::string code = "";
    if (!useUniforms) {
        code += "\nconst M : u32 = " + std::to_string(M) + ";\n";
        code += "\nconst N : u32 = " + std::to_string(N) + ";\n";
    } else {
        code += R"ARR(
          struct Uniforms {
            A_B : u32,
            A_M : u32,
            A_N : u32,
            pad1 : f32,
            B_B : u32,
            B_M : u32,
            B_N : u32,
            pad2 : f32
          };
          @group(0) @binding(1) var<uniform> uniforms : Uniforms;
          )ARR";
    }
    code += "\nconst kWorkgroupSize : u32 = " + std::to_string(kWorkgroupSize) + ";\n";
    
    code += std::string(wgsl_row_softmax_header);
    if (useUniforms) {
        code += R"ARR(
          let B : u32 = uniforms.A_B;
          let M : u32 = uniforms.A_M;
          let N : u32 = uniforms.A_N;
          )ARR";
    }
    code += std::string(wgsl_row_softmax_body);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);
    
    store_pipeline_validation(pipeline, &inputOutputBuffer);

    return pipeline;
}

CommandBuffer cmdbuf_row_softmax(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        WGPUComputePassEncoder pass,
        ComputePipeline* pipeline,
        const TensorBuffer& inputOutputBuffer,
        WGPUBuffer dimBuffer)
{
    int64_t B = inputOutputBuffer.shape.b;
    const int64_t M = inputOutputBuffer.shape.r;
    const int64_t N = inputOutputBuffer.shape.c;
    if (B == 0) { B = 1; }

    bool useUniforms = (dimBuffer != nullptr);

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_row_softmax(device, inputOutputBuffer, useUniforms, B, M, N);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else if (!useUniforms) {
        if (!validate_pipeline(*pipeline, &inputOutputBuffer)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = inputOutputBuffer.gpu,
        .offset = 0,
        .size = (uint32_t)inputOutputBuffer.get_size_bytes(),
      },
    };

    if (useUniforms) {
      bgEntries.push_back(
          WGPUBindGroupEntry{
            .binding = 1,
            .buffer = dimBuffer,
            .offset = 0,
            .size = kLlamaUniformsSize
          });
    }

    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc); // Do we need to release what we are overwriting?

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
        encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
        pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, inputOutputBuffer.shape.r, B);
    
    CommandBuffer out{};

    if (!hasPass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if(!hasEncoder && !hasPass) {
        out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

static const char* wgsl_addition = R"(
@group(0) @binding(0) var<storage,read> a : array<f32>;
@group(0) @binding(1) var<storage,read> b : array<f32>;
@group(0) @binding(2) var<storage,read_write> c : array<f32>;

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupX, kWorkgroupY, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
  let matStride : u32 = M * N * workGroupID.z;

  // Each thread loads 4x4 tile into the shared workgroup memory.
  let outputIndexA = vec2<u32>(
    workGroupID.xy * vec2<u32>(kWorkgroupX, kWorkgroupY) +
    localInvocationId.xy);

  if (!isOutOfBounds(M, N, outputIndexA.x, outputIndexA.y)) {
    let outIndex : u32 = (outputIndexA.y) * N + outputIndexA.x;
    let index : u32 = matStride + outIndex;
    c[index] = a[index] + b[index];
  }
}
)";

static const int64_t kAdditionWorkgroupX = 8;
static const int64_t kAdditionWorkgroupY = 8;

static bool validate_addition(
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C) {
    if (A.shape != B.shape) {
        printf("create_addition: a.shape not equal to b.shape\n");
        assert(false);
        return {};
    }

    if (A.shape != C.shape) {
        printf("create_addition: a.shape not equal to c.shape\n");
        assert(false);
        return {};
    }
    
    return true;
}

static ComputePipeline pipeline_addition(
        WGPUDevice device,
        const TensorBuffer& a,
        const TensorBuffer& b,
        const TensorBuffer& c,
        int64_t B, int64_t M, int64_t N) {
    if (!validate_addition(a,b,c)) {
      return {};
    }

    const char* label = "addition";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        .binding = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
    };

    // We removed this optimization as we are caching pipelines.
    // Look into adding it back as our pipelines are cached
    // based on the token size (this optimization is for 1 token).
    //if (M == 1 && B == 1) {
    //  kWorkgroupX = 256;
    //  kWorkgroupY = 1;
    //}

    std::string code = "";
    code += "\nconst B : u32 = " + std::to_string(B) + ";\n";
    code += "\nconst M : u32 = " + std::to_string(M) + ";\n";
    code += "\nconst N : u32 = " + std::to_string(N) + ";\n";
    code += "\nconst kWorkgroupX = " + std::to_string(kAdditionWorkgroupX) + ";\n";
    code += "\nconst kWorkgroupY = " + std::to_string(kAdditionWorkgroupY) + ";\n";
    code += std::string(wgsl_addition);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &a, &b, &c);

    return pipeline;
}

CommandBuffer cmdbuf_addition(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        ComputePipeline* pipeline,
        const TensorBuffer& a,
        const TensorBuffer& b,
        const TensorBuffer& c) {
    if (!validate_addition(a,b,c)) {
      return {};
    }

    int64_t B = a.shape.b;
    const int64_t M = a.shape.r;
    const int64_t N = a.shape.c;
    if (B == 0) { B = 1; }

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_addition(device, a, b, c, B, M, N);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
        if (!validate_pipeline(*pipeline, &a, &b, &c)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = a.gpu,
        .offset = 0,
        .size = (uint32_t)a.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = b.gpu,
        .offset = 0,
        .size = (uint32_t)b.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 2,
        .buffer = c.gpu,
        .offset = 0,
        .size = (uint32_t)c.get_size_bytes(),
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t workgroupsX = N / (kAdditionWorkgroupX);
    int64_t workgroupsY = M / (kAdditionWorkgroupY);
    int64_t workgroupsZ = B;
    
    if (N % (kAdditionWorkgroupX) != 0) {
      workgroupsX += 1;
    }
    
    if (M % (kAdditionWorkgroupY) != 0) {
      workgroupsY += 1;
    }

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    CommandBuffer out{};

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

static const char* wgsl_element_wise_mult = R"(
@group(0) @binding(0) var<storage,read> a : array<f32>;
@group(0) @binding(1) var<storage,read> b : array<f32>;
@group(0) @binding(2) var<storage,read_write> c : array<f32>;

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupX, kWorkgroupY, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
  let matStride : u32 = M * N * workGroupID.z;

  // Each thread loads 4x4 tile into the shared workgroup memory.
  let outputIndexA = vec2<u32>(
    workGroupID.xy * vec2<u32>(kWorkgroupX, kWorkgroupY) +
    localInvocationId.xy);

  if (!isOutOfBounds(M, N, outputIndexA.x, outputIndexA.y)) {
    let outIndex : u32 = (outputIndexA.y) * N + outputIndexA.x;
    let index : u32 = matStride + outIndex;
    c[index] = a[index] * b[index];
  }
}
)";

CommandBuffer create_element_mult(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        const TensorBuffer& a,
        const TensorBuffer& b,
        const TensorBuffer& c) {
    const char* label = "hadamard";

    CommandBuffer out{};

    if (a.shape != b.shape) {
        printf("create_addition: a.shape not equal to b.shape\n");
        return {};
    }

    if (a.shape != c.shape) {
        printf("create_addition: a.shape not equal to c.shape\n");
        return {};
    }

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        .binding = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
    };

    int64_t B = a.shape.b;
    const int64_t M = a.shape.r;
    const int64_t N = a.shape.c;

    if (B == 0) { B = 1; }

    int64_t kWorkgroupX = 8;
    int64_t kWorkgroupY = 8;
    
    if (M == 1 && B == 1) {
      kWorkgroupX = 256;
      kWorkgroupY = 1;
    }

    std::string code = "";
    code += "\nconst B : u32 = " + std::to_string(B) + ";\n";
    code += "\nconst M : u32 = " + std::to_string(M) + ";\n";
    code += "\nconst N : u32 = " + std::to_string(N) + ";\n";
    code += "\nconst kWorkgroupX = " + std::to_string(kWorkgroupX) + ";\n";
    code += "\nconst kWorkgroupY = " + std::to_string(kWorkgroupY) + ";\n";
    code += std::string(wgsl_element_wise_mult);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = a.gpu,
        .offset = 0,
        .size = (uint32_t)a.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = b.gpu,
        .offset = 0,
        .size = (uint32_t)b.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 2,
        .buffer = c.gpu,
        .offset = 0,
        .size = (uint32_t)c.get_size_bytes(),
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline.bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    pipeline.bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t workgroupsX = N / (kWorkgroupX);
    int64_t workgroupsY = M / (kWorkgroupY);
    int64_t workgroupsZ = B;
    
    if (N % (kWorkgroupX) != 0) { workgroupsX++; }
    if (M % (kWorkgroupY) != 0) { workgroupsY++; }

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, pipeline.bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }

    return out;
}

static const char* wgsl_element_wise_mult_in_place = R"(
@group(0) @binding(0) var<storage,read_write> a : array<f32>;
@group(0) @binding(1) var<storage,read> b : array<f32>;

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupX, kWorkgroupY, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
  let matStride : u32 = M * N * workGroupID.z;

  // Each thread loads 4x4 tile into the shared workgroup memory.
  let outputIndexA = vec2<u32>(
    workGroupID.xy * vec2<u32>(kWorkgroupX, kWorkgroupY) +
    localInvocationId.xy);

  if (!isOutOfBounds(M, N, outputIndexA.x, outputIndexA.y))
  {
    let outIndex : u32 = (outputIndexA.y) * N + outputIndexA.x;
    let index : u32 = matStride + outIndex;
    a[index] = a[index] * b[index];
  }
}
)";

static const int64_t kHadamardInPlaceWorkgroupX = 8;
static const int64_t kHadamardInPlaceWorkgroupY = 8;
    

static bool validate_element_mult_in_place(
        const TensorBuffer& a,
        const TensorBuffer& b) {
    if (a.shape != b.shape) {
        printf("create_element_mult_in_place: a.shape not equal to b.shape\n");
        assert(false);
        return false;
    }

    return true;
}

ComputePipeline pipeline_element_mult_in_place(
        WGPUDevice device,
        const TensorBuffer& a,
        const TensorBuffer& b,
        int64_t B, int64_t M, int64_t N) {
    if (!validate_element_mult_in_place(a, b)) {
        return {};
    }

    const char* label = "hadamard_in_place";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
      WGPUBindGroupLayoutEntry{
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_ReadOnlyStorage,
          .hasDynamicOffset = false,
        },
      },
    };

    std::string code = "";
    code += "\nconst B : u32 = " + std::to_string(B) + ";\n";
    code += "\nconst M : u32 = " + std::to_string(M) + ";\n";
    code += "\nconst N : u32 = " + std::to_string(N) + ";\n";
    code += "\nconst kWorkgroupX = " + std::to_string(kHadamardInPlaceWorkgroupX) + ";\n";
    code += "\nconst kWorkgroupY = " + std::to_string(kHadamardInPlaceWorkgroupY) + ";\n";
    code += std::string(wgsl_element_wise_mult_in_place);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &a, &b);
      
    return pipeline;
}

CommandBuffer cmdbuf_element_mult_in_place(
        WGPUDevice device,
        WGPUCommandEncoder encoder,
        WGPUComputePassEncoder pass,
        ComputePipeline* pipeline,
        const TensorBuffer& a,
        const TensorBuffer& b) {
    if (!validate_element_mult_in_place(a, b)) {
        return {};
    }

    int64_t B = a.shape.b;
    const int64_t M = a.shape.r;
    const int64_t N = a.shape.c;
    if (B == 0) { B = 1; }

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_element_mult_in_place(device, a, b, B, M, N);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
        if (!validate_pipeline(*pipeline, &a, &b)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = a.gpu,
        .offset = 0,
        .size = (uint32_t)a.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = b.gpu,
        .offset = 0,
        .size = (uint32_t)b.get_size_bytes(),
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t workgroupsX = N / (kHadamardInPlaceWorkgroupX);
    int64_t workgroupsY = M / (kHadamardInPlaceWorkgroupY);
    int64_t workgroupsZ = B;
    
    if (N % (kHadamardInPlaceWorkgroupX) != 0) { workgroupsX++; }
    if (M % (kHadamardInPlaceWorkgroupY) != 0) { workgroupsY++; }

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    CommandBuffer out{};

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}


static const char* wgsl_silu = R"(
@group(0) @binding(0) var<storage,read_write> a : array<f32>;

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupX, kWorkgroupY, 1u)
fn main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationId : vec3<u32>
)
{
  let matStride : u32 = M * N * workGroupID.z;

  // Each thread loads 4x4 tile into the shared workgroup memory.
  let outputIndexA = vec2<u32>(
    workGroupID.xy * vec2<u32>(kWorkgroupX, kWorkgroupY) +
    localInvocationId.xy);
    
  if (isOutOfBounds(M, N, outputIndexA.x, outputIndexA.y)) {
    return;
  }

  let outIndex : u32 = (outputIndexA.y) * N + outputIndexA.x;
  let index : u32 = matStride + outIndex;
  let val = a[index];
  a[index] = val / (1.0 + exp(-val));
}
)";

static const int64_t kSiluWorkgroupX = 8;
static const int64_t kSiluWorkgroupY = 8;

ComputePipeline pipeline_silu(
        WGPUDevice device,
        const TensorBuffer& a,
        int64_t B, int64_t M, int64_t N) {
    const char* label = "silu";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
      WGPUBindGroupLayoutEntry{
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = WGPUBufferBindingLayout{
          .type = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
        },
      },
    };

    // Optimization disable due to pipeline caching. Revisit.
    int64_t workgroupX = kSiluWorkgroupX;
    int64_t workgroupY = kSiluWorkgroupY;
    if (B == 1 && M == 1) {
        workgroupX = 256;
        workgroupY = 1;
    }

    std::string code = "";
    code += "\nconst B : u32 = " + std::to_string(B) + ";\n";
    code += "\nconst M : u32 = " + std::to_string(M) + ";\n";
    code += "\nconst N : u32 = " + std::to_string(N) + ";\n";
    code += "\nconst kWorkgroupX = " + std::to_string(workgroupX) + ";\n";
    code += "\nconst kWorkgroupY = " + std::to_string(workgroupY) + ";\n";
    code += std::string(wgsl_silu);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);

    store_pipeline_validation(pipeline, &a);
    
    return pipeline;
}

CommandBuffer cmdbuf_silu(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        ComputePipeline* pipeline,
        const TensorBuffer& a) {
    int64_t B = a.shape.b;
    const int64_t M = a.shape.r;
    const int64_t N = a.shape.c;
    if (B == 0) { B = 1; }

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_silu(device, a, B, M, N);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
        if (!validate_pipeline(*pipeline, &a)) {
            printf("create_rms_norm: Pipeline validation failed.\n");
            assert(false);
            return {};
        }
    }

    // BUG: I think there is a bug where we can't use 8x8 workgroups for B=1, M=1.
    int64_t workgroupX = kSiluWorkgroupX;
    int64_t workgroupY = kSiluWorkgroupY;
    // NOTE: Disabling this optimization causes invalid results.
    if (B == 1 && M == 1) {
        workgroupX = 256;
        workgroupY = 1;
    }

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = a.gpu,
        .offset = 0,
        .size = (uint32_t)a.get_size_bytes(),
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
                            };
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t workgroupsX = N / (workgroupX);
    int64_t workgroupsY = M / (workgroupY);
    int64_t workgroupsZ = B;

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
      encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
      pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    CommandBuffer out{};

    if (!hasPass) {
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
      out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

static const char* wgsl_vector_mat_mul_transpose = R"(
// 'a' is an input batch x 'row' vector.
// 'b' is an input matrix.
// 'c' is on output batch x 'row' vector.

var<workgroup> mshare : array<f32,kWorkgroupSize>;

@compute @workgroup_size(kWorkgroupSize, 1u, 1u)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
)
{
  var bs : u32 = 1;
  if (is_b_f16) { bs = 2; }

  let matStrideA : u32 = 1 * C * wid.z;
  let matStrideB : u32 = R * C * wid.z;
  let matStrideO : u32 = 1 * R * wid.z;

  //               batch           col
  let aIndex = matStrideA + lid.x*kTileSize;
  let bIndex = matStrideB + lid.x*kTileSize  + wid.x*C;
  let cIndex = matStrideO + wid.x;

  let si = lid.x;

  var sum : f32 = 0.0;
  for (var i : u32 = 0; i < kTileSize; i = i + 1) {
    if (is_b_f16) {
      let f16_part = (bIndex + i) % 2;
      let b = compute_fp16_to_fp32(u32(b[(bIndex + i)/bs]), f16_part);
      sum = sum + a[aIndex + i] * b;
    } else {
      sum = sum + a[aIndex + i] * f32(b[bIndex + i]);
    }
  }

  mshare[si] = sum;

  workgroupBarrier();
  
  for (var stride : u32 = kWorkgroupSize / 2; stride > 0; stride = stride / 2) {
    if (lid.x < stride) {
      mshare[si] = mshare[si] + mshare[si+stride];
    }
    workgroupBarrier();
  }

  if (lid.x == 0) {
    c[cIndex] = mshare[si];
  }
}
)";

static bool validate_vector_mat_mul_trans(
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C) {
    if (A.type != TensorType_F32) {
        fprintf(stderr, "create_vector_mat_mul_trans: A.type != TensorType_F32\n");
        assert(false);
        return false;
    }

    if (!(B.type == TensorType_F32 || B.type == TensorType_F16)) {
        fprintf(stderr, "create_vector_mat_mul_trans: B.type != TensorType_F32 or TensorType_F16\n");
        assert(false);
        return false;
    }

    if (C.type != TensorType_F32) {
        fprintf(stderr, "create_vector_mat_mul_trans: C.type != TensorType_F32\n");
        assert(false);
        return false;
    }

    if (A.shape.c != B.shape.c) {
        fprintf(stderr, "create_vector_mat_mul_trans: Expecting number of columns in A to match B\n");
        assert(false);
        return false;
    }
    
    if (B.shape.r == 0 || B.shape.c == 0) {
        printf("create_vector_mat_mul_trans: one of the dimensions of B is zero.\n");
        assert(false);
        return false;
    }

    if (A.shape.b > 1) {
        if (A.shape.b != B.shape.b) {
            printf("create_vector_mat_mul_trans: A.shape.b != B.shape.b\n");
            assert(false);
            return false;
        }

        if (A.shape.b != C.shape.b) {
            printf("create_vector_mat_mul_trans: A.shape.b != C.shape.b\n");
            assert(false);
            return false;
        }
    }

    if (C.shape.c != B.shape.r) { // Inverted from what we would expect as we are performing a transpose.
        printf("C's number of columns do not match B's\n");
        assert(false);
        return false;
    }

    return true;
}

static ComputePipeline pipeline_vector_mat_mul_trans(
        WGPUDevice device,
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C) {
    // Computing pipeline for vector x matrix.
    if (!validate_vector_mat_mul_trans(A, B, C)) {
        return {};
    }

    const char* label = "vector_mat_mul";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
        WGPUBindGroupLayoutEntry{
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_ReadOnlyStorage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_ReadOnlyStorage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Storage,
              .hasDynamicOffset = false,
            },
        },
    };
    
    int64_t rows = B.shape.r;
    int64_t cols = B.shape.c;

    // Note workgroup size (gWorkgroupX*gWorkgroupY) shouldn't exceed 256.
    const int64_t kWorkgroupSize = 256; // 16

    if (A.shape.c < kWorkgroupSize) {
        printf("A columns is less than kWorkgroupSize\n");
        assert(false);
        return {};
    }

    if (A.shape.c % kWorkgroupSize != 0) {
        printf("A is not a multiple of kWorkgroupSize\n");
        assert(false);
        return {};
    }

    const int64_t kTileSize = cols / kWorkgroupSize; // 1

    bool is_f16 = (B.type == TensorType_F16);
    std::string is_f16_str = "false";
    if (is_f16) { is_f16_str = "true"; }
    
    // TODO:  Use uniform buffers for the ability to change sizes of input matrices.
    //        We shouldn't recreate pipelines.
    // Add embedding dims.
    std::string code = "";
    code += "\nconst R = " + std::to_string(rows) + ";\n";
    code += "\nconst C = " + std::to_string(cols) + ";\n";
    code += "\nconst kWorkgroupSize : u32 = " + std::to_string(kWorkgroupSize) + ";\n";
    code += "\nconst kTileSize : u32 = " + std::to_string(kTileSize) + ";\n";
    code += "\nconst is_b_f16 : bool = " + is_f16_str + ";\n";
    if (is_f16) {
        code += R"(
        @group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
        @group(0) @binding(1) var<storage,read> b : array<u32>; // [K, N] (f16)
        @group(0) @binding(2) var<storage,read_write> c : array<f32>;
        )";
    } else {
        code += R"(
        @group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
        @group(0) @binding(1) var<storage,read> b : array<f32>; // [K, N]
        @group(0) @binding(2) var<storage,read_write> c : array<f32>;
        )";
    }
    code += std::string(wgsl_fp16_to_fp32);
    code += std::string(wgsl_vector_mat_mul_transpose);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);
    
    store_pipeline_validation(pipeline, &A, &B, &C);

    return pipeline;
}

CommandBuffer cmdbuf_vector_mat_mul_trans(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        ComputePipeline* pipeline,
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        int64_t aOffset)
{
    if (!validate_vector_mat_mul_trans(A, B, C)) {
        return {};
    }

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_vector_mat_mul_trans(device, A, B, C);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
      if (!validate_pipeline(*pipeline, &A, &B, &C)) {
          printf("Pipeline shapes not equal to input shapes.\n");
          assert(false);
          return {};
      }
    }

    int64_t rows = B.shape.r;
    int64_t cols = B.shape.c;

    CommandBuffer out{};

    std::vector<WGPUBindGroupEntry> bgEntries = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = A.gpu,
        .offset = uint64_t(aOffset),
        .size = uint64_t(cols * get_TensorType_size(A.type)), // Only want one row of A.
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = B.gpu,
        .offset = 0,
        .size = (uint32_t)B.get_size_bytes(),
      },
      WGPUBindGroupEntry{
        .binding = 2,
        .buffer = C.gpu,
        .offset = 0,
        .size = (uint32_t)C.get_size_bytes(),
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntries.size(),
                              .entries    = bgEntries.data(),
              };

    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t workgroupsX = rows;
    int64_t workgroupsY = 1;
    int64_t workgroupsZ = A.shape.b == 0 ? 1: A.shape.b;

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
        encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
        pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    if (!hasPass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
        out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}


static const char* wgsl_vector_mat_mul_split_transpose_header = R"(
struct Uniforms {
    A_B : u32,
    A_M : u32,
    A_N : u32,
    split : u32, // Split 0, 1, 2, 3
    B_B : u32,
    B_M : u32,
    B_N : u32,
    totalSplits : u32 // Total number of splits
};
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

var<workgroup> mshare : array<f32,kWorkgroupSize>;

@compute @workgroup_size(kWorkgroupSize, 1u, 1u)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
)
{
)";
static const char* wgsl_vector_mat_mul_split_transpose_body = R"(
  var bs : u32 = 1;
  if (is_b_f16) { bs = 2; }

  let matStrideA : u32 = 1 * C * wid.z;
  let matStrideB : u32 = R * C * wid.z;
  let matStrideO : u32 = 1 * R * wid.z;
  
  let bSplitSize : u32 = C / uniforms.totalSplits;
  let bColOff : u32 = uniforms.split * bSplitSize;

  //               batch           col
  let aIndex = matStrideA + lid.x*kTileSize  + bColOff;
  let bIndex = matStrideB + lid.x*kTileSize  + wid.x*C + bColOff;
  let cIndex = matStrideO + wid.x;

  let si = lid.x;

  var sum : f32 = 0.0;
  for (var i : u32 = 0; i < kTileSize; i = i + 1) {
    if (is_b_f16) {
      let f16_part = (bIndex + i) % 2;
      let b = compute_fp16_to_fp32(u32(b[(bIndex + i)/bs]), f16_part);
      sum = sum + a[aIndex + i] * b;
    } else {
      sum = sum + a[aIndex + i] * f32(b[bIndex + i]);
    }
  }

  mshare[si] = sum;

  workgroupBarrier();
  
  for (var stride : u32 = kWorkgroupSize / 2; stride > 0; stride = stride / 2) {
    if (lid.x < stride) {
      mshare[si] = mshare[si] + mshare[si+stride];
    }
    workgroupBarrier();
  }

  if (lid.x == 0) {
    //c[cIndex] = f32(uniforms.split);//mshare[si];
    c[cIndex] = mshare[si];
  }
}
)";

static bool validate_vector_mat_mul_split_trans(
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        const std::vector<TensorBuffer*>& /*scratchBuffers*/) {
    if (A.type != TensorType_F32) {
        fprintf(stderr, "create_vector_mat_mul_trans: A.type != TensorType_F32\n");
        assert(false);
        return false;
    }

    if (!(B.type == TensorType_F32 || B.type == TensorType_F16)) {
        fprintf(stderr, "create_vector_mat_mul_trans: B.type != TensorType_F32 or TensorType_F16\n");
        assert(false);
        return false;
    }

    if (C.type != TensorType_F32) {
        fprintf(stderr, "create_vector_mat_mul_trans: C.type != TensorType_F32\n");
        assert(false);
        return false;
    }

    if (A.shape.c != B.shape.c) {
        fprintf(stderr, "create_vector_mat_mul_trans: Expecting number of columns in A to match B\n");
        assert(false);
        return false;
    }
    
    if (B.shape.r == 0 || B.shape.c == 0) {
        printf("create_vector_mat_mul_trans: one of the dimensions of B is zero.\n");
        assert(false);
        return false;
    }

    if (A.shape.b > 1) {
        if (A.shape.b != B.shape.b) {
            printf("create_vector_mat_mul_trans: A.shape.b != B.shape.b\n");
            assert(false);
            return false;
        }

        if (A.shape.b != C.shape.b) {
            printf("create_vector_mat_mul_trans: A.shape.b != C.shape.b\n");
            assert(false);
            return false;
        }
    }

    if (C.shape.c != B.shape.r) { // Inverted from what we would expect as we are performing a transpose.
        printf("C's number of columns do not match B's\n");
        assert(false);
        return false;
    }
    
    //for (TensorBuffer* buffer : scratchBuffers) {
    //    if (C.shape != buffer->shape) {
    //        printf("C.shape != CScratch.shape\n");
    //        assert(false);
    //        return false;
    //    }
    //}

    return true;
}


static ComputePipeline pipeline_vector_mat_mul_split_trans(
        WGPUDevice device,
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        const std::vector<TensorBuffer*>& scratchBuffers,
        int numSplits,
        bool useDimsFromUniforms) {
    // Computing pipeline for vector x matrix.
    if (!validate_vector_mat_mul_split_trans(A, B, C, scratchBuffers)) {
        return {};
    }

    const char* label = "vector_mat_mul_split";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
        WGPUBindGroupLayoutEntry{
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_ReadOnlyStorage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_ReadOnlyStorage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Storage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 3,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Uniform,
              .hasDynamicOffset = false,
            },
        },
    };
    
    int64_t rows = B.shape.r;
    int64_t cols = B.shape.c;

    // Note workgroup size (gWorkgroupX*gWorkgroupY) shouldn't exceed 256.
    const int64_t kWorkgroupSize = 256;

    if ((cols / numSplits) < kWorkgroupSize) {
        printf("A columns is less than kWorkgroupSize cols:%" PRId64 " numSplits:%d kWorkgroupSize:%" PRId64 "\n", cols, numSplits, kWorkgroupSize);

        assert(false);
        return {};
    }

    if (((cols / numSplits) % kWorkgroupSize) != 0) {
        printf("A is not a multiple of kWorkgroupSize\n");
        assert(false);
        return {};
    }

    const int64_t kTileSize = (cols / numSplits) / kWorkgroupSize; // 1

    bool is_f16 = (B.type == TensorType_F16);
    std::string is_f16_str = "false";
    if (is_f16) { is_f16_str = "true"; }

    
    // TODO:  Use uniform buffers for the ability to change sizes of input matrices.
    //        We shouldn't recreate pipelines.
    // Add embedding dims.
    std::string code = "";
    code += "\nconst kWorkgroupSize : u32 = " + std::to_string(kWorkgroupSize) + ";\n";
    code += "\nconst kTileSize : u32 = " + std::to_string(kTileSize) + ";\n";
    code += "\nconst is_b_f16 : bool = " + is_f16_str + ";\n";
    if (is_f16) {
        code += R"(
        @group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
        @group(0) @binding(1) var<storage,read> b : array<u32>; // [K, N] (f16)
        @group(0) @binding(2) var<storage,read_write> c : array<f32>;
        )";
    } else {
        code += R"(
        @group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
        @group(0) @binding(1) var<storage,read> b : array<f32>; // [K, N]
        @group(0) @binding(2) var<storage,read_write> c : array<f32>;
        )";
    }
    code += std::string(wgsl_fp16_to_fp32);
    code += std::string(wgsl_vector_mat_mul_split_transpose_header);
    if (useDimsFromUniforms) {
        code += R"(
          let R = uniforms.B_M;
          let C = uniforms.B_N;
          )";
    } else {
        code += "\nlet R : u32 = " + std::to_string(rows) + ";\n";
        code += "\nlet C : u32 = " + std::to_string(cols) + ";\n";
    }
    code += std::string(wgsl_vector_mat_mul_split_transpose_body);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);
    
    store_pipeline_validation(pipeline, &A, &B, &C);

    return pipeline;
}

// TODO Need another pass that combines the split results. We have a separate
//      function for this to avoid having to switch pipelines during heavy
//      parallel computation.
// TODO Prep a split buffer. WebGPU doesn't support buffers larger than 256 MB.
//      There is one buffer that is. So we should accomodate.
CommandBuffer cmdbuf_vector_mat_mul_split_trans(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        ComputePipeline* pipeline,
        const TensorBuffer& A,
        const TensorBuffer& B,
        const TensorBuffer& C,
        const std::vector<TensorBuffer*>& scratchBuffers,
        // int32_t numSplits, // By default the number of splits is two.
        int64_t aOffset,
        const std::vector<WGPUBuffer>& splitBuffers,
        bool useDimsFromUniforms)
{
    if (!validate_vector_mat_mul_split_trans(A, B, C, scratchBuffers)) {
        return {};
    }
    
    assert(scratchBuffers.size() >= splitBuffers.size() - 1);

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_vector_mat_mul_split_trans(device, A, B, C, scratchBuffers, splitBuffers.size(), useDimsFromUniforms);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
      if (!validate_pipeline(*pipeline, &A, &B, &C)) {
          printf("Pipeline shapes not equal to input shapes.\n");
          assert(false);
          return {};
      }
    }


    int64_t rows = B.shape.r;
    int64_t cols = B.shape.c;

    CommandBuffer out{};

    std::vector<std::vector<WGPUBindGroupEntry>> bgEntriesSplit;
    for (int i = 0; i < (int)splitBuffers.size(); ++i) {
        bgEntriesSplit.push_back({
          WGPUBindGroupEntry{
            .binding = 0,
            .buffer = A.gpu,
            .offset = uint64_t(aOffset),
            .size = uint64_t(cols * get_TensorType_size(A.type)), // Only want one row of A.
          },
          WGPUBindGroupEntry{
            .binding = 1,
            .buffer = B.gpu,
            .offset = 0,
            .size = (uint32_t)B.get_size_bytes(),
          },
          WGPUBindGroupEntry{
            .binding = 2,
            .buffer = (i == 0) ? C.gpu : scratchBuffers[i-1]->gpu,
            .offset = 0,
            .size = (uint32_t)C.get_size_bytes(),
          },
          WGPUBindGroupEntry{
            .binding = 3,
            .buffer = splitBuffers[i],
            .offset = 0,
            .size = kLlamaUniformsSize,
          },
        });
    }

    int64_t workgroupsX = rows;
    int64_t workgroupsY = 1;
    int64_t workgroupsZ = A.shape.b == 0 ? 1: A.shape.b;

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
        encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
        pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);

    for (int i = 0; i < (int)splitBuffers.size(); ++i) {
        WGPUBindGroupDescriptor bgDescSplit1 = {
                                  .layout     = pipeline->bindGroupLayout,
                                  .entryCount = (uint32_t)bgEntriesSplit[i].size(),
                                  .entries    = bgEntriesSplit[i].data(),
                  };

        WGPUBindGroup bindGroup1 = wgpuDeviceCreateBindGroup(device, &bgDescSplit1);

        // TODO Should be based on numSplits
        wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup1, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);
          
        wgpuBindGroupRelease(bindGroup1);
    }

    if (!hasPass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
        out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
    }

    return out;
}


static const char* wgsl_vector_multi_mat_mul_split_transpose_header = R"(
struct Uniforms {
    A_B : u32,
    A_M : u32,
    A_N : u32,
    split : u32, // Split 0, 1, 2, 3
    B_B : u32,
    B_M : u32,
    B_N : u32,
    totalSplits : u32 // Total number of splits
};
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

var<workgroup> mshare : array<f32,kWorkgroupSize>;

@compute @workgroup_size(kWorkgroupSize, 1u, 1u)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
)
{
)";
static const char* wgsl_vector_multi_mat_mul_split_transpose_body = R"(
  var bs : u32 = 1;
  if (is_b_f16) { bs = 2; }

  let C_splt = C / uniforms.totalSplits;
  let C_orig = C;

  let matStrideA : u32 = 1 * C_orig * wid.z;
  let matStrideB : u32 = R * C_splt * wid.z;
  let matStrideO : u32 = 1 * R * wid.z;
  
  let bSplitSize : u32 = C_orig / uniforms.totalSplits;
  let bColOff : u32 = uniforms.split * bSplitSize;

  //               batch           col
  let aIndex = matStrideA + lid.x*kTileSize  + bColOff;
  let bIndex = matStrideB + lid.x*kTileSize  + wid.x*C_splt;
  let cIndex = matStrideO + wid.x;

  let si = lid.x;

  var sum : f32 = 0.0;
  for (var i : u32 = 0; i < kTileSize; i = i + 1) {
    if (is_b_f16) {
      let f16_part = (bIndex + i) % 2;
      let b = compute_fp16_to_fp32(u32(b[(bIndex + i)/bs]), f16_part);
      sum = sum + a[aIndex + i] * b;
    } else {
      sum = sum + a[aIndex + i] * f32(b[bIndex + i]);
    }
  }

  mshare[si] = sum;

  workgroupBarrier();
  
  for (var stride : u32 = kWorkgroupSize / 2; stride > 0; stride = stride / 2) {
    if (lid.x < stride) {
      mshare[si] = mshare[si] + mshare[si+stride];
    }
    workgroupBarrier();
  }

  if (lid.x == 0) {
    //c[cIndex] = f32(uniforms.split);//mshare[si];
    c[cIndex] = mshare[si];
  }
}
)";

static bool validate_vector_multi_mat_mul_split_trans(
        const TensorBuffer& A,
        const std::vector<TensorBuffer*>& B,
        const TensorBuffer& C,
        const std::vector<TensorBuffer*>& /*scratchBuffers*/,
        int numSplits) {
    if ((int)B.size() != numSplits) {
        fprintf(stderr, "create_vector_multi_mat_mul_trans: Expected B array to equal numSplits\n");
        assert(false);
        return false;
    }

    for (TensorBuffer* buffer1 : B) {
        if (buffer1 == nullptr || buffer1->gpu == nullptr) {
            fprintf(stderr, "create_vector_multi_mat_mul_trans: Encountered an empty gpu buffer\n");
            assert(false);
            return false;
        }
    }

    if (A.type != TensorType_F32) {
        fprintf(stderr, "create_vector_multi_mat_mul_trans: A.type != TensorType_F32\n");
        assert(false);
        return false;
    }

    if (!(B[0]->type == TensorType_F32 || B[0]->type == TensorType_F16)) {
        fprintf(stderr, "create_vector_multi_mat_mul_trans: B.type != TensorType_F32 or TensorType_F16\n");
        assert(false);
        return false;
    }

    if (C.type != TensorType_F32) {
        fprintf(stderr, "create_vector_multi_mat_mul_trans: C.type != TensorType_F32\n");
        assert(false);
        return false;
    }

    if (A.shape.c != B[0]->shape.c * numSplits) {
        fprintf(stderr, "create_vector_multi_mat_mul_trans: Expecting number of columns in A to match B\n");
        assert(false);
        return false;
    }
    
    if (B[0]->shape.r == 0 || B[0]->shape.c == 0) {
        printf("create_vector_multi_mat_mul_trans: one of the dimensions of B is zero.\n");
        assert(false);
        return false;
    }

    if (A.shape.b > 1) {
        if (A.shape.b != B[0]->shape.b) {
            printf("create_vector_multi_mat_mul_trans: A.shape.b != B.shape.b\n");
            assert(false);
            return false;
        }

        if (A.shape.b != C.shape.b) {
            printf("create_vector_multi_mat_mul_trans: A.shape.b != C.shape.b\n");
            assert(false);
            return false;
        }
    }

    if (C.shape.c != B[0]->shape.r) { // Inverted from what we would expect as we are performing a transpose.
        printf("C's number of columns do not match B's\n");
        assert(false);
        return false;
    }
    
    //for (TensorBuffer* buffer : scratchBuffers) {
    //    if (C.shape != buffer->shape) {
    //        printf("C.shape != CScratch.shape\n");
    //        assert(false);
    //        return false;
    //    }
    //}

    return true;
}


static ComputePipeline pipeline_vector_multi_mat_mul_split_trans(
        WGPUDevice device,
        const TensorBuffer& A,
        const std::vector<TensorBuffer*> B,
        const TensorBuffer& C,
        const std::vector<TensorBuffer*>& scratchBuffers,
        int numSplits,
        bool useDimsFromUniforms) {
    // Computing pipeline for vector x matrix.
    if (!validate_vector_multi_mat_mul_split_trans(A, B, C, scratchBuffers, numSplits)) {
        return {};
    }

    const char* label = "vector_mat_mul_split";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
        WGPUBindGroupLayoutEntry{
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_ReadOnlyStorage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_ReadOnlyStorage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Storage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 3,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Uniform,
              .hasDynamicOffset = false,
            },
        },
    };
    
    int64_t rows = B[0]->shape.r;
    int64_t cols = B[0]->shape.c * numSplits;

    assert(cols == A.shape.c);

    // Note workgroup size (gWorkgroupX*gWorkgroupY) shouldn't exceed 256.
    const int64_t kWorkgroupSize = 256;

    if ((cols / numSplits) < kWorkgroupSize) {
        printf("A columns is less than kWorkgroupSize cols:%" PRId64 " numSplits:%d kWorkgroupSize:%" PRId64 "\n", cols, numSplits, kWorkgroupSize);

        assert(false);
        return {};
    }

    if (((cols / numSplits) % kWorkgroupSize) != 0) {
        printf("A is not a multiple of kWorkgroupSize\n");
        assert(false);
        return {};
    }

    const int64_t kTileSize = (cols / numSplits) / kWorkgroupSize; // 1

    bool is_f16 = (B[0]->type == TensorType_F16);
    std::string is_f16_str = "false";
    if (is_f16) { is_f16_str = "true"; }


    printf("STATS: %" PRId64 " %" PRId64 " %d %" PRId64 " f16:%d\n", kTileSize, cols, numSplits, kWorkgroupSize, is_f16);
    
    // TODO:  Use uniform buffers for the ability to change sizes of input matrices.
    //        We shouldn't recreate pipelines.
    // Add embedding dims.
    std::string code = "";
    code += "\nconst kWorkgroupSize : u32 = " + std::to_string(kWorkgroupSize) + ";\n";
    code += "\nconst kTileSize : u32 = " + std::to_string(kTileSize) + ";\n";
    code += "\nconst is_b_f16 : bool = " + is_f16_str + ";\n";
    if (is_f16) {
        code += R"(
        @group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
        @group(0) @binding(1) var<storage,read> b : array<u32>; // [K, N] (f16)
        @group(0) @binding(2) var<storage,read_write> c : array<f32>;
        )";
    } else {
        code += R"(
        @group(0) @binding(0) var<storage,read> a : array<f32>; // [M, K]
        @group(0) @binding(1) var<storage,read> b : array<f32>; // [K, N]
        @group(0) @binding(2) var<storage,read_write> c : array<f32>;
        )";
    }
    code += std::string(wgsl_fp16_to_fp32);
    code += std::string(wgsl_vector_multi_mat_mul_split_transpose_header);
    if (useDimsFromUniforms) {
        code += R"(
          let R = uniforms.B_M;
          let C = uniforms.B_N;
          )";
    } else {
        code += "\nlet R : u32 = " + std::to_string(rows) + ";\n";
        code += "\nlet C : u32 = " + std::to_string(cols) + ";\n";
    }
    code += std::string(wgsl_vector_multi_mat_mul_split_transpose_body);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);
    
    store_pipeline_validation(pipeline, &A, B[0], &C);

    return pipeline;
}

// TODO Need another pass that combines the split results. We have a separate
//      function for this to avoid having to switch pipelines during heavy
//      parallel computation.
// TODO Prep a split buffer. WebGPU doesn't support buffers larger than 256 MB.
//      There is one buffer that is. So we should accomodate.
CommandBuffer cmdbuf_vector_multi_mat_mul_split_trans(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        ComputePipeline* pipeline,
        const TensorBuffer& A,
        const std::vector<TensorBuffer*> B,
        const TensorBuffer& C,
        const std::vector<TensorBuffer*>& scratchBuffers,
        // int32_t numSplits, // By default the number of splits is two.
        int64_t aOffset,
        const std::vector<WGPUBuffer>& splitBuffers,
        bool useDimsFromUniforms)
{
    int32_t numSplits = splitBuffers.size();
    if (!validate_vector_multi_mat_mul_split_trans(A, B, C, scratchBuffers, numSplits)) {
        return {};
    }
    
    assert(scratchBuffers.size() >= splitBuffers.size() - 1);

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_vector_multi_mat_mul_split_trans(device, A, B, C, scratchBuffers, numSplits, useDimsFromUniforms);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
      if (!validate_pipeline(*pipeline, &A, B[0], &C)) {
          printf("Pipeline shapes not equal to input shapes.\n");
          assert(false);
          return {};
      }
    }


    int64_t rows = B[0]->shape.r;
    int64_t cols = B[0]->shape.c * numSplits;

    assert(cols == A.shape.c);

    CommandBuffer out{};

    std::vector<std::vector<WGPUBindGroupEntry>> bgEntriesSplit;
    for (int i = 0; i < (int)splitBuffers.size(); ++i) {
        bgEntriesSplit.push_back({
          WGPUBindGroupEntry{
            .binding = 0,
            .buffer = A.gpu,
            .offset = uint64_t(aOffset),
            .size = uint64_t(cols * get_TensorType_size(A.type)), // Only want one row of A.
          },
          WGPUBindGroupEntry{
            .binding = 1,
            .buffer = B[i]->gpu,
            .offset = 0,
            .size = (uint32_t)B[i]->get_size_bytes(),
          },
          WGPUBindGroupEntry{
            .binding = 2,
            .buffer = (i == 0) ? C.gpu : scratchBuffers[i-1]->gpu,
            .offset = 0,
            .size = (uint32_t)C.get_size_bytes(),
          },
          WGPUBindGroupEntry{
            .binding = 3,
            .buffer = splitBuffers[i],
            .offset = 0,
            .size = kLlamaUniformsSize,
          },
        });
    }

    int64_t workgroupsX = rows;
    int64_t workgroupsY = 1;
    int64_t workgroupsZ = A.shape.b == 0 ? 1: A.shape.b;

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
        encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
        pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);

    for (int i = 0; i < (int)splitBuffers.size(); ++i) {
        WGPUBindGroupDescriptor bgDescSplit1 = {
                                  .layout     = pipeline->bindGroupLayout,
                                  .entryCount = (uint32_t)bgEntriesSplit[i].size(),
                                  .entries    = bgEntriesSplit[i].data(),
                  };

        WGPUBindGroup bindGroup1 = wgpuDeviceCreateBindGroup(device, &bgDescSplit1);

        // TODO Should be based on numSplits
        wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup1, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);
          
        wgpuBindGroupRelease(bindGroup1);
    }

    if (!hasPass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
        out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
    }

    return out;
}

static const char* wgsl_vector_reduce = R"(

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupSize, 1u, 1u)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
)
{
  let matStrideA : u32 = 1 * C * wid.z;
  let matStrideB : u32 = 1 * C * wid.z;

  //let bSplitSize : u32 = C / numSplits;
  let bColOff : u32 = wid.x * splitSize;

  let xIndex = lid.x*kTileSize + bColOff;
  
  //               batch           col
  let aIndex = matStrideA + xIndex;
  let bIndex = matStrideB + xIndex;

  var sum : f32 = 0.0;
  for (var i : u32 = 0; i < kTileSize; i = i + 1) {
    if (!isOutOfBounds(1,C,xIndex + i,0)) {
        a[aIndex + i] = a[aIndex + i] + b[bIndex + i];
    }
  }
}
)";

static bool validate_vector_reduce(
        const TensorBuffer& A,
        const TensorBuffer& B) {
    if (A.shape != B.shape) {
        printf("C.shape != CScratch.shape\n");
        assert(false);
        return false;
    }
    
    return true;
}

static ComputePipeline pipeline_vector_reduce(
        WGPUDevice device,
        const TensorBuffer& A,
        const TensorBuffer& B,
        int numSplits) {
    // Computing pipeline for vector x matrix.
    if (!validate_vector_reduce(A, B)) {
        return {};
    }

    const char* label = "vector_reduce";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
        WGPUBindGroupLayoutEntry{
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Storage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_ReadOnlyStorage,
              .hasDynamicOffset = false,
            },
        },
    };
    
    int64_t cols = B.shape.c;

    const int64_t kWorkgroupSize = 256;
    int64_t kTileSize = (cols / numSplits) / kWorkgroupSize; // 1
    if (kTileSize == 0) {
        kTileSize = 1;
    }
    //printf("kTileSize: %d\n", kTileSize);
    //printf("numSplits: %d\n", numSplits);
    //printf("cols: %d\n", cols);
    //printf("Split size: %d\n", cols/numSplits);

    // Note workgroup size (gWorkgroupX*gWorkgroupY) shouldn't exceed 256.

    bool is_f16 = (B.type == TensorType_F16);
    std::string is_f16_str = "false";
    if (is_f16) { is_f16_str = "true"; }
    
    assert(!is_f16); // Not tested.
    
    // TODO:  Use uniform buffers for the ability to change sizes of input matrices.
    //        We shouldn't recreate pipelines.
    // Add embedding dims.
    std::string code = "";
    code += "\nconst numSplits : u32 = " + std::to_string(numSplits) + ";\n";
    code += "\nconst splitSize : u32 = " + std::to_string(cols/numSplits) + ";\n";
    code += "\nconst C : u32 = " + std::to_string(cols) + ";\n";
    code += "\nconst kWorkgroupSize : u32 = " + std::to_string(kWorkgroupSize) + ";\n";
    code += "\nconst kTileSize : u32 = " + std::to_string(kTileSize) + ";\n";
    code += "\nconst is_b_f16 : bool = " + is_f16_str + ";\n";
    if (is_f16) {
        code += R"(
        @group(0) @binding(0) var<storage,read_write> a : array<f32>; // [M, K]
        @group(0) @binding(1) var<storage,read> b : array<u32>; // [K, N] (f16)
        )";
    } else {
        code += R"(
        @group(0) @binding(0) var<storage,read_write> a : array<f32>; // [M, K]
        @group(0) @binding(1) var<storage,read> b : array<f32>; // [K, N]
        )";
    }
    code += std::string(wgsl_fp16_to_fp32);
    code += std::string(wgsl_vector_reduce);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);
    
    store_pipeline_validation(pipeline, &A, &B);

    return pipeline;
}


CommandBuffer cmdbuf_vector_reduce(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        ComputePipeline* pipeline,
        const TensorBuffer& A, // OUT
        const TensorBuffer& B,
        int numSplits) {
    if (!validate_vector_reduce(A, B)) {
        return {};
    }

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_vector_reduce(device, A, B, numSplits);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
      if (!validate_pipeline(*pipeline, &A, &B)) {
          printf("Pipeline shapes not equal to input shapes.\n");
          assert(false);
          return {};
      }
    }

    //int64_t rows = B.shape.r;
    //int64_t cols = B.shape.c;

    CommandBuffer out{};

    std::vector<WGPUBindGroupEntry> bgEntriesSplit = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = A.gpu,
        .offset = 0,
        .size = (uint32_t)A.get_size_bytes(), // Only want one row of A.
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = B.gpu,
        .offset = 0,
        .size = (uint32_t)B.get_size_bytes(),
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntriesSplit.size(),
                              .entries    = bgEntriesSplit.data(),
              };

    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t workgroupsX = numSplits;
    int64_t workgroupsY = 1;
    int64_t workgroupsZ = A.shape.b == 0 ? 1: A.shape.b;

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
        encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
        pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    if (!hasPass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
        out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

static const char* wgsl_f16_f32_conversion = R"(
@group(0) @binding(0) var<storage,read_write> a : array<f32>; // [M, K]
@group(0) @binding(1) var<storage,read> b : array<u32>; // [K, N] (f16)

fn isOutOfBounds(rows : u32, cols : u32, x : u32, y : u32) -> bool {
  return (x >= cols || y >= rows);
}

@compute @workgroup_size(kWorkgroupSize, 1u, 1u)
fn main(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
)
{
  let bs : u32 = 2;

  let matStrideA : u32 = 1 * C * wid.z;
  let matStrideB : u32 = 1 * C * wid.z;

  let bColOff : u32 = wid.x * splitSize;

  let xIndex = lid.x*kTileSize + bColOff;
  
  //               batch           col
  let aIndex = matStrideA + xIndex;
  let bIndex = matStrideB + xIndex;

  var sum : f32 = 0.0;
  for (var i : u32 = 0; i < kTileSize; i = i + 1) {
    if (!isOutOfBounds(1,C,xIndex + i,0)) {
        let f16_part = (bIndex + i) % 2;
        let b = compute_fp16_to_fp32(u32(b[(bIndex + i)/bs]), f16_part);
        a[aIndex + i] = b;
    }
  }
}
)";

static bool validate_f16_f32_conversion(
        const TensorBuffer& A,
        const TensorBuffer& B) {
    if (A.shape.c != B.shape.c) {
        printf("A.shape.c != B.shape.c\n");
        assert(false);
        return false;
    }

    if (A.type == B.type) {
        printf("Expected A and B to be different types\n");
        assert(false);
        return false;
    }

    if (!( A.type == TensorType_F32 || A.type == TensorType_F16)) {
        printf("A must be either F16 or F32\n");
        assert(false);
        return false;
    }

    if (!( B.type == TensorType_F32 || B.type == TensorType_F16)) {
        printf("B must be either F16 or F32\n");
        assert(false);
        return false;
    }

    return true;
}

static ComputePipeline pipeline_f16_f32_conversion(
        WGPUDevice device,
        const TensorBuffer& A,
        const TensorBuffer& B,
        int numSplits) {
    // Computing pipeline for vector x matrix.
    if (!validate_f16_f32_conversion(A, B)) {
        return {};
    }

    const char* label = "f16_f32_conversion";

    std::vector<WGPUBindGroupLayoutEntry> bglEntries = {
        WGPUBindGroupLayoutEntry{
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Storage,
              .hasDynamicOffset = false,
            },
        },
        WGPUBindGroupLayoutEntry{
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_ReadOnlyStorage,
              .hasDynamicOffset = false,
            },
        },
    };
    
    int64_t cols = B.shape.c;

    const int64_t kWorkgroupSize = 256;
    int64_t kTileSize = (cols / numSplits) / kWorkgroupSize; // 1
    if (kTileSize == 0) {
        kTileSize = 1;
    }
    //printf("kTileSize: %d\n", kTileSize);
    //printf("numSplits: %d\n", numSplits);
    //printf("cols: %d\n", cols);
    //printf("Split size: %d\n", cols/numSplits);

    // Note workgroup size (gWorkgroupX*gWorkgroupY) shouldn't exceed 256.
    
    assert(B.type == TensorType_F16); // Not tested.
    
    // TODO:  Use uniform buffers for the ability to change sizes of input matrices.
    //        We shouldn't recreate pipelines.
    // Add embedding dims.
    std::string code = "";
    code += "\nconst numSplits : u32 = " + std::to_string(numSplits) + ";\n";
    code += "\nconst splitSize : u32 = " + std::to_string(cols/numSplits) + ";\n";
    code += "\nconst C : u32 = " + std::to_string(cols) + ";\n";
    code += "\nconst kWorkgroupSize : u32 = " + std::to_string(kWorkgroupSize) + ";\n";
    code += "\nconst kTileSize : u32 = " + std::to_string(kTileSize) + ";\n";
    code += std::string(wgsl_fp16_to_fp32);
    code += std::string(wgsl_f16_f32_conversion);
    ComputePipeline pipeline = 
        create_compute_pipeline(device, code.c_str(), bglEntries, label);
    
    store_pipeline_validation(pipeline, &A, &B);

    return pipeline;
}


CommandBuffer cmdbuf_f16_f32_conversion(
        WGPUDevice device,
        WGPUCommandEncoder encoder,     // Optional (use nullptr)
        WGPUComputePassEncoder pass,    // Optional (use nullptr - takes precedence over encoder)
        ComputePipeline* pipeline,
        const TensorBuffer& A, // OUT
        const TensorBuffer& B,
        int numSplits,
        int aOffset,
        int bOffset) {
    if (!validate_f16_f32_conversion(A, B)) {
        return {};
    }

    // The following allows construction of pipelines and avoids high-level code duplication.
    ComputePipeline pipelineTemp{}; // Memory of this local variable is referenced throughout the function.
    if (!pipeline || !pipeline->is_valid()) {
        pipelineTemp = pipeline_f16_f32_conversion(device, A, B, numSplits);
        if (pipeline && pipeline->buildPipelineFlag) {
            *pipeline = std::move(pipelineTemp);
            return {};
        }
        pipeline = &pipelineTemp;
    } else {
      if (!validate_pipeline(*pipeline, &A, &B)) {
          printf("Pipeline shapes not equal to input shapes.\n");
          assert(false);
          return {};
      }
    }

    //int64_t rows = B.shape.r;
    //int64_t cols = B.shape.c;

    CommandBuffer out{};

    std::vector<WGPUBindGroupEntry> bgEntriesSplit = {
      WGPUBindGroupEntry{
        .binding = 0,
        .buffer = A.gpu,
        .offset = uint64_t(aOffset),
        .size = uint64_t(A.shape.c * get_TensorType_size(A.type)),
      },
      WGPUBindGroupEntry{
        .binding = 1,
        .buffer = B.gpu,
        .offset = uint64_t(bOffset),
        .size = uint64_t(B.shape.c * get_TensorType_size(B.type)),
      },
    };
    WGPUBindGroupDescriptor bgDesc = {
                              .layout     = pipeline->bindGroupLayout,
                              .entryCount = (uint32_t)bgEntriesSplit.size(),
                              .entries    = bgEntriesSplit.data(),
              };

    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    int64_t workgroupsX = numSplits;
    int64_t workgroupsY = 1;
    int64_t workgroupsZ = A.shape.b == 0 ? 1: A.shape.b;

    bool hasEncoder = (encoder != nullptr);
    bool hasPass = (pass != nullptr);
    if (!hasEncoder && !hasPass) {
        encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    }
    if (!hasPass) {
        pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    }

    wgpuComputePassEncoderSetPipeline(pass, pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);

    if (!hasPass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass); // There is a question whether we should release before building command buffer.
    }
    if (!hasEncoder && !hasPass) {
        out.cmdBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
    }
    
    wgpuBindGroupRelease(bindGroup);

    return out;
}

} // namespace th
