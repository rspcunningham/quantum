#include "runtime.hpp"

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef QUANTUM_METALLIB_FILENAME
#define QUANTUM_METALLIB_FILENAME "quantum_native_runtime.metallib"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kOpWidth = 8;
constexpr int kDispatchWidth = 4;

constexpr int kOpCodeDiagSubset = 1;
constexpr int kOpCodeDiagFull = 2;
constexpr int kOpCodePermSubset = 3;
constexpr int kOpCodePermFull = 4;
constexpr int kOpCodeDense = 5;
constexpr int kOpCodeMonomialStream = 6;
constexpr uint32_t kHistogramEmptyKey = std::numeric_limits<uint32_t>::max();

struct OpRow {
    int opcode;
    int target_offset;
    int target_len;
    int coeff_offset;
    int coeff_len;
    int flags;
    int aux0;
    int aux1;
};

struct DispatchRow {
    int kernel_id;
    int op_start;
    int op_count;
    int lane_id;
};

struct OpParams {
    uint32_t n_qubits;
    uint32_t dim;
    int32_t target_offset;
    int32_t target_len;
    int32_t coeff_offset;
    int32_t coeff_len;
    int32_t aux0;
    int32_t aux1;
};

struct DispatchGroupParams {
    uint32_t n_qubits;
    uint32_t dim;
    int32_t op_start;
    int32_t op_count;
};

struct MonomialParams {
    uint32_t n_qubits;
    uint32_t dim;
    uint32_t gate_count;
};

struct SamplingProbParams {
    uint32_t dim;
};

struct SamplingScanParams {
    uint32_t dim;
    uint32_t offset;
};

struct SamplingNormalizeParams {
    uint32_t dim;
};

struct SamplingDrawParams {
    uint32_t dim;
    uint32_t n_qubits;
    uint32_t n_bits;
    uint32_t terminal_pairs;
    uint32_t num_shots;
    uint32_t identity_measure;
    uint32_t _pad0;
    uint32_t _pad1;
    uint64_t seed;
};

struct HistogramParams {
    uint32_t num_shots;
    uint32_t table_size;
};

struct MonomialSpecCPU {
    std::vector<int32_t> gate_ks;
    std::vector<int32_t> gate_targets;
    std::vector<int32_t> gate_permutations;
    std::vector<float> gate_factors_re;
    std::vector<float> gate_factors_im;
};

struct MonomialSpecGPU {
    uint32_t gate_count = 0;
    id<MTLBuffer> gate_ks = nil;
    id<MTLBuffer> gate_targets = nil;
    id<MTLBuffer> gate_permutations = nil;
    id<MTLBuffer> gate_factors_re = nil;
    id<MTLBuffer> gate_factors_im = nil;
};

struct Program {
    int n_qubits = 0;
    int n_bits = 0;
    uint64_t dim = 0;
    std::mutex execute_mu;

    std::vector<OpRow> ops;
    std::vector<DispatchRow> dispatch;
    std::vector<int32_t> terminal_measurements;
    id<MTLBuffer> terminal_measurements_buffer = nil;
    id<MTLBuffer> op_table = nil;

    id<MTLBuffer> target_pool = nil;
    id<MTLBuffer> diag_re = nil;
    id<MTLBuffer> diag_im = nil;
    id<MTLBuffer> perm_pool = nil;
    id<MTLBuffer> phase_re = nil;
    id<MTLBuffer> phase_im = nil;
    id<MTLBuffer> dense_re = nil;
    id<MTLBuffer> dense_im = nil;

    id<MTLBuffer> state_a_re = nil;
    id<MTLBuffer> state_a_im = nil;
    id<MTLBuffer> state_b_re = nil;
    id<MTLBuffer> state_b_im = nil;
    uint64_t state_buffer_dim = 0;
    std::int64_t state_buffer_allocations = 0;

    std::vector<MonomialSpecGPU> monomial_specs;
};

struct RuntimeState {
    std::mutex mu;
    std::string module_file_path;
    std::string metallib_override;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;

    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    std::unordered_map<int64_t, std::shared_ptr<Program>> programs;
    int64_t next_handle = 1;
};

RuntimeState& state() {
    static RuntimeState s;
    return s;
}

bool debug_timeline_enabled() {
    const char* v = std::getenv("QUANTUM_METAL_DUMP_NATIVE_TIMELINE");
    return v != nullptr && std::string(v) == "1";
}

void maybe_log(const std::string& msg) {
    if (debug_timeline_enabled()) {
        std::fprintf(stderr, "[native-metal] %s\n", msg.c_str());
    }
}

std::string dirname_of(const std::string& path) {
    const std::size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

std::string default_metallib_path_locked() {
    RuntimeState& s = state();
    if (!s.metallib_override.empty()) {
        return s.metallib_override;
    }
    if (s.module_file_path.empty()) {
        return std::string(QUANTUM_METALLIB_FILENAME);
    }
    return dirname_of(s.module_file_path) + "/" + QUANTUM_METALLIB_FILENAME;
}

void ensure_device_locked() {
    RuntimeState& s = state();
    if (s.device != nil && s.queue != nil) {
        return;
    }
    s.device = MTLCreateSystemDefaultDevice();
    if (s.device == nil) {
        throw std::runtime_error("Metal runtime required; no MTLDevice available");
    }
    s.queue = [s.device newCommandQueue];
    if (s.queue == nil) {
        throw std::runtime_error("Metal runtime initialization failed: command queue unavailable");
    }
}

void ensure_library_locked() {
    RuntimeState& s = state();
    if (s.library != nil) {
        return;
    }
    ensure_device_locked();

    const std::string metallib_path = default_metallib_path_locked();
    NSString* ns_path = [NSString stringWithUTF8String:metallib_path.c_str()];
    NSURL* ns_url = [NSURL fileURLWithPath:ns_path];

    NSError* error = nil;
    s.library = [s.device newLibraryWithURL:ns_url error:&error];
    if (s.library == nil) {
        std::string err = "unknown";
        if (error != nil && error.localizedDescription != nil) {
            err = std::string([[error localizedDescription] UTF8String]);
        }
        throw std::runtime_error("Unable to create Metal library: " + err);
    }
}

id<MTLComputePipelineState> pipeline_locked(const std::string& name) {
    RuntimeState& s = state();
    auto it = s.pipelines.find(name);
    if (it != s.pipelines.end()) {
        return it->second;
    }

    ensure_library_locked();

    NSString* fn_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> fn = [s.library newFunctionWithName:fn_name];
    if (fn == nil) {
        throw std::runtime_error("Missing Metal kernel function: " + name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [s.device newComputePipelineStateWithFunction:fn error:&error];
    if (pipeline == nil) {
        std::string err = "unknown";
        if (error != nil && error.localizedDescription != nil) {
            err = std::string([[error localizedDescription] UTF8String]);
        }
        throw std::runtime_error("Failed to create pipeline " + name + ": " + err);
    }

    s.pipelines.emplace(name, pipeline);
    return pipeline;
}

template <typename T>
id<MTLBuffer> make_buffer(id<MTLDevice> device, const std::vector<T>& values) {
    const std::size_t bytes = values.size() * sizeof(T);
    if (bytes == 0) {
        return nil;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error("Metal buffer allocation failed");
    }
    std::memcpy([buffer contents], values.data(), bytes);
    return buffer;
}

id<MTLBuffer> make_empty_buffer(id<MTLDevice> device, std::size_t bytes) {
    if (bytes == 0) {
        return nil;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error("Metal buffer allocation failed");
    }
    return buffer;
}

void ensure_state_buffers(
    Program& program,
    id<MTLDevice> device,
    uint64_t dim
) {
    if (program.state_buffer_dim == dim
        && program.state_a_re != nil
        && program.state_a_im != nil
        && program.state_b_re != nil
        && program.state_b_im != nil) {
        return;
    }

    const uint64_t bytes = dim * sizeof(float);
    id<MTLBuffer> state_a_re = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> state_a_im = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> state_b_re = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> state_b_im = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (state_a_re == nil || state_a_im == nil || state_b_re == nil || state_b_im == nil) {
        throw std::runtime_error("Metal runtime OOM while allocating persistent state buffers");
    }

    program.state_a_re = state_a_re;
    program.state_a_im = state_a_im;
    program.state_b_re = state_b_re;
    program.state_b_im = state_b_im;
    program.state_buffer_dim = dim;
    program.state_buffer_allocations += 1;
}

template <typename T>
T read_scalar_le(const std::vector<uint8_t>& blob, std::size_t& cursor) {
    if (cursor + sizeof(T) > blob.size()) {
        throw std::runtime_error("Malformed monomial blob: truncated payload");
    }
    T out;
    std::memcpy(&out, blob.data() + cursor, sizeof(T));
    cursor += sizeof(T);
    return out;
}

std::vector<MonomialSpecCPU> parse_monomial_blob(const std::vector<uint8_t>& blob) {
    std::vector<MonomialSpecCPU> specs;
    if (blob.empty()) {
        return specs;
    }

    std::size_t cursor = 0;
    const uint32_t spec_count = read_scalar_le<uint32_t>(blob, cursor);
    specs.reserve(spec_count);

    for (uint32_t spec_idx = 0; spec_idx < spec_count; ++spec_idx) {
        MonomialSpecCPU spec;
        const uint32_t gate_count = read_scalar_le<uint32_t>(blob, cursor);
        spec.gate_ks.reserve(gate_count);
        spec.gate_targets.reserve(static_cast<std::size_t>(gate_count) * 2);
        spec.gate_permutations.reserve(static_cast<std::size_t>(gate_count) * 4);
        spec.gate_factors_re.reserve(static_cast<std::size_t>(gate_count) * 4);
        spec.gate_factors_im.reserve(static_cast<std::size_t>(gate_count) * 4);

        for (uint32_t gate_idx = 0; gate_idx < gate_count; ++gate_idx) {
            spec.gate_ks.push_back(read_scalar_le<int32_t>(blob, cursor));
            spec.gate_targets.push_back(read_scalar_le<int32_t>(blob, cursor));
            spec.gate_targets.push_back(read_scalar_le<int32_t>(blob, cursor));
            for (int i = 0; i < 4; ++i) {
                spec.gate_permutations.push_back(read_scalar_le<int32_t>(blob, cursor));
            }
            for (int i = 0; i < 4; ++i) {
                spec.gate_factors_re.push_back(read_scalar_le<float>(blob, cursor));
            }
            for (int i = 0; i < 4; ++i) {
                spec.gate_factors_im.push_back(read_scalar_le<float>(blob, cursor));
            }
        }

        specs.push_back(std::move(spec));
    }

    if (cursor != blob.size()) {
        throw std::runtime_error("Malformed monomial blob: trailing bytes");
    }

    return specs;
}

inline void validate_static_program(const quantum_native::StaticProgramData& data) {
    if (data.n_qubits <= 0) {
        throw std::runtime_error("n_qubits must be positive");
    }
    if (data.n_qubits > 30) {
        throw std::runtime_error("n_qubits > 30 is unsupported in static-only Metal runtime");
    }
    if (data.n_bits < 0) {
        throw std::runtime_error("n_bits must be non-negative");
    }
    if (data.op_table.size() % kOpWidth != 0) {
        throw std::runtime_error("op_table length must be a multiple of 8");
    }
    if (data.dispatch_table.size() % kDispatchWidth != 0) {
        throw std::runtime_error("dispatch_table length must be a multiple of 4");
    }
    if (data.terminal_measurements.size() % 2 != 0) {
        throw std::runtime_error("terminal_measurements must be flattened pairs");
    }
    if (data.diag_pool_re.size() != data.diag_pool_im.size()) {
        throw std::runtime_error("diag pools are mismatched");
    }
    if (data.phase_pool_re.size() != data.phase_pool_im.size()) {
        throw std::runtime_error("phase pools are mismatched");
    }
    if (data.dense_pool_re.size() != data.dense_pool_im.size()) {
        throw std::runtime_error("dense pools are mismatched");
    }

    const std::size_t op_count = data.op_table.size() / kOpWidth;
    for (std::size_t i = 0; i < op_count; ++i) {
        const std::size_t base = i * kOpWidth;
        const int opcode = data.op_table[base + 0];
        const int target_len = data.op_table[base + 2];

        if (target_len < 0) {
            throw std::runtime_error("op_table contains negative target_len");
        }
        if (opcode == kOpCodeDense && target_len > 6) {
            throw std::runtime_error(
                "Dense gates with arity > 6 are unsupported in static-only Metal runtime."
            );
        }
    }
}

inline uint64_t dim_for_qubits(int n_qubits) {
    if (n_qubits <= 0 || n_qubits > 62) {
        throw std::runtime_error("Invalid qubit count");
    }
    return 1ULL << static_cast<uint64_t>(n_qubits);
}

inline uint32_t next_pow2_u32(uint32_t value) {
    if (value <= 1u) {
        return 1u;
    }
    value -= 1u;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

inline void swap_state_buffers(
    id<MTLBuffer>& in_re,
    id<MTLBuffer>& in_im,
    id<MTLBuffer>& out_re,
    id<MTLBuffer>& out_im
) {
    std::swap(in_re, out_re);
    std::swap(in_im, out_im);
}

inline std::string binary_string(std::int64_t code, int n_bits) {
    if (n_bits <= 0) {
        return std::string();
    }
    std::string out(static_cast<std::size_t>(n_bits), '0');
    for (int i = 0; i < n_bits; ++i) {
        const int shift = n_bits - 1 - i;
        out[static_cast<std::size_t>(i)] = ((code >> shift) & 1LL) != 0 ? '1' : '0';
    }
    return out;
}

inline bool is_identity_measurement_map(const Program& program) {
    const std::size_t term_count = program.terminal_measurements.size() / 2;
    if (program.n_bits != program.n_qubits) {
        return false;
    }
    if (term_count != static_cast<std::size_t>(program.n_qubits)) {
        return false;
    }
    for (int i = 0; i < program.n_qubits; ++i) {
        const int q = program.terminal_measurements[static_cast<std::size_t>(i) * 2 + 0];
        const int b = program.terminal_measurements[static_cast<std::size_t>(i) * 2 + 1];
        if (q != i || b != i) {
            return false;
        }
    }
    return true;
}

void encode_op(
    Program& program,
    OpRow op,
    uint32_t dim,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> in_re,
    id<MTLBuffer> in_im,
    id<MTLBuffer> out_re,
    id<MTLBuffer> out_im
) {
    RuntimeState& s = state();

    OpParams params{};
    params.n_qubits = static_cast<uint32_t>(program.n_qubits);
    params.dim = dim;
    params.target_offset = op.target_offset;
    params.target_len = op.target_len;
    params.coeff_offset = op.coeff_offset;
    params.coeff_len = op.coeff_len;
    params.aux0 = op.aux0;
    params.aux1 = op.aux1;

    id<MTLComputePipelineState> pipeline = nil;

    switch (op.opcode) {
        case kOpCodeDiagFull: {
            pipeline = pipeline_locked("diag_full");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:in_re offset:0 atIndex:0];
            [encoder setBuffer:in_im offset:0 atIndex:1];
            [encoder setBuffer:out_re offset:0 atIndex:2];
            [encoder setBuffer:out_im offset:0 atIndex:3];
            [encoder setBuffer:program.diag_re offset:0 atIndex:4];
            [encoder setBuffer:program.diag_im offset:0 atIndex:5];
            [encoder setBytes:&params length:sizeof(params) atIndex:6];
            break;
        }
        case kOpCodeDiagSubset: {
            pipeline = pipeline_locked("diag_subset");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:in_re offset:0 atIndex:0];
            [encoder setBuffer:in_im offset:0 atIndex:1];
            [encoder setBuffer:out_re offset:0 atIndex:2];
            [encoder setBuffer:out_im offset:0 atIndex:3];
            [encoder setBuffer:program.target_pool offset:0 atIndex:4];
            [encoder setBuffer:program.diag_re offset:0 atIndex:5];
            [encoder setBuffer:program.diag_im offset:0 atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            break;
        }
        case kOpCodePermFull: {
            const bool with_phase = op.aux0 >= 0 && op.aux1 > 0;
            pipeline = pipeline_locked(with_phase ? "perm_full_phase" : "perm_full");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:in_re offset:0 atIndex:0];
            [encoder setBuffer:in_im offset:0 atIndex:1];
            [encoder setBuffer:out_re offset:0 atIndex:2];
            [encoder setBuffer:out_im offset:0 atIndex:3];
            [encoder setBuffer:program.perm_pool offset:0 atIndex:4];
            if (with_phase) {
                [encoder setBuffer:program.phase_re offset:0 atIndex:5];
                [encoder setBuffer:program.phase_im offset:0 atIndex:6];
                [encoder setBytes:&params length:sizeof(params) atIndex:7];
            } else {
                [encoder setBytes:&params length:sizeof(params) atIndex:5];
            }
            break;
        }
        case kOpCodePermSubset: {
            const bool with_phase = op.aux0 >= 0 && op.aux1 > 0;
            pipeline = pipeline_locked(with_phase ? "perm_subset_phase" : "perm_subset");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:in_re offset:0 atIndex:0];
            [encoder setBuffer:in_im offset:0 atIndex:1];
            [encoder setBuffer:out_re offset:0 atIndex:2];
            [encoder setBuffer:out_im offset:0 atIndex:3];
            [encoder setBuffer:program.target_pool offset:0 atIndex:4];
            [encoder setBuffer:program.perm_pool offset:0 atIndex:5];
            if (with_phase) {
                [encoder setBuffer:program.phase_re offset:0 atIndex:6];
                [encoder setBuffer:program.phase_im offset:0 atIndex:7];
                [encoder setBytes:&params length:sizeof(params) atIndex:8];
            } else {
                [encoder setBytes:&params length:sizeof(params) atIndex:6];
            }
            break;
        }
        case kOpCodeDense: {
            if (op.target_len == 1) {
                pipeline = pipeline_locked("dense1");
            } else if (op.target_len == 2) {
                pipeline = pipeline_locked("dense2");
            } else {
                pipeline = pipeline_locked("densek");
            }
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:in_re offset:0 atIndex:0];
            [encoder setBuffer:in_im offset:0 atIndex:1];
            [encoder setBuffer:out_re offset:0 atIndex:2];
            [encoder setBuffer:out_im offset:0 atIndex:3];
            [encoder setBuffer:program.target_pool offset:0 atIndex:4];
            [encoder setBuffer:program.dense_re offset:0 atIndex:5];
            [encoder setBuffer:program.dense_im offset:0 atIndex:6];
            [encoder setBytes:&params length:sizeof(params) atIndex:7];
            break;
        }
        case kOpCodeMonomialStream: {
            const int spec_index = op.coeff_offset;
            if (spec_index < 0 || spec_index >= static_cast<int>(program.monomial_specs.size())) {
                throw std::runtime_error("Invalid monomial spec index in op table");
            }
            const MonomialSpecGPU& spec = program.monomial_specs[static_cast<std::size_t>(spec_index)];
            pipeline = pipeline_locked("monomial_stream");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:in_re offset:0 atIndex:0];
            [encoder setBuffer:in_im offset:0 atIndex:1];
            [encoder setBuffer:out_re offset:0 atIndex:2];
            [encoder setBuffer:out_im offset:0 atIndex:3];
            [encoder setBuffer:spec.gate_ks offset:0 atIndex:4];
            [encoder setBuffer:spec.gate_targets offset:0 atIndex:5];
            [encoder setBuffer:spec.gate_permutations offset:0 atIndex:6];
            [encoder setBuffer:spec.gate_factors_re offset:0 atIndex:7];
            [encoder setBuffer:spec.gate_factors_im offset:0 atIndex:8];
            MonomialParams monomial_params{};
            monomial_params.n_qubits = static_cast<uint32_t>(program.n_qubits);
            monomial_params.dim = dim;
            monomial_params.gate_count = spec.gate_count;
            [encoder setBytes:&monomial_params length:sizeof(monomial_params) atIndex:9];
            break;
        }
        default:
            throw std::runtime_error("Unsupported opcode in native runtime");
    }

    const NSUInteger threads_per_group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    const NSUInteger groups = (static_cast<NSUInteger>(dim) + threads_per_group - 1) / threads_per_group;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
}

void encode_dispatch_group(
    Program& program,
    DispatchRow dispatch,
    uint32_t dim,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> in_re,
    id<MTLBuffer> in_im,
    id<MTLBuffer> out_re,
    id<MTLBuffer> out_im,
    bool& group_common_bound
) {
    const int op_begin = dispatch.op_start;
    const int op_end = dispatch.op_start + dispatch.op_count;
    if (op_begin < 0 || op_end > static_cast<int>(program.ops.size()) || dispatch.op_count <= 0) {
        throw std::runtime_error("Dispatch table references invalid op range");
    }

    const OpRow first = program.ops[static_cast<std::size_t>(op_begin)];
    for (int op_idx = op_begin + 1; op_idx < op_end; ++op_idx) {
        const OpRow op = program.ops[static_cast<std::size_t>(op_idx)];
        if (op.opcode != first.opcode) {
            throw std::runtime_error("Dispatch group mixes opcodes; compiler invariant violated");
        }
    }

    if ((first.opcode == kOpCodeDense || first.opcode == kOpCodeMonomialStream) && dispatch.op_count != 1) {
        throw std::runtime_error(
            "Dense/monomial dispatch groups must be single-op in native runtime"
        );
    }

    if (first.opcode == kOpCodeDense || first.opcode == kOpCodeMonomialStream) {
        encode_op(program, first, dim, encoder, in_re, in_im, out_re, out_im);
        group_common_bound = false;
        return;
    }

    if (program.op_table == nil) {
        throw std::runtime_error("Missing op table buffer in native runtime");
    }
    if (!group_common_bound) {
        [encoder setBuffer:program.op_table offset:0 atIndex:4];
        [encoder setBuffer:program.target_pool offset:0 atIndex:5];
        [encoder setBuffer:program.diag_re offset:0 atIndex:6];
        [encoder setBuffer:program.diag_im offset:0 atIndex:7];
        [encoder setBuffer:program.perm_pool offset:0 atIndex:8];
        [encoder setBuffer:program.phase_re offset:0 atIndex:9];
        [encoder setBuffer:program.phase_im offset:0 atIndex:10];
        group_common_bound = true;
    }

    DispatchGroupParams params{};
    params.n_qubits = static_cast<uint32_t>(program.n_qubits);
    params.dim = dim;
    params.op_start = dispatch.op_start;
    params.op_count = dispatch.op_count;

    id<MTLComputePipelineState> pipeline = nil;
    switch (first.opcode) {
        case kOpCodeDiagFull: {
            pipeline = pipeline_locked("diag_full_group");
            break;
        }
        case kOpCodeDiagSubset: {
            pipeline = pipeline_locked("diag_subset_group");
            break;
        }
        case kOpCodePermFull: {
            pipeline = pipeline_locked("perm_full_group");
            break;
        }
        case kOpCodePermSubset: {
            pipeline = pipeline_locked("perm_subset_group");
            break;
        }
        default:
            throw std::runtime_error("Unsupported grouped opcode in native runtime");
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:in_re offset:0 atIndex:0];
    [encoder setBuffer:in_im offset:0 atIndex:1];
    [encoder setBuffer:out_re offset:0 atIndex:2];
    [encoder setBuffer:out_im offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:11];

    const NSUInteger threads_per_group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    const NSUInteger groups = (static_cast<NSUInteger>(dim) + threads_per_group - 1) / threads_per_group;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
}

std::shared_ptr<Program> program_for_handle(int64_t handle) {
    RuntimeState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = s.programs.find(handle);
    if (it == s.programs.end()) {
        throw std::runtime_error("Unknown native program handle");
    }
    return it->second;
}

}  // namespace

namespace quantum_native {

void set_module_file_path(const std::string& module_file_path) {
    RuntimeState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.module_file_path = module_file_path;
}

void set_metallib_path_override(const std::string& metallib_path) {
    RuntimeState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.metallib_override = metallib_path;
    s.library = nil;
    s.pipelines.clear();
}

std::int64_t compile_static_program(const StaticProgramData& data) {
    const auto t0 = Clock::now();
    validate_static_program(data);

    std::vector<MonomialSpecCPU> monomial_specs = parse_monomial_blob(data.monomial_blob);

    RuntimeState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);

    ensure_library_locked();

    auto program = std::make_shared<Program>();
    program->n_qubits = data.n_qubits;
    program->n_bits = data.n_bits;
    program->dim = dim_for_qubits(data.n_qubits);

    const std::size_t op_count = data.op_table.size() / kOpWidth;
    program->ops.reserve(op_count);
    for (std::size_t i = 0; i < op_count; ++i) {
        const std::size_t base = i * kOpWidth;
        program->ops.push_back(OpRow{
            data.op_table[base + 0],
            data.op_table[base + 1],
            data.op_table[base + 2],
            data.op_table[base + 3],
            data.op_table[base + 4],
            data.op_table[base + 5],
            data.op_table[base + 6],
            data.op_table[base + 7],
        });
    }
    program->op_table = make_buffer(s.device, data.op_table);

    const std::size_t dispatch_count = data.dispatch_table.size() / kDispatchWidth;
    program->dispatch.reserve(dispatch_count);
    for (std::size_t i = 0; i < dispatch_count; ++i) {
        const std::size_t base = i * kDispatchWidth;
        program->dispatch.push_back(DispatchRow{
            data.dispatch_table[base + 0],
            data.dispatch_table[base + 1],
            data.dispatch_table[base + 2],
            data.dispatch_table[base + 3],
        });
    }

    program->terminal_measurements = data.terminal_measurements;
    program->terminal_measurements_buffer = make_buffer(s.device, data.terminal_measurements);

    program->target_pool = make_buffer(s.device, data.target_pool);
    program->diag_re = make_buffer(s.device, data.diag_pool_re);
    program->diag_im = make_buffer(s.device, data.diag_pool_im);
    program->perm_pool = make_buffer(s.device, data.perm_pool);
    program->phase_re = make_buffer(s.device, data.phase_pool_re);
    program->phase_im = make_buffer(s.device, data.phase_pool_im);
    program->dense_re = make_buffer(s.device, data.dense_pool_re);
    program->dense_im = make_buffer(s.device, data.dense_pool_im);

    program->monomial_specs.reserve(monomial_specs.size());
    for (const MonomialSpecCPU& spec_cpu : monomial_specs) {
        MonomialSpecGPU spec_gpu;
        spec_gpu.gate_count = static_cast<uint32_t>(spec_cpu.gate_ks.size());
        spec_gpu.gate_ks = make_buffer(s.device, spec_cpu.gate_ks);
        spec_gpu.gate_targets = make_buffer(s.device, spec_cpu.gate_targets);
        spec_gpu.gate_permutations = make_buffer(s.device, spec_cpu.gate_permutations);
        spec_gpu.gate_factors_re = make_buffer(s.device, spec_cpu.gate_factors_re);
        spec_gpu.gate_factors_im = make_buffer(s.device, spec_cpu.gate_factors_im);
        program->monomial_specs.push_back(spec_gpu);
    }

    const int64_t handle = s.next_handle++;
    s.programs.emplace(handle, program);

    if (debug_timeline_enabled()) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
        std::ostringstream oss;
        oss << "compile handle=" << handle << " dim=" << program->dim << " ops=" << program->ops.size()
            << " dispatch=" << program->dispatch.size() << " ms=" << ms;
        maybe_log(oss.str());
    }

    return handle;
}

std::unordered_map<std::string, std::int64_t> execute_static_program(
    std::int64_t handle,
    std::int64_t num_shots,
    std::optional<std::uint64_t> seed
) {
    if (num_shots < 0) {
        throw std::runtime_error("num_shots must be non-negative");
    }
    if (num_shots == 0) {
        return {};
    }

    const auto t0 = Clock::now();

    std::shared_ptr<Program> program = program_for_handle(handle);
    RuntimeState& s = state();
    std::lock_guard<std::mutex> execute_lock(program->execute_mu);

    if (program->n_qubits > 30) {
        throw std::runtime_error("n_qubits > 30 is unsupported in static-only Metal runtime");
    }
    const uint64_t dim = program->dim;
    if (dim == 0 || dim > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("State dimension exceeds native runtime limits");
    }
    if (num_shots > static_cast<std::int64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("num_shots exceeds native runtime limits");
    }
    const uint32_t dim_u32 = static_cast<uint32_t>(dim);
    const uint32_t shots_u32 = static_cast<uint32_t>(num_shots);

    const std::size_t terminal_count = program->terminal_measurements.size() / 2;
    for (std::size_t t = 0; t < terminal_count; ++t) {
        const int q = program->terminal_measurements[t * 2 + 0];
        const int b = program->terminal_measurements[t * 2 + 1];
        if (q < 0 || q >= program->n_qubits || b < 0 || b >= program->n_bits) {
            throw std::runtime_error("Invalid terminal measurement mapping");
        }
    }
    const bool identity_measure = is_identity_measurement_map(*program);

    uint32_t histogram_table_size = 0;
    uint64_t histogram_bytes_total = 0;
    if (program->n_bits > 0) {
        if (shots_u32 > std::numeric_limits<uint32_t>::max() / 2u) {
            throw std::runtime_error("num_shots too large for native histogram table");
        }
        const uint32_t min_hist_size = std::max<uint32_t>(1u, shots_u32 * 2u);
        histogram_table_size = next_pow2_u32(min_hist_size);
        histogram_bytes_total = static_cast<uint64_t>(histogram_table_size) * sizeof(uint32_t) * 2ULL;
    }

    const uint64_t state_bytes_total = dim * sizeof(float) * 4ULL;
    const uint64_t sampling_bytes_total =
        (program->n_bits > 0)
            ? (dim * sizeof(float) * 2ULL
               + static_cast<uint64_t>(shots_u32) * sizeof(uint32_t)
               + histogram_bytes_total)
            : 0ULL;
    uint64_t recommended = 0;
    if ([s.device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        recommended = [s.device recommendedMaxWorkingSetSize];
    }
    if (recommended > 0
        && (state_bytes_total + sampling_bytes_total) > (recommended * 8ULL) / 10ULL) {
        throw std::runtime_error("State allocation exceeds recommended Metal working set size");
    }

    ensure_state_buffers(*program, s.device, dim);

    id<MTLBuffer> state_in_re = program->state_a_re;
    id<MTLBuffer> state_in_im = program->state_a_im;
    id<MTLBuffer> state_out_re = program->state_b_re;
    id<MTLBuffer> state_out_im = program->state_b_im;

    std::memset([state_in_re contents], 0, dim * sizeof(float));
    std::memset([state_in_im contents], 0, dim * sizeof(float));
    static_cast<float*>([state_in_re contents])[0] = 1.0f;

    id<MTLBuffer> sample_probs_a = nil;
    id<MTLBuffer> sample_probs_b = nil;
    id<MTLBuffer> sampled_codes = nil;
    id<MTLBuffer> hist_keys = nil;
    id<MTLBuffer> hist_counts = nil;
    if (program->n_bits > 0) {
        sample_probs_a = make_empty_buffer(s.device, static_cast<std::size_t>(dim) * sizeof(float));
        sample_probs_b = make_empty_buffer(s.device, static_cast<std::size_t>(dim) * sizeof(float));
        sampled_codes = make_empty_buffer(s.device, static_cast<std::size_t>(shots_u32) * sizeof(uint32_t));
        hist_keys = make_empty_buffer(
            s.device,
            static_cast<std::size_t>(histogram_table_size) * sizeof(uint32_t)
        );
        hist_counts = make_empty_buffer(
            s.device,
            static_cast<std::size_t>(histogram_table_size) * sizeof(uint32_t)
        );
        if (terminal_count > 0 && program->terminal_measurements_buffer == nil) {
            throw std::runtime_error("Missing terminal measurement buffer in native runtime");
        }
        if (hist_keys == nil || hist_counts == nil) {
            throw std::runtime_error("Failed to allocate native histogram buffers");
        }
        std::memset(
            [hist_keys contents],
            0xFF,
            static_cast<std::size_t>(histogram_table_size) * sizeof(uint32_t)
        );
        std::memset(
            [hist_counts contents],
            0,
            static_cast<std::size_t>(histogram_table_size) * sizeof(uint32_t)
        );
    }

    const uint64_t resolved_seed = seed.value_or(static_cast<uint64_t>(std::random_device{}()));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
        if (cmd == nil) {
            throw std::runtime_error("Failed to allocate Metal command buffer");
        }

        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        if (encoder == nil) {
            throw std::runtime_error("Failed to allocate Metal command encoder");
        }

        bool group_common_bound = false;
        for (const DispatchRow& dispatch : program->dispatch) {
            if (dispatch.op_count <= 0) {
                continue;
            }
            encode_dispatch_group(
                *program,
                dispatch,
                static_cast<uint32_t>(dim),
                encoder,
                state_in_re,
                state_in_im,
                state_out_re,
                state_out_im,
                group_common_bound
            );
            swap_state_buffers(state_in_re, state_in_im, state_out_re, state_out_im);
        }

        [encoder endEncoding];

        if (program->n_bits > 0) {
            id<MTLComputeCommandEncoder> sampling_encoder = [cmd computeCommandEncoder];
            if (sampling_encoder == nil) {
                throw std::runtime_error("Failed to allocate sampling command encoder");
            }

            {
                id<MTLComputePipelineState> pipeline = pipeline_locked("compute_probabilities");
                [sampling_encoder setComputePipelineState:pipeline];
                [sampling_encoder setBuffer:state_in_re offset:0 atIndex:0];
                [sampling_encoder setBuffer:state_in_im offset:0 atIndex:1];
                [sampling_encoder setBuffer:sample_probs_a offset:0 atIndex:2];
                SamplingProbParams params{dim_u32};
                [sampling_encoder setBytes:&params length:sizeof(params) atIndex:3];

                const NSUInteger threads_per_group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
                const NSUInteger groups = (static_cast<NSUInteger>(dim_u32) + threads_per_group - 1) / threads_per_group;
                [sampling_encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            }

            id<MTLBuffer> cdf_in = sample_probs_a;
            id<MTLBuffer> cdf_out = sample_probs_b;
            for (uint32_t offset = 1; offset < dim_u32; offset <<= 1) {
                id<MTLComputePipelineState> pipeline = pipeline_locked("inclusive_scan_step");
                [sampling_encoder setComputePipelineState:pipeline];
                [sampling_encoder setBuffer:cdf_in offset:0 atIndex:0];
                [sampling_encoder setBuffer:cdf_out offset:0 atIndex:1];
                SamplingScanParams params{dim_u32, offset};
                [sampling_encoder setBytes:&params length:sizeof(params) atIndex:2];

                const NSUInteger threads_per_group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
                const NSUInteger groups = (static_cast<NSUInteger>(dim_u32) + threads_per_group - 1) / threads_per_group;
                [sampling_encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
                std::swap(cdf_in, cdf_out);
            }

            {
                id<MTLComputePipelineState> pipeline = pipeline_locked("normalize_cdf");
                [sampling_encoder setComputePipelineState:pipeline];
                [sampling_encoder setBuffer:cdf_in offset:0 atIndex:0];
                SamplingNormalizeParams params{dim_u32};
                [sampling_encoder setBytes:&params length:sizeof(params) atIndex:1];

                const NSUInteger threads_per_group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
                const NSUInteger groups = (static_cast<NSUInteger>(dim_u32) + threads_per_group - 1) / threads_per_group;
                [sampling_encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            }

            {
                id<MTLComputePipelineState> pipeline = pipeline_locked("sample_indices_to_codes");
                [sampling_encoder setComputePipelineState:pipeline];
                [sampling_encoder setBuffer:cdf_in offset:0 atIndex:0];
                [sampling_encoder setBuffer:program->terminal_measurements_buffer offset:0 atIndex:1];
                [sampling_encoder setBuffer:sampled_codes offset:0 atIndex:2];
                SamplingDrawParams params{};
                params.dim = dim_u32;
                params.n_qubits = static_cast<uint32_t>(program->n_qubits);
                params.n_bits = static_cast<uint32_t>(program->n_bits);
                params.terminal_pairs = static_cast<uint32_t>(terminal_count);
                params.num_shots = shots_u32;
                params.identity_measure = identity_measure ? 1u : 0u;
                params.seed = resolved_seed;
                [sampling_encoder setBytes:&params length:sizeof(params) atIndex:3];

                const NSUInteger threads_per_group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
                const NSUInteger groups = (static_cast<NSUInteger>(shots_u32) + threads_per_group - 1) / threads_per_group;
                [sampling_encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            }

            {
                id<MTLComputePipelineState> pipeline = pipeline_locked("histogram_codes");
                [sampling_encoder setComputePipelineState:pipeline];
                [sampling_encoder setBuffer:sampled_codes offset:0 atIndex:0];
                [sampling_encoder setBuffer:hist_keys offset:0 atIndex:1];
                [sampling_encoder setBuffer:hist_counts offset:0 atIndex:2];
                HistogramParams params{};
                params.num_shots = shots_u32;
                params.table_size = histogram_table_size;
                [sampling_encoder setBytes:&params length:sizeof(params) atIndex:3];

                const NSUInteger threads_per_group = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
                const NSUInteger groups = (static_cast<NSUInteger>(shots_u32) + threads_per_group - 1) / threads_per_group;
                [sampling_encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            }

            [sampling_encoder endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status != MTLCommandBufferStatusCompleted) {
            std::string err = "unknown";
            if (cmd.error != nil && cmd.error.localizedDescription != nil) {
                err = std::string([cmd.error.localizedDescription UTF8String]);
            }
            throw std::runtime_error("Metal command buffer failed: " + err);
        }
    }

    if (program->n_bits == 0) {
        return {{"", num_shots}};
    }

    if (hist_keys == nil || hist_counts == nil) {
        throw std::runtime_error("Sampling histogram buffers missing in native runtime");
    }
    const uint32_t* key_data = static_cast<const uint32_t*>([hist_keys contents]);
    const uint32_t* count_data = static_cast<const uint32_t*>([hist_counts contents]);
    std::unordered_map<std::int64_t, std::int64_t> code_counts;
    code_counts.reserve(std::min<std::size_t>(
        static_cast<std::size_t>(histogram_table_size),
        static_cast<std::size_t>(shots_u32)
    ));
    uint64_t total_counted = 0;
    for (uint32_t i = 0; i < histogram_table_size; ++i) {
        const uint32_t bucket_count = count_data[i];
        if (bucket_count == 0) {
            continue;
        }
        const uint32_t key = key_data[i];
        if (key == kHistogramEmptyKey) {
            throw std::runtime_error("Histogram corruption: empty key has non-zero count");
        }
        code_counts[static_cast<std::int64_t>(key)] += static_cast<std::int64_t>(bucket_count);
        total_counted += static_cast<uint64_t>(bucket_count);
    }
    if (total_counted != shots_u32) {
        throw std::runtime_error("Histogram count mismatch in native runtime");
    }

    std::unordered_map<std::string, std::int64_t> output;
    output.reserve(code_counts.size());
    for (const auto& entry : code_counts) {
        output.emplace(binary_string(entry.first, program->n_bits), entry.second);
    }

    if (debug_timeline_enabled()) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
        std::ostringstream oss;
        oss << "execute handle=" << handle << " shots=" << num_shots << " dim=" << dim
            << " counts=" << output.size() << " ms=" << ms;
        maybe_log(oss.str());
    }

    return output;
}

std::unordered_map<std::string, std::int64_t> get_program_stats(std::int64_t handle) {
    std::shared_ptr<Program> program = program_for_handle(handle);

    std::unordered_map<std::string, std::int64_t> stats;
    stats.emplace("n_qubits", static_cast<std::int64_t>(program->n_qubits));
    stats.emplace("n_bits", static_cast<std::int64_t>(program->n_bits));
    stats.emplace("op_count", static_cast<std::int64_t>(program->ops.size()));
    stats.emplace("dispatch_count", static_cast<std::int64_t>(program->dispatch.size()));
    stats.emplace(
        "terminal_measurement_pairs",
        static_cast<std::int64_t>(program->terminal_measurements.size() / 2)
    );
    stats.emplace("monomial_spec_count", static_cast<std::int64_t>(program->monomial_specs.size()));
    stats.emplace(
        "target_pool_len",
        static_cast<std::int64_t>(program->target_pool == nil ? 0 : [program->target_pool length] / sizeof(int32_t))
    );
    stats.emplace(
        "diag_pool_len",
        static_cast<std::int64_t>(program->diag_re == nil ? 0 : [program->diag_re length] / sizeof(float))
    );
    stats.emplace(
        "perm_pool_len",
        static_cast<std::int64_t>(program->perm_pool == nil ? 0 : [program->perm_pool length] / sizeof(int32_t))
    );
    stats.emplace(
        "phase_pool_len",
        static_cast<std::int64_t>(program->phase_re == nil ? 0 : [program->phase_re length] / sizeof(float))
    );
    stats.emplace(
        "dense_pool_len",
        static_cast<std::int64_t>(program->dense_re == nil ? 0 : [program->dense_re length] / sizeof(float))
    );
    stats.emplace(
        "state_buffer_allocations",
        static_cast<std::int64_t>(program->state_buffer_allocations)
    );
    stats.emplace(
        "state_buffer_dim",
        static_cast<std::int64_t>(program->state_buffer_dim)
    );
    return stats;
}

void free_program(std::int64_t handle) {
    RuntimeState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.programs.erase(handle);
}

}  // namespace quantum_native
