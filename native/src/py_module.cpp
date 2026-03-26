#include "runtime.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int32_t kOpcodeDiagSubset = 1;
constexpr int32_t kOpcodeDiagFull = 2;
constexpr int32_t kOpcodePermSubset = 3;
constexpr int32_t kOpcodePermFull = 4;
constexpr int32_t kOpcodeDense = 5;
constexpr int32_t kOpcodeMonomialStream = 6;

constexpr int32_t kMaxDenseArity = 6;

constexpr float kAbsTol = 1e-6f;
constexpr float kRelTol = 1e-5f;

// ---------------------------------------------------------------------------
// Gate kind enum — matches Python-side constants in gates.py
// ---------------------------------------------------------------------------

enum class GateKind : int32_t {
    I = 0,
    H = 1,
    X = 2,
    Y = 3,
    Z = 4,
    S = 5,
    Sdg = 6,
    T = 7,
    Tdg = 8,
    SX = 9,
    RX = 10,
    RY = 11,
    RZ = 12,
    CX = 13,
    CZ = 14,
    CCX = 15,
    SWAP = 16,
    CP = 17,
};

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

struct PairHash {
    std::size_t operator()(const std::pair<int32_t, int32_t>& value) const noexcept {
        const std::uint64_t packed =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(value.first)) << 32)
            | static_cast<std::uint32_t>(value.second);
        return static_cast<std::size_t>(packed);
    }
};

struct PoolOffset {
    int32_t offset;
    int32_t length;
};

struct OpRecord {
    int32_t opcode;
    int32_t target_offset;
    int32_t target_len;
    int32_t coeff_offset;
    int32_t coeff_len;
    int32_t flags;
    int32_t aux0;
    int32_t aux1;
};

struct DispatchRecord {
    int32_t kernel_id;
    int32_t op_start;
    int32_t op_count;
    int32_t lane_id;
};

enum class LocalKind {
    Diagonal,
    Permutation,
    Dense,
    MonomialStream,
};

struct LocalOp {
    LocalKind kind = LocalKind::Dense;
    std::vector<int32_t> targets;
    std::vector<std::complex<float>> diagonal;
    std::vector<int32_t> permutation;
    std::vector<std::complex<float>> phase;
    std::vector<std::complex<float>> dense;
    int32_t monomial_spec_index = -1;

    static LocalOp monomial_stream(int32_t spec_index) {
        LocalOp op;
        op.kind = LocalKind::MonomialStream;
        op.monomial_spec_index = spec_index;
        return op;
    }
};

struct MonomialGate {
    int32_t k = 0;
    int32_t target0 = -1;
    int32_t target1 = -1;
    std::array<int32_t, 4> perm{};
    std::array<std::complex<float>, 4> factors{};
};

struct MonomialSpec {
    std::vector<MonomialGate> gates;
};

// ---------------------------------------------------------------------------
// Pybind11-exposed types
// ---------------------------------------------------------------------------

struct NativeGate {
    LocalOp op;
};

struct NativeMeasurement {
    int32_t qubit;
    int32_t bit;
};

struct NativeConditionalGate {
    NativeGate gate;
    int32_t condition;
};

// ---------------------------------------------------------------------------
// Gate factory
// ---------------------------------------------------------------------------

LocalOp make_gate_op(int32_t kind, const std::vector<int32_t>& targets, float param) {
    LocalOp op;
    op.targets = targets;

    static const float INV_SQRT2 = 1.0f / std::sqrt(2.0f);

    switch (static_cast<GateKind>(kind)) {
    case GateKind::I:
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{1, 0}, {1, 0}};
        break;

    case GateKind::H:
        op.kind = LocalKind::Dense;
        op.dense = {{INV_SQRT2, 0}, {INV_SQRT2, 0}, {INV_SQRT2, 0}, {-INV_SQRT2, 0}};
        break;

    case GateKind::X:
        op.kind = LocalKind::Permutation;
        op.permutation = {1, 0};
        break;

    case GateKind::Y:
        op.kind = LocalKind::Permutation;
        op.permutation = {1, 0};
        op.phase = {{0, -1}, {0, 1}};
        break;

    case GateKind::Z:
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{1, 0}, {-1, 0}};
        break;

    case GateKind::S:
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{1, 0}, {0, 1}};
        break;

    case GateKind::Sdg:
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{1, 0}, {0, -1}};
        break;

    case GateKind::T:
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{1, 0}, {INV_SQRT2, INV_SQRT2}};
        break;

    case GateKind::Tdg:
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{1, 0}, {INV_SQRT2, -INV_SQRT2}};
        break;

    case GateKind::SX:
        op.kind = LocalKind::Dense;
        op.dense = {{0.5f, 0.5f}, {0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}};
        break;

    case GateKind::RX: {
        const float c = std::cos(param / 2.0f);
        const float s = std::sin(param / 2.0f);
        op.kind = LocalKind::Dense;
        op.dense = {{c, 0}, {0, -s}, {0, -s}, {c, 0}};
        break;
    }

    case GateKind::RY: {
        const float c = std::cos(param / 2.0f);
        const float s = std::sin(param / 2.0f);
        op.kind = LocalKind::Dense;
        op.dense = {{c, 0}, {-s, 0}, {s, 0}, {c, 0}};
        break;
    }

    case GateKind::RZ: {
        const float c = std::cos(param / 2.0f);
        const float s = std::sin(param / 2.0f);
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{c, -s}, {c, s}};
        break;
    }

    case GateKind::CX:
        op.kind = LocalKind::Permutation;
        op.permutation = {0, 1, 3, 2};
        break;

    case GateKind::CZ:
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{1, 0}, {1, 0}, {1, 0}, {-1, 0}};
        break;

    case GateKind::CCX:
        op.kind = LocalKind::Permutation;
        op.permutation = {0, 1, 2, 3, 4, 5, 7, 6};
        break;

    case GateKind::SWAP:
        op.kind = LocalKind::Permutation;
        op.permutation = {0, 2, 1, 3};
        break;

    case GateKind::CP: {
        const float c = std::cos(param);
        const float s = std::sin(param);
        op.kind = LocalKind::Diagonal;
        op.diagonal = {{1, 0}, {1, 0}, {1, 0}, {c, s}};
        break;
    }

    default:
        throw std::runtime_error("Unknown gate kind: " + std::to_string(kind));
    }

    return op;
}

// ---------------------------------------------------------------------------
// Helpers used by optimization passes
// ---------------------------------------------------------------------------

template <typename T>
void append_scalar(std::vector<uint8_t>& out, T value) {
    const std::size_t offset = out.size();
    out.resize(offset + sizeof(T));
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

bool approx_eq(std::complex<float> a, std::complex<float> b) {
    const float diff = std::abs(a - b);
    const float scale = kAbsTol + kRelTol * std::abs(b);
    return diff <= scale;
}

bool is_identity_diagonal(const std::vector<std::complex<float>>& diag) {
    for (const std::complex<float>& value : diag) {
        if (!approx_eq(value, std::complex<float>(1.0f, 0.0f))) {
            return false;
        }
    }
    return true;
}

bool targets_equal(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int local_dim(const LocalOp& op) {
    return 1 << static_cast<int>(op.targets.size());
}

std::vector<std::complex<float>> dense_matrix(const LocalOp& op) {
    const int dim = local_dim(op);
    std::vector<std::complex<float>> out(static_cast<std::size_t>(dim) * dim, std::complex<float>(0.0f, 0.0f));

    switch (op.kind) {
        case LocalKind::Diagonal: {
            if (static_cast<int>(op.diagonal.size()) != dim) {
                throw std::runtime_error("Malformed diagonal op in native compiler");
            }
            for (int i = 0; i < dim; ++i) {
                out[static_cast<std::size_t>(i) * dim + i] = op.diagonal[static_cast<std::size_t>(i)];
            }
            return out;
        }
        case LocalKind::Permutation: {
            if (static_cast<int>(op.permutation.size()) != dim) {
                throw std::runtime_error("Malformed permutation op in native compiler");
            }
            for (int row = 0; row < dim; ++row) {
                const int col = op.permutation[static_cast<std::size_t>(row)];
                std::complex<float> factor(1.0f, 0.0f);
                if (!op.phase.empty()) {
                    if (static_cast<int>(op.phase.size()) != dim) {
                        throw std::runtime_error("Malformed permutation-phase op in native compiler");
                    }
                    factor = op.phase[static_cast<std::size_t>(row)];
                }
                out[static_cast<std::size_t>(row) * dim + col] = factor;
            }
            return out;
        }
        case LocalKind::Dense: {
            if (static_cast<int>(op.dense.size()) != dim * dim) {
                throw std::runtime_error("Malformed dense op in native compiler");
            }
            return op.dense;
        }
        case LocalKind::MonomialStream:
            throw std::runtime_error("Unexpected monomial stream op in dense_matrix");
    }

    throw std::runtime_error("Unknown local op kind");
}

bool is_identity_matrix(const std::vector<std::complex<float>>& matrix, int dim) {
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            const std::complex<float> expected =
                (r == c) ? std::complex<float>(1.0f, 0.0f) : std::complex<float>(0.0f, 0.0f);
            if (!approx_eq(matrix[static_cast<std::size_t>(r) * dim + c], expected)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::complex<float>> matmul(
    const std::vector<std::complex<float>>& a,
    const std::vector<std::complex<float>>& b,
    int dim
) {
    std::vector<std::complex<float>> out(static_cast<std::size_t>(dim) * dim, std::complex<float>(0.0f, 0.0f));
    for (int r = 0; r < dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            std::complex<float> acc(0.0f, 0.0f);
            for (int k = 0; k < dim; ++k) {
                acc += a[static_cast<std::size_t>(r) * dim + k] * b[static_cast<std::size_t>(k) * dim + c];
            }
            out[static_cast<std::size_t>(r) * dim + c] = acc;
        }
    }
    return out;
}

bool is_inverse_pair(const LocalOp& current, const LocalOp& previous) {
    if (!targets_equal(current.targets, previous.targets)) {
        return false;
    }

    if (current.kind == LocalKind::Diagonal && previous.kind == LocalKind::Diagonal) {
        if (current.diagonal.size() != previous.diagonal.size()) {
            return false;
        }
        for (std::size_t i = 0; i < current.diagonal.size(); ++i) {
            const std::complex<float> prod = current.diagonal[i] * previous.diagonal[i];
            if (!approx_eq(prod, std::complex<float>(1.0f, 0.0f))) {
                return false;
            }
        }
        return true;
    }

    if (current.targets.size() > 2) {
        return false;
    }

    const int dim = local_dim(current);
    const std::vector<std::complex<float>> mat_current = dense_matrix(current);
    const std::vector<std::complex<float>> mat_previous = dense_matrix(previous);
    const std::vector<std::complex<float>> product = matmul(mat_current, mat_previous, dim);
    return is_identity_matrix(product, dim);
}

// ---------------------------------------------------------------------------
// Optimization passes
// ---------------------------------------------------------------------------

void pass_inverse_cancellation(std::vector<LocalOp>& ops) {
    if (ops.size() < 2) {
        return;
    }

    bool has_candidate = false;
    for (std::size_t i = 0; i + 1 < ops.size(); ++i) {
        if (targets_equal(ops[i].targets, ops[i + 1].targets)) {
            has_candidate = true;
            break;
        }
    }
    if (!has_candidate) {
        return;
    }

    std::vector<LocalOp> stack;
    stack.reserve(ops.size());
    for (const LocalOp& op : ops) {
        if (!stack.empty() && is_inverse_pair(op, stack.back())) {
            stack.pop_back();
        } else {
            stack.push_back(op);
        }
    }

    if (stack.size() != ops.size()) {
        ops = std::move(stack);
    }
}

void pass_local_diagonal_compaction(std::vector<LocalOp>& ops) {
    if (ops.size() < 2) {
        return;
    }

    bool has_candidate = false;
    for (std::size_t i = 0; i + 1 < ops.size(); ++i) {
        if (
            ops[i].kind == LocalKind::Diagonal
            && ops[i + 1].kind == LocalKind::Diagonal
            && ops[i].targets.size() <= 2
            && ops[i + 1].targets.size() <= 2
        ) {
            has_candidate = true;
            break;
        }
    }
    if (!has_candidate) {
        return;
    }

    std::vector<LocalOp> out;
    out.reserve(ops.size());

    std::size_t i = 0;
    while (i < ops.size()) {
        if (ops[i].kind != LocalKind::Diagonal || ops[i].targets.size() > 2) {
            out.push_back(ops[i]);
            ++i;
            continue;
        }

        std::size_t j = i + 1;
        while (
            j < ops.size()
            && ops[j].kind == LocalKind::Diagonal
            && ops[j].targets.size() <= 2
        ) {
            ++j;
        }

        if (j - i < 2) {
            out.push_back(ops[i]);
            i = j;
            continue;
        }

        std::vector<int32_t> oneq_order;
        std::vector<std::pair<int32_t, int32_t>> twoq_order;

        std::unordered_map<int32_t, std::array<std::complex<float>, 2>> oneq_fused;
        std::unordered_map<std::pair<int32_t, int32_t>, std::array<std::complex<float>, 4>, PairHash> twoq_fused;

        for (std::size_t k = i; k < j; ++k) {
            const LocalOp& op = ops[k];
            if (op.targets.size() == 1) {
                const int32_t q = op.targets[0];
                auto it = oneq_fused.find(q);
                if (it == oneq_fused.end()) {
                    oneq_order.push_back(q);
                    oneq_fused.emplace(
                        q,
                        std::array<std::complex<float>, 2>{op.diagonal[0], op.diagonal[1]}
                    );
                } else {
                    it->second[0] *= op.diagonal[0];
                    it->second[1] *= op.diagonal[1];
                }
                continue;
            }

            int32_t t0 = op.targets[0];
            int32_t t1 = op.targets[1];
            std::array<std::complex<float>, 4> vals{
                op.diagonal[0],
                op.diagonal[1],
                op.diagonal[2],
                op.diagonal[3],
            };
            if (t0 > t1) {
                std::swap(t0, t1);
                vals = {vals[0], vals[2], vals[1], vals[3]};
            }

            const std::pair<int32_t, int32_t> key{t0, t1};
            auto it = twoq_fused.find(key);
            if (it == twoq_fused.end()) {
                twoq_order.push_back(key);
                twoq_fused.emplace(key, vals);
            } else {
                for (int idx = 0; idx < 4; ++idx) {
                    it->second[static_cast<std::size_t>(idx)] *= vals[static_cast<std::size_t>(idx)];
                }
            }
        }

        std::vector<LocalOp> fused_run;
        fused_run.reserve(oneq_order.size() + twoq_order.size());

        for (int32_t q : oneq_order) {
            const auto& vals = oneq_fused[q];
            std::vector<std::complex<float>> diag{vals[0], vals[1]};
            if (is_identity_diagonal(diag)) {
                continue;
            }
            LocalOp op;
            op.kind = LocalKind::Diagonal;
            op.targets = {q};
            op.diagonal = std::move(diag);
            fused_run.push_back(std::move(op));
        }

        for (const std::pair<int32_t, int32_t>& key : twoq_order) {
            const auto& vals = twoq_fused[key];
            std::vector<std::complex<float>> diag{vals[0], vals[1], vals[2], vals[3]};
            if (is_identity_diagonal(diag)) {
                continue;
            }
            LocalOp op;
            op.kind = LocalKind::Diagonal;
            op.targets = {key.first, key.second};
            op.diagonal = std::move(diag);
            fused_run.push_back(std::move(op));
        }

        if (fused_run.size() >= (j - i)) {
            out.insert(out.end(), ops.begin() + static_cast<std::ptrdiff_t>(i), ops.begin() + static_cast<std::ptrdiff_t>(j));
        } else {
            out.insert(
                out.end(),
                std::make_move_iterator(fused_run.begin()),
                std::make_move_iterator(fused_run.end())
            );
        }

        i = j;
    }

    if (out.size() != ops.size()) {
        ops = std::move(out);
    }
}

bool to_monomial_gate(const LocalOp& op, MonomialGate& gate) {
    if (op.targets.size() != 1 && op.targets.size() != 2) {
        return false;
    }
    if (op.kind != LocalKind::Diagonal && op.kind != LocalKind::Permutation) {
        return false;
    }

    gate = MonomialGate{};
    gate.k = static_cast<int32_t>(op.targets.size());
    gate.target0 = op.targets[0];
    gate.target1 = (op.targets.size() == 2) ? op.targets[1] : -1;
    gate.factors = {
        std::complex<float>(1.0f, 0.0f),
        std::complex<float>(1.0f, 0.0f),
        std::complex<float>(1.0f, 0.0f),
        std::complex<float>(1.0f, 0.0f),
    };

    if (op.kind == LocalKind::Diagonal) {
        const int dim = 1 << gate.k;
        for (int idx = 0; idx < dim; ++idx) {
            gate.perm[static_cast<std::size_t>(idx)] = idx;
            gate.factors[static_cast<std::size_t>(idx)] = op.diagonal[static_cast<std::size_t>(idx)];
        }
        return true;
    }

    const int dim = 1 << gate.k;
    if (static_cast<int>(op.permutation.size()) != dim) {
        return false;
    }
    for (int idx = 0; idx < dim; ++idx) {
        gate.perm[static_cast<std::size_t>(idx)] = op.permutation[static_cast<std::size_t>(idx)];
    }

    if (!op.phase.empty()) {
        if (static_cast<int>(op.phase.size()) != dim) {
            return false;
        }
        for (int idx = 0; idx < dim; ++idx) {
            gate.factors[static_cast<std::size_t>(idx)] = op.phase[static_cast<std::size_t>(idx)];
        }
    }

    return true;
}

void pass_monomial_stream_packing(
    std::vector<LocalOp>& ops,
    std::vector<MonomialSpec>& specs,
    int min_run_len
) {
    if (ops.size() < static_cast<std::size_t>(min_run_len)) {
        return;
    }

    std::vector<LocalOp> out;
    out.reserve(ops.size());

    std::size_t i = 0;
    while (i < ops.size()) {
        MonomialGate first_gate;
        if (!to_monomial_gate(ops[i], first_gate)) {
            out.push_back(ops[i]);
            ++i;
            continue;
        }

        std::size_t j = i;
        std::vector<MonomialGate> run;
        run.reserve(16);

        while (j < ops.size()) {
            MonomialGate gate;
            if (!to_monomial_gate(ops[j], gate)) {
                break;
            }
            run.push_back(gate);
            ++j;
        }

        if (run.size() >= static_cast<std::size_t>(min_run_len)) {
            MonomialSpec spec;
            spec.gates = std::move(run);
            const int32_t spec_index = static_cast<int32_t>(specs.size());
            specs.push_back(std::move(spec));
            out.push_back(LocalOp::monomial_stream(spec_index));
        } else {
            out.insert(out.end(), ops.begin() + static_cast<std::ptrdiff_t>(i), ops.begin() + static_cast<std::ptrdiff_t>(j));
        }

        i = j;
    }

    if (out.size() != ops.size()) {
        ops = std::move(out);
    }
}

// ---------------------------------------------------------------------------
// Pool helpers and lowering
// ---------------------------------------------------------------------------

std::string key_from_i32(const std::vector<int32_t>& values) {
    if (values.empty()) {
        return std::string();
    }
    return std::string(
        reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(int32_t)
    );
}

std::string key_from_complex(const std::vector<std::complex<float>>& values) {
    if (values.empty()) {
        return std::string();
    }
    return std::string(
        reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(std::complex<float>)
    );
}

PoolOffset add_target_pool(
    const std::vector<int32_t>& targets,
    std::vector<int32_t>& target_pool,
    std::unordered_map<std::string, PoolOffset>& cache
) {
    const std::string key = key_from_i32(targets);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const int32_t offset = static_cast<int32_t>(target_pool.size());
    target_pool.insert(target_pool.end(), targets.begin(), targets.end());
    const PoolOffset result{offset, static_cast<int32_t>(targets.size())};
    cache.emplace(key, result);
    return result;
}

PoolOffset add_int_pool(
    const std::vector<int32_t>& values,
    std::vector<int32_t>& pool,
    std::unordered_map<std::string, PoolOffset>& cache
) {
    const std::string key = key_from_i32(values);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const int32_t offset = static_cast<int32_t>(pool.size());
    pool.insert(pool.end(), values.begin(), values.end());
    const PoolOffset result{offset, static_cast<int32_t>(values.size())};
    cache.emplace(key, result);
    return result;
}

PoolOffset add_complex_pool(
    const std::vector<std::complex<float>>& values,
    std::vector<float>& pool_re,
    std::vector<float>& pool_im,
    std::unordered_map<std::string, PoolOffset>& cache
) {
    const std::string key = key_from_complex(values);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const int32_t offset = static_cast<int32_t>(pool_re.size());
    pool_re.reserve(pool_re.size() + values.size());
    pool_im.reserve(pool_im.size() + values.size());
    for (const std::complex<float>& v : values) {
        pool_re.push_back(v.real());
        pool_im.push_back(v.imag());
    }

    const PoolOffset result{offset, static_cast<int32_t>(values.size())};
    cache.emplace(key, result);
    return result;
}

int32_t kernel_id_for_op(const OpRecord& op) {
    return static_cast<int32_t>((op.opcode << 4) | (op.target_len & 0xF));
}

std::vector<uint8_t> serialize_monomial_blob(const std::vector<MonomialSpec>& specs) {
    std::vector<uint8_t> blob;
    append_scalar<uint32_t>(blob, static_cast<uint32_t>(specs.size()));

    for (const MonomialSpec& spec : specs) {
        append_scalar<uint32_t>(blob, static_cast<uint32_t>(spec.gates.size()));
        for (const MonomialGate& gate : spec.gates) {
            append_scalar<int32_t>(blob, gate.k);
            append_scalar<int32_t>(blob, gate.target0);
            append_scalar<int32_t>(blob, gate.target1);
            for (int idx = 0; idx < 4; ++idx) {
                append_scalar<int32_t>(blob, gate.perm[static_cast<std::size_t>(idx)]);
            }
            for (int idx = 0; idx < 4; ++idx) {
                append_scalar<float>(blob, gate.factors[static_cast<std::size_t>(idx)].real());
            }
            for (int idx = 0; idx < 4; ++idx) {
                append_scalar<float>(blob, gate.factors[static_cast<std::size_t>(idx)].imag());
            }
        }
    }

    return blob;
}

// ---------------------------------------------------------------------------
// Compile local ops to program
// ---------------------------------------------------------------------------

std::int64_t compile_local_ops_to_program(
    std::vector<LocalOp> ops,
    std::vector<int32_t> terminal_measurements,
    int n_qubits,
    int n_bits
) {
    pass_inverse_cancellation(ops);
    pass_local_diagonal_compaction(ops);

    std::vector<MonomialSpec> monomial_specs;
    const char* disable_monomial = std::getenv("QUANTUM_DISABLE_MONOMIAL_STREAM");
    if (!(disable_monomial != nullptr && std::string(disable_monomial) == "1")) {
        int min_run_len = 8;
        if (const char* min_run_env = std::getenv("QUANTUM_MONOMIAL_MIN_RUN")) {
            const int parsed = std::atoi(min_run_env);
            if (parsed > 1) {
                min_run_len = parsed;
            }
        }
        pass_monomial_stream_packing(ops, monomial_specs, min_run_len);
    }

    std::vector<int32_t> target_pool;
    std::unordered_map<std::string, PoolOffset> target_cache;

    std::vector<float> diag_pool_re;
    std::vector<float> diag_pool_im;
    std::unordered_map<std::string, PoolOffset> diag_cache;

    std::vector<int32_t> perm_pool;
    std::unordered_map<std::string, PoolOffset> perm_cache;

    std::vector<float> phase_pool_re;
    std::vector<float> phase_pool_im;
    std::unordered_map<std::string, PoolOffset> phase_cache;

    std::vector<float> dense_pool_re;
    std::vector<float> dense_pool_im;
    std::unordered_map<std::string, PoolOffset> dense_cache;

    std::vector<OpRecord> lowered_ops;
    lowered_ops.reserve(ops.size());

    for (const LocalOp& op : ops) {
        if (op.kind == LocalKind::MonomialStream) {
            lowered_ops.push_back(OpRecord{
                kOpcodeMonomialStream,
                0,
                0,
                op.monomial_spec_index,
                1,
                0,
                -1,
                -1,
            });
            continue;
        }

        const PoolOffset target_meta = add_target_pool(op.targets, target_pool, target_cache);

        if (op.kind == LocalKind::Diagonal) {
            const PoolOffset coeff = add_complex_pool(op.diagonal, diag_pool_re, diag_pool_im, diag_cache);
            const int32_t opcode =
                (static_cast<int>(op.targets.size()) == n_qubits) ? kOpcodeDiagFull : kOpcodeDiagSubset;
            lowered_ops.push_back(OpRecord{
                opcode,
                target_meta.offset,
                target_meta.length,
                coeff.offset,
                coeff.length,
                0,
                -1,
                -1,
            });
            continue;
        }

        if (op.kind == LocalKind::Permutation) {
            const PoolOffset perm_meta = add_int_pool(op.permutation, perm_pool, perm_cache);
            int32_t aux0 = -1;
            int32_t aux1 = 0;
            if (!op.phase.empty()) {
                const PoolOffset phase_meta = add_complex_pool(op.phase, phase_pool_re, phase_pool_im, phase_cache);
                aux0 = phase_meta.offset;
                aux1 = phase_meta.length;
            }
            const int32_t opcode =
                (static_cast<int>(op.targets.size()) == n_qubits) ? kOpcodePermFull : kOpcodePermSubset;
            lowered_ops.push_back(OpRecord{
                opcode,
                target_meta.offset,
                target_meta.length,
                perm_meta.offset,
                perm_meta.length,
                0,
                aux0,
                aux1,
            });
            continue;
        }

        if (static_cast<int32_t>(op.targets.size()) > kMaxDenseArity) {
            throw std::runtime_error(
                "Dense gates with arity > 6 are unsupported in static-only Metal runtime."
            );
        }
        const PoolOffset dense_meta = add_complex_pool(op.dense, dense_pool_re, dense_pool_im, dense_cache);
        lowered_ops.push_back(OpRecord{
            kOpcodeDense,
            target_meta.offset,
            target_meta.length,
            dense_meta.offset,
            dense_meta.length,
            0,
            -1,
            -1,
        });
    }

    std::vector<DispatchRecord> dispatch;
    if (!lowered_ops.empty()) {
        int32_t group_start = 0;
        int32_t prev_kernel = kernel_id_for_op(lowered_ops[0]);
        for (int32_t i = 1; i < static_cast<int32_t>(lowered_ops.size()); ++i) {
            const OpRecord& prev_op = lowered_ops[static_cast<std::size_t>(i - 1)];
            const OpRecord& curr_op = lowered_ops[static_cast<std::size_t>(i)];
            const bool prev_dense_like =
                (prev_op.opcode == kOpcodeDense) || (prev_op.opcode == kOpcodeMonomialStream);
            const bool curr_dense_like =
                (curr_op.opcode == kOpcodeDense) || (curr_op.opcode == kOpcodeMonomialStream);
            const int32_t kernel = kernel_id_for_op(curr_op);
            if (prev_dense_like || curr_dense_like || kernel != prev_kernel) {
                dispatch.push_back(DispatchRecord{
                    prev_kernel,
                    group_start,
                    i - group_start,
                    0,
                });
                group_start = i;
                prev_kernel = kernel;
            }
        }
        dispatch.push_back(DispatchRecord{
            prev_kernel,
            group_start,
            static_cast<int32_t>(lowered_ops.size()) - group_start,
            0,
        });
    }

    std::vector<int32_t> op_table;
    op_table.reserve(lowered_ops.size() * 8);
    for (const OpRecord& op : lowered_ops) {
        op_table.push_back(op.opcode);
        op_table.push_back(op.target_offset);
        op_table.push_back(op.target_len);
        op_table.push_back(op.coeff_offset);
        op_table.push_back(op.coeff_len);
        op_table.push_back(op.flags);
        op_table.push_back(op.aux0);
        op_table.push_back(op.aux1);
    }

    std::vector<int32_t> dispatch_table;
    dispatch_table.reserve(dispatch.size() * 4);
    for (const DispatchRecord& row : dispatch) {
        dispatch_table.push_back(row.kernel_id);
        dispatch_table.push_back(row.op_start);
        dispatch_table.push_back(row.op_count);
        dispatch_table.push_back(row.lane_id);
    }

    quantum_native::StaticProgramData data{};
    data.n_qubits = n_qubits;
    data.n_bits = n_bits;
    data.op_table = std::move(op_table);
    data.dispatch_table = std::move(dispatch_table);
    data.target_pool = std::move(target_pool);
    data.diag_pool_re = std::move(diag_pool_re);
    data.diag_pool_im = std::move(diag_pool_im);
    data.perm_pool = std::move(perm_pool);
    data.phase_pool_re = std::move(phase_pool_re);
    data.phase_pool_im = std::move(phase_pool_im);
    data.dense_pool_re = std::move(dense_pool_re);
    data.dense_pool_im = std::move(dense_pool_im);
    data.monomial_blob = serialize_monomial_blob(monomial_specs);
    data.terminal_measurements = std::move(terminal_measurements);

    return quantum_native::compile_static_program(data);
}

// ---------------------------------------------------------------------------
// Circuit extraction — walk a flat Python list of NativeGate/NativeMeasurement
// ---------------------------------------------------------------------------

struct SegmentEntry {
    LocalOp op;
    int32_t condition;  // -1 = unconditional
};

struct DynamicSegment {
    std::vector<SegmentEntry> entries;
};

struct MidMeasurement {
    int32_t qubit;
    int32_t bit;
};

struct DynamicCircuit {
    int n_qubits;
    int n_bits;
    bool is_static;
    std::vector<DynamicSegment> segments;
    std::vector<MidMeasurement> measurements;  // measurements[i] is between segment[i] and segment[i+1]
    std::vector<int32_t> terminal_measurements;
};

DynamicCircuit extract_dynamic_circuit(py::list flat_ops, int n_qubits, int n_bits) {
    const std::size_t n = flat_ops.size();

    // Find terminal measurement boundary
    std::size_t terminal_start = n;
    while (terminal_start > 0
           && py::isinstance<NativeMeasurement>(flat_ops[terminal_start - 1])) {
        terminal_start--;
    }

    DynamicCircuit result;
    result.n_qubits = n_qubits;
    result.n_bits = n_bits;
    result.is_static = true;

    DynamicSegment current_segment;

    for (std::size_t i = 0; i < terminal_start; ++i) {
        py::handle op = flat_ops[i];

        if (py::isinstance<NativeGate>(op)) {
            current_segment.entries.push_back(
                SegmentEntry{op.cast<NativeGate&>().op, -1}
            );
        } else if (py::isinstance<NativeMeasurement>(op)) {
            // Mid-circuit measurement: close current segment, record measurement
            result.is_static = false;
            result.segments.push_back(std::move(current_segment));
            current_segment = DynamicSegment{};
            auto m = op.cast<NativeMeasurement&>();
            result.measurements.push_back(MidMeasurement{m.qubit, m.bit});
        } else if (py::isinstance<NativeConditionalGate>(op)) {
            result.is_static = false;
            auto cond = op.cast<NativeConditionalGate&>();
            current_segment.entries.push_back(
                SegmentEntry{cond.gate.op, cond.condition}
            );
        } else {
            throw std::runtime_error("Unknown operation type in circuit");
        }
    }

    // Last segment (gates after the last mid-circuit measurement)
    result.segments.push_back(std::move(current_segment));

    // Terminal measurements
    result.terminal_measurements.reserve((n - terminal_start) * 2);
    for (std::size_t i = terminal_start; i < n; ++i) {
        if (!py::isinstance<NativeMeasurement>(flat_ops[i])) {
            throw std::runtime_error(
                "Non-measurement op found in terminal measurement block"
            );
        }
        auto m = flat_ops[i].cast<NativeMeasurement&>();
        result.terminal_measurements.push_back(m.qubit);
        result.terminal_measurements.push_back(m.bit);
    }

    return result;
}

// Resolve conditional gates for a known classical register state.
// Unconditional gates are always included; conditional gates are included
// only if the full classical register integer matches the condition.
std::vector<LocalOp> resolve_segment(
    const DynamicSegment& segment,
    uint32_t classical_reg
) {
    std::vector<LocalOp> ops;
    ops.reserve(segment.entries.size());
    for (const auto& entry : segment.entries) {
        if (entry.condition < 0) {
            ops.push_back(entry.op);
        } else if (static_cast<uint32_t>(entry.condition) == classical_reg) {
            ops.push_back(entry.op);
        }
        // else: skip (condition not met)
    }
    return ops;
}

// Check if two state vectors are equal within tolerance.
bool states_equal(const float* a_re, const float* a_im,
                  const float* b_re, const float* b_im,
                  uint64_t dim) {
    constexpr float tol = 1e-5f;
    for (uint64_t i = 0; i < dim; ++i) {
        float dr = a_re[i] - b_re[i];
        float di = a_im[i] - b_im[i];
        if (dr * dr + di * di > tol * tol) {
            return false;
        }
    }
    return true;
}

// Execute dynamic circuit via iterative branch processing with pruning.
std::unordered_map<std::string, std::int64_t> execute_dynamic_circuit(
    const DynamicCircuit& circuit,
    std::int64_t num_shots,
    std::optional<std::uint64_t> seed
) {
    const int n_qubits = circuit.n_qubits;
    const int n_bits = circuit.n_bits;
    const uint64_t dim = 1ULL << n_qubits;
    const std::size_t bytes = static_cast<std::size_t>(dim) * sizeof(float);

    const std::size_t prob_size = std::max<std::size_t>(
        static_cast<std::size_t>(dim),
        n_bits > 0 ? (1ULL << n_bits) : 1
    );
    std::vector<double> combined_probs(prob_size, 0.0);

    float* scratch_re = static_cast<float*>(std::calloc(dim, sizeof(float)));
    float* scratch_im = static_cast<float*>(std::calloc(dim, sizeof(float)));

    auto run_segment = [&](std::size_t seg_idx, uint32_t classical_reg,
                           float* state_re, float* state_im) {
        std::vector<LocalOp> ops = resolve_segment(circuit.segments[seg_idx], classical_reg);
        if (ops.empty()) return;
        const std::int64_t handle = compile_local_ops_to_program(
            std::move(ops), {}, n_qubits, 0
        );
        try {
            quantum_native::execute_gates_only(
                handle, state_re, state_im, scratch_re, scratch_im, dim
            );
            quantum_native::free_program(handle);
        } catch (...) {
            try { quantum_native::free_program(handle); } catch (...) {}
            throw;
        }
    };

    auto collapse = [&](float* re, float* im, int qubit, int outcome) {
        const uint64_t bit_pos = static_cast<uint64_t>(n_qubits - 1 - qubit);
        const uint64_t mask = 1ULL << bit_pos;
        const uint64_t target = (outcome == 1) ? mask : 0ULL;
        double norm_sq = 0.0;
        for (uint64_t i = 0; i < dim; ++i) {
            if ((i & mask) != target) {
                re[i] = 0.0f; im[i] = 0.0f;
            } else {
                norm_sq += static_cast<double>(re[i]) * re[i]
                         + static_cast<double>(im[i]) * im[i];
            }
        }
        float scale = 1.0f / std::sqrt(static_cast<float>(norm_sq));
        for (uint64_t i = 0; i < dim; ++i) {
            re[i] *= scale; im[i] *= scale;
        }
    };

    auto accumulate_leaf = [&](const float* state_re, const float* state_im,
                               uint32_t classical_reg, double branch_prob) {
        const std::size_t terminal_count = circuit.terminal_measurements.size() / 2;
        if (terminal_count == 0) {
            for (uint64_t i = 0; i < dim; ++i) {
                double re = state_re[i], im = state_im[i];
                combined_probs[i] += branch_prob * (re * re + im * im);
            }
        } else {
            for (uint64_t basis = 0; basis < dim; ++basis) {
                double re = state_re[basis], im = state_im[basis];
                double prob = re * re + im * im;
                if (prob < 1e-15) continue;
                uint32_t code = classical_reg;
                for (std::size_t t = 0; t < terminal_count; ++t) {
                    int q = circuit.terminal_measurements[t * 2 + 0];
                    int b = circuit.terminal_measurements[t * 2 + 1];
                    uint32_t qubit_shift = static_cast<uint32_t>(n_qubits - 1 - q);
                    uint32_t bit_shift = static_cast<uint32_t>(n_bits - 1 - b);
                    uint32_t measured = (static_cast<uint32_t>(basis) >> qubit_shift) & 1u;
                    code = (code & ~(1u << bit_shift)) | (measured << bit_shift);
                }
                combined_probs[code] += branch_prob * prob;
            }
        }
    };

    // Iterative: maintain a list of active branches.
    // Process one segment+measurement at a time across all branches.
    struct Branch {
        uint32_t classical_reg;
        double prob;
        std::vector<float> state_re;
        std::vector<float> state_im;
    };

    // Initialize with single branch: |0...0⟩
    std::vector<Branch> branches;
    branches.push_back(Branch{0, 1.0, std::vector<float>(dim, 0.0f), std::vector<float>(dim, 0.0f)});
    branches[0].state_re[0] = 1.0f;

    // Process each segment + measurement.
    // seg_already_run tracks whether we pre-executed the current segment
    // during the previous iteration's pruning step.
    bool seg_already_run = false;

    for (std::size_t seg = 0; seg < circuit.segments.size(); ++seg) {
        // Execute this segment for all branches (unless pre-executed by pruning)
        if (!seg_already_run) {
            for (auto& branch : branches) {
                run_segment(seg, branch.classical_reg,
                           branch.state_re.data(), branch.state_im.data());
            }
        }
        seg_already_run = false;

        // If there's a measurement after this segment, fork branches
        if (seg < circuit.measurements.size()) {
            const MidMeasurement& meas = circuit.measurements[seg];
            const uint64_t bit_pos = static_cast<uint64_t>(n_qubits - 1 - meas.qubit);
            const uint64_t mask = 1ULL << bit_pos;
            const uint32_t reg_bit_shift = static_cast<uint32_t>(n_bits - 1 - meas.bit);

            std::vector<Branch> next_branches;
            next_branches.reserve(branches.size() * 2);

            for (auto& branch : branches) {
                // Compute p(outcome=0)
                double prob_0 = 0.0;
                for (uint64_t i = 0; i < dim; ++i) {
                    if ((i & mask) == 0) {
                        double re = branch.state_re[i], im = branch.state_im[i];
                        prob_0 += re * re + im * im;
                    }
                }
                double prob_1 = 1.0 - prob_0;

                if (prob_0 > 1e-12) {
                    Branch b0;
                    b0.classical_reg = branch.classical_reg & ~(1u << reg_bit_shift);
                    b0.prob = branch.prob * prob_0;
                    b0.state_re = branch.state_re;  // copy
                    b0.state_im = branch.state_im;
                    collapse(b0.state_re.data(), b0.state_im.data(), meas.qubit, 0);
                    next_branches.push_back(std::move(b0));
                }
                if (prob_1 > 1e-12) {
                    Branch b1;
                    b1.classical_reg = branch.classical_reg | (1u << reg_bit_shift);
                    b1.prob = branch.prob * prob_1;
                    b1.state_re = std::move(branch.state_re);  // move (last use)
                    b1.state_im = std::move(branch.state_im);
                    collapse(b1.state_re.data(), b1.state_im.data(), meas.qubit, 1);
                    next_branches.push_back(std::move(b1));
                }
            }

            // Pruning: merge branches with identical quantum states.
            // Execute next segment first so conditionals are resolved before comparison.
            if (seg + 1 < circuit.segments.size()) {
                for (auto& branch : next_branches) {
                    run_segment(seg + 1, branch.classical_reg,
                               branch.state_re.data(), branch.state_im.data());
                }

                // Compare all pairs and merge branches with identical
                // classical register AND quantum state.
                std::vector<bool> merged(next_branches.size(), false);
                for (std::size_t i = 0; i < next_branches.size(); ++i) {
                    if (merged[i]) continue;
                    for (std::size_t j = i + 1; j < next_branches.size(); ++j) {
                        if (merged[j]) continue;
                        if (next_branches[i].classical_reg == next_branches[j].classical_reg
                            && states_equal(next_branches[i].state_re.data(),
                                           next_branches[i].state_im.data(),
                                           next_branches[j].state_re.data(),
                                           next_branches[j].state_im.data(), dim)) {
                            next_branches[i].prob += next_branches[j].prob;
                            merged[j] = true;
                        }
                    }
                }

                // Remove merged branches
                std::vector<Branch> pruned;
                pruned.reserve(next_branches.size());
                for (std::size_t i = 0; i < next_branches.size(); ++i) {
                    if (!merged[i]) {
                        pruned.push_back(std::move(next_branches[i]));
                    }
                }

                // Mark next segment as already executed (don't skip it —
                // we still need to process its measurement)
                seg_already_run = true;
                branches = std::move(pruned);
            } else {
                branches = std::move(next_branches);
            }
        }
        // else: no measurement, just continue to next segment (or leaf)
    }

    // All segments processed — accumulate leaf distributions
    for (const auto& branch : branches) {
        accumulate_leaf(branch.state_re.data(), branch.state_im.data(),
                       branch.classical_reg, branch.prob);
    }

    std::free(scratch_re);
    std::free(scratch_im);

    // Sample num_shots from the combined probability distribution
    const uint64_t resolved_seed = seed.value_or(static_cast<uint64_t>(std::random_device{}()));
    std::mt19937_64 rng(resolved_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Build CDF
    std::vector<double> cdf(combined_probs.size());
    double total = 0.0;
    for (std::size_t i = 0; i < combined_probs.size(); ++i) {
        total += combined_probs[i];
        cdf[i] = total;
    }
    // Normalize
    if (total > 0.0) {
        for (auto& v : cdf) v /= total;
    }
    cdf.back() = 1.0;  // ensure last entry is exactly 1

    // Sample
    std::unordered_map<uint32_t, std::int64_t> code_counts;
    for (std::int64_t s = 0; s < num_shots; ++s) {
        double u = dist(rng);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        uint32_t code = static_cast<uint32_t>(std::distance(cdf.begin(), it));
        if (code >= static_cast<uint32_t>(cdf.size())) {
            code = static_cast<uint32_t>(cdf.size() - 1);
        }
        code_counts[code]++;
    }

    // Convert to bitstring output
    // Determine the number of output bits
    int output_bits = n_bits;
    if (output_bits == 0 && circuit.terminal_measurements.empty()) {
        output_bits = n_qubits;
    }

    std::unordered_map<std::string, std::int64_t> output;
    output.reserve(code_counts.size());
    for (const auto& entry : code_counts) {
        std::string bits(output_bits, '0');
        for (int b = 0; b < output_bits; ++b) {
            if ((entry.first >> (output_bits - 1 - b)) & 1u) {
                bits[b] = '1';
            }
        }
        output[bits] = entry.second;
    }

    return output;
}

}  // namespace

PYBIND11_MODULE(quantum_native_runtime, m) {
    m.doc() = "Native Metal runtime for quantum circuit simulation";

    m.def(
        "set_module_file_path",
        &quantum_native::set_module_file_path,
        py::arg("module_file_path")
    );

    m.def(
        "set_metallib_path_override",
        &quantum_native::set_metallib_path_override,
        py::arg("metallib_path")
    );

    // -- Exposed types --

    py::class_<NativeGate>(m, "NativeGate")
        .def("targets", [](const NativeGate& g) -> std::vector<int32_t> {
            return g.op.targets;
        })
        .def("matrix", [](const NativeGate& g) -> py::array_t<std::complex<float>> {
            auto mat = dense_matrix(g.op);
            const int dim = 1 << static_cast<int>(g.op.targets.size());
            py::array_t<std::complex<float>> result({dim, dim});
            auto buf = result.mutable_unchecked<2>();
            for (int r = 0; r < dim; ++r) {
                for (int c = 0; c < dim; ++c) {
                    buf(r, c) = mat[static_cast<std::size_t>(r) * dim + c];
                }
            }
            return result;
        });

    py::class_<NativeMeasurement>(m, "NativeMeasurement")
        .def(py::init<int32_t, int32_t>(), py::arg("qubit"), py::arg("bit"))
        .def_readonly("qubit", &NativeMeasurement::qubit)
        .def_readonly("bit", &NativeMeasurement::bit);

    py::class_<NativeConditionalGate>(m, "NativeConditionalGate")
        .def_readonly("gate", &NativeConditionalGate::gate)
        .def_readonly("condition", &NativeConditionalGate::condition);

    // -- Gate factory --

    m.def(
        "make_gate",
        [](int32_t kind, std::vector<int32_t> targets, float param) -> NativeGate {
            return NativeGate{make_gate_op(kind, targets, param)};
        },
        py::arg("kind"),
        py::arg("targets"),
        py::arg("param") = 0.0f
    );

    m.def(
        "make_diagonal_gate",
        [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> diagonal,
           std::vector<int32_t> targets) -> NativeGate {
            auto buf = diagonal.unchecked<1>();
            LocalOp op;
            op.kind = LocalKind::Diagonal;
            op.targets = std::move(targets);
            op.diagonal.reserve(static_cast<std::size_t>(buf.shape(0)));
            for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                op.diagonal.push_back(buf(i));
            }
            return NativeGate{std::move(op)};
        },
        py::arg("diagonal"),
        py::arg("targets")
    );

    m.def(
        "make_dense_gate",
        [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> matrix,
           std::vector<int32_t> targets) -> NativeGate {
            auto buf = matrix.unchecked<2>();
            const int dim = static_cast<int>(buf.shape(0));
            LocalOp op;
            op.kind = LocalKind::Dense;
            op.targets = std::move(targets);
            op.dense.reserve(static_cast<std::size_t>(dim * dim));
            for (int r = 0; r < dim; ++r) {
                for (int c = 0; c < dim; ++c) {
                    op.dense.push_back(buf(r, c));
                }
            }
            return NativeGate{std::move(op)};
        },
        py::arg("matrix"),
        py::arg("targets")
    );

    m.def(
        "make_conditional",
        [](NativeGate gate, int32_t condition) -> NativeConditionalGate {
            return NativeConditionalGate{std::move(gate), condition};
        },
        py::arg("gate"),
        py::arg("condition")
    );

    // -- Compile and execute --

    m.def(
        "compile_circuit",
        [](py::list flat_ops, int n_qubits, int n_bits) -> std::int64_t {
            auto circuit = extract_dynamic_circuit(flat_ops, n_qubits, n_bits);
            if (!circuit.is_static) {
                throw std::runtime_error(
                    "compile_circuit does not support dynamic circuits. Use run_circuit instead."
                );
            }
            // Extract ops from the single segment
            std::vector<LocalOp> ops;
            for (auto& entry : circuit.segments[0].entries) {
                ops.push_back(std::move(entry.op));
            }
            return compile_local_ops_to_program(
                std::move(ops),
                std::move(circuit.terminal_measurements),
                n_qubits,
                n_bits
            );
        },
        py::arg("flat_ops"),
        py::arg("n_qubits"),
        py::arg("n_bits")
    );

    m.def(
        "run_circuit",
        [](py::list flat_ops, int n_qubits, int n_bits,
           std::int64_t num_shots, py::object seed_obj, double timeout)
            -> std::unordered_map<std::string, std::int64_t>
        {
            std::optional<std::uint64_t> seed = std::nullopt;
            if (!seed_obj.is_none()) {
                seed = seed_obj.cast<std::uint64_t>();
            }

            auto circuit = extract_dynamic_circuit(flat_ops, n_qubits, n_bits);

            if (circuit.is_static) {
                // Static fast path — same as before
                std::vector<LocalOp> ops;
                for (auto& entry : circuit.segments[0].entries) {
                    ops.push_back(std::move(entry.op));
                }
                const std::int64_t handle = compile_local_ops_to_program(
                    std::move(ops),
                    std::move(circuit.terminal_measurements),
                    n_qubits,
                    n_bits
                );
                try {
                    auto counts = quantum_native::execute_static_program(handle, num_shots, seed, timeout);
                    quantum_native::free_program(handle);
                    return counts;
                } catch (...) {
                    try { quantum_native::free_program(handle); } catch (...) {}
                    throw;
                }
            }

            // Dynamic path — branch tree execution
            return execute_dynamic_circuit(circuit, num_shots, seed);
        },
        py::arg("flat_ops"),
        py::arg("n_qubits"),
        py::arg("n_bits"),
        py::arg("num_shots"),
        py::arg("seed") = py::none(),
        py::arg("timeout") = 0.0
    );

    m.def(
        "execute_static_program",
        [](std::int64_t handle, std::int64_t num_shots, py::object seed_obj, double timeout) {
            std::optional<std::uint64_t> seed = std::nullopt;
            if (!seed_obj.is_none()) {
                seed = seed_obj.cast<std::uint64_t>();
            }
            return quantum_native::execute_static_program(handle, num_shots, seed, timeout);
        },
        py::arg("handle"),
        py::arg("num_shots"),
        py::arg("seed") = py::none(),
        py::arg("timeout") = 0.0
    );

    m.def(
        "get_program_stats",
        &quantum_native::get_program_stats,
        py::arg("handle")
    );

    m.def(
        "free_program",
        &quantum_native::free_program,
        py::arg("handle")
    );
}
