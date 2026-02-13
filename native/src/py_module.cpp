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
#include <optional>
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

constexpr int32_t kCanonicalKindDiagonal = 1;
constexpr int32_t kCanonicalKindPermutation = 2;
constexpr int32_t kCanonicalKindDense = 3;
constexpr int32_t kMaxDenseArity = 6;
constexpr uint32_t kPackedStaticAbiMagic = 0x31505351U;   // "QSP1"
constexpr uint32_t kPackedStaticAbiVersion = 1U;

constexpr float kAbsTol = 1e-6f;
constexpr float kRelTol = 1e-5f;

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

template <typename T>
void append_scalar(std::vector<uint8_t>& out, T value) {
    const std::size_t offset = out.size();
    out.resize(offset + sizeof(T));
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

template <typename T>
T read_scalar(const std::string& blob, std::size_t& cursor, const char* field_name) {
    if (cursor + sizeof(T) > blob.size()) {
        throw std::runtime_error(
            std::string("Malformed packed static circuit: truncated field ") + field_name
        );
    }
    T out;
    std::memcpy(&out, blob.data() + cursor, sizeof(T));
    cursor += sizeof(T);
    return out;
}

template <typename T>
std::vector<T> read_array(
    const std::string& blob,
    std::size_t& cursor,
    std::size_t count,
    const char* field_name
) {
    const std::size_t bytes = count * sizeof(T);
    if (cursor + bytes > blob.size()) {
        throw std::runtime_error(
            std::string("Malformed packed static circuit: truncated array ") + field_name
        );
    }

    std::vector<T> out(count);
    if (bytes > 0) {
        std::memcpy(out.data(), blob.data() + cursor, bytes);
    }
    cursor += bytes;
    return out;
}

std::vector<std::complex<float>> read_complex_array(
    const std::string& blob,
    std::size_t& cursor,
    std::size_t count,
    const char* field_name
) {
    std::vector<std::complex<float>> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float re = read_scalar<float>(blob, cursor, field_name);
        const float im = read_scalar<float>(blob, cursor, field_name);
        out.emplace_back(re, im);
    }
    return out;
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

template <typename T>
std::vector<T> vector_from_array(const py::object& obj, const char* arg_name) {
    py::array_t<T, py::array::c_style | py::array::forcecast> arr =
        py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw py::type_error(std::string("Expected numeric array-like for argument: ") + arg_name);
    }

    const T* data = static_cast<const T*>(arr.data());
    const py::ssize_t size = arr.size();
    return std::vector<T>(data, data + size);
}

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

std::vector<int32_t> i32_vector_from_sequence(const py::handle& obj, const char* field_name) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    const py::ssize_t length = py::len(seq);
    if (length < 0) {
        throw std::runtime_error(std::string("Invalid sequence length for ") + field_name);
    }

    std::vector<int32_t> out;
    out.reserve(static_cast<std::size_t>(length));
    for (py::handle value : seq) {
        out.push_back(py::cast<int32_t>(value));
    }
    return out;
}

std::vector<std::complex<float>> complex_vector_from_sequences(
    const py::handle& re_obj,
    const py::handle& im_obj,
    const char* field_name
) {
    py::sequence re_seq = py::reinterpret_borrow<py::sequence>(re_obj);
    py::sequence im_seq = py::reinterpret_borrow<py::sequence>(im_obj);
    const py::ssize_t re_len = py::len(re_seq);
    const py::ssize_t im_len = py::len(im_seq);
    if (re_len < 0 || im_len < 0 || re_len != im_len) {
        throw std::runtime_error(std::string("Mismatched real/imag sequence lengths for ") + field_name);
    }

    std::vector<std::complex<float>> out;
    out.reserve(static_cast<std::size_t>(re_len));
    for (py::ssize_t i = 0; i < re_len; ++i) {
        out.emplace_back(
            py::cast<float>(re_seq[static_cast<std::size_t>(i)]),
            py::cast<float>(im_seq[static_cast<std::size_t>(i)])
        );
    }
    return out;
}

bool is_circuit(const py::handle& op) {
    return py::hasattr(op, "operations");
}

bool is_gate(const py::handle& op) {
    return py::hasattr(op, "targets");
}

bool is_measurement(const py::handle& op) {
    return py::hasattr(op, "qubit") && py::hasattr(op, "bit") && !py::hasattr(op, "targets");
}

bool is_conditional(const py::handle& op) {
    return py::hasattr(op, "gate") && py::hasattr(op, "condition") && !py::hasattr(op, "targets");
}

void flatten_operations(const py::handle& op, std::vector<py::object>& out) {
    if (is_circuit(op)) {
        for (py::handle child : op.attr("operations")) {
            flatten_operations(child, out);
        }
        return;
    }
    out.push_back(py::reinterpret_borrow<py::object>(op));
}

std::vector<int32_t> targets_from_gate(const py::handle& gate) {
    if (py::hasattr(gate, "_native_targets_i32")) {
        return vector_from_array<int32_t>(gate.attr("_native_targets_i32"), "_native_targets_i32");
    }
    if (py::hasattr(gate, "targets")) {
        return i32_vector_from_sequence(gate.attr("targets"), "targets");
    }
    if (py::hasattr(gate, "_canonical_targets")) {
        return i32_vector_from_sequence(gate.attr("_canonical_targets"), "_canonical_targets");
    }
    throw std::runtime_error("Gate missing targets");
}

LocalOp local_op_from_gate(const py::handle& gate) {
    LocalOp op;
    op.targets = targets_from_gate(gate);

    if (!py::hasattr(gate, "_canonical_kind")) {
        throw std::runtime_error("Gate missing canonical cache fields required by native static compiler");
    }
    const int32_t canonical_kind = py::cast<int32_t>(gate.attr("_canonical_kind"));

    if (canonical_kind == kCanonicalKindDiagonal) {
        op.kind = LocalKind::Diagonal;
        if (py::hasattr(gate, "_native_coeff_c64")) {
            op.diagonal = vector_from_array<std::complex<float>>(gate.attr("_native_coeff_c64"), "_native_coeff_c64");
        } else {
            op.diagonal = complex_vector_from_sequences(
                gate.attr("_canonical_coeff_re"),
                gate.attr("_canonical_coeff_im"),
                "_canonical_coeff"
            );
        }
        return op;
    }

    if (canonical_kind == kCanonicalKindPermutation) {
        op.kind = LocalKind::Permutation;
        if (py::hasattr(gate, "_native_perm_i32")) {
            op.permutation = vector_from_array<int32_t>(gate.attr("_native_perm_i32"), "_native_perm_i32");
        } else {
            op.permutation = i32_vector_from_sequence(gate.attr("_canonical_perm"), "_canonical_perm");
        }
        if (py::hasattr(gate, "_native_aux_c64")) {
            op.phase = vector_from_array<std::complex<float>>(gate.attr("_native_aux_c64"), "_native_aux_c64");
        } else {
            op.phase = complex_vector_from_sequences(
                gate.attr("_canonical_aux_re"),
                gate.attr("_canonical_aux_im"),
                "_canonical_aux"
            );
        }
        if (!op.phase.empty() && op.phase.size() != op.permutation.size()) {
            throw std::runtime_error("Permutation phase length must match permutation length");
        }
        return op;
    }

    if (canonical_kind == kCanonicalKindDense) {
        op.kind = LocalKind::Dense;
        if (py::hasattr(gate, "_native_coeff_c64")) {
            op.dense = vector_from_array<std::complex<float>>(gate.attr("_native_coeff_c64"), "_native_coeff_c64");
        } else {
            op.dense = complex_vector_from_sequences(
                gate.attr("_canonical_coeff_re"),
                gate.attr("_canonical_coeff_im"),
                "_canonical_coeff"
            );
        }
        return op;
    }

    throw std::runtime_error("Unknown canonical gate kind in native static compiler");
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

std::int64_t compile_static_circuit(
    py::object circuit,
    int n_qubits,
    int n_bits
) {
    std::vector<py::object> linear_ops;
    linear_ops.reserve(512);
    flatten_operations(circuit, linear_ops);

    std::size_t terminal_start = linear_ops.size();
    while (terminal_start > 0 && is_measurement(linear_ops[terminal_start - 1])) {
        terminal_start -= 1;
    }

    for (std::size_t i = 0; i < terminal_start; ++i) {
        const py::handle op = linear_ops[i];
        if (is_conditional(op) || is_measurement(op) || !is_gate(op)) {
            throw std::runtime_error(
                "Dynamic circuits are temporarily unsupported in static-only Metal build."
            );
        }
    }

    for (std::size_t i = terminal_start; i < linear_ops.size(); ++i) {
        if (!is_measurement(linear_ops[i])) {
            throw std::runtime_error(
                "Dynamic circuits are temporarily unsupported in static-only Metal build."
            );
        }
    }

    std::vector<LocalOp> ops;
    ops.reserve(terminal_start);
    for (std::size_t i = 0; i < terminal_start; ++i) {
        ops.push_back(local_op_from_gate(linear_ops[i]));
    }

    std::vector<int32_t> terminal_measurements;
    terminal_measurements.reserve((linear_ops.size() - terminal_start) * 2);
    for (std::size_t i = terminal_start; i < linear_ops.size(); ++i) {
        const py::handle meas = linear_ops[i];
        terminal_measurements.push_back(py::cast<int32_t>(meas.attr("qubit")));
        terminal_measurements.push_back(py::cast<int32_t>(meas.attr("bit")));
    }

    return compile_local_ops_to_program(
        std::move(ops),
        std::move(terminal_measurements),
        n_qubits,
        n_bits
    );
}

std::int64_t compile_static_canonical(
    int n_qubits,
    int n_bits,
    const py::object& op_kinds_obj,
    const py::object& op_target_offsets_obj,
    const py::object& op_target_lens_obj,
    const py::object& op_coeff_offsets_obj,
    const py::object& op_coeff_lens_obj,
    const py::object& op_aux_offsets_obj,
    const py::object& op_aux_lens_obj,
    const py::object& target_pool_obj,
    const py::object& diag_re_obj,
    const py::object& diag_im_obj,
    const py::object& perm_pool_obj,
    const py::object& phase_re_obj,
    const py::object& phase_im_obj,
    const py::object& dense_re_obj,
    const py::object& dense_im_obj,
    const py::object& terminal_measurements_obj
) {
    const std::vector<int32_t> op_kinds = vector_from_array<int32_t>(op_kinds_obj, "op_kinds");
    const std::vector<int32_t> op_target_offsets = vector_from_array<int32_t>(op_target_offsets_obj, "op_target_offsets");
    const std::vector<int32_t> op_target_lens = vector_from_array<int32_t>(op_target_lens_obj, "op_target_lens");
    const std::vector<int32_t> op_coeff_offsets = vector_from_array<int32_t>(op_coeff_offsets_obj, "op_coeff_offsets");
    const std::vector<int32_t> op_coeff_lens = vector_from_array<int32_t>(op_coeff_lens_obj, "op_coeff_lens");
    const std::vector<int32_t> op_aux_offsets = vector_from_array<int32_t>(op_aux_offsets_obj, "op_aux_offsets");
    const std::vector<int32_t> op_aux_lens = vector_from_array<int32_t>(op_aux_lens_obj, "op_aux_lens");

    const std::vector<int32_t> target_pool = vector_from_array<int32_t>(target_pool_obj, "target_pool");
    const std::vector<float> diag_re = vector_from_array<float>(diag_re_obj, "diag_re");
    const std::vector<float> diag_im = vector_from_array<float>(diag_im_obj, "diag_im");
    const std::vector<int32_t> perm_pool = vector_from_array<int32_t>(perm_pool_obj, "perm_pool");
    const std::vector<float> phase_re = vector_from_array<float>(phase_re_obj, "phase_re");
    const std::vector<float> phase_im = vector_from_array<float>(phase_im_obj, "phase_im");
    const std::vector<float> dense_re = vector_from_array<float>(dense_re_obj, "dense_re");
    const std::vector<float> dense_im = vector_from_array<float>(dense_im_obj, "dense_im");
    std::vector<int32_t> terminal_measurements =
        vector_from_array<int32_t>(terminal_measurements_obj, "terminal_measurements");

    const std::size_t op_count = op_kinds.size();
    if (op_target_offsets.size() != op_count
        || op_target_lens.size() != op_count
        || op_coeff_offsets.size() != op_count
        || op_coeff_lens.size() != op_count
        || op_aux_offsets.size() != op_count
        || op_aux_lens.size() != op_count) {
        throw std::runtime_error("Canonical op metadata arrays must have equal length");
    }
    if (diag_re.size() != diag_im.size()) {
        throw std::runtime_error("diag_re/diag_im length mismatch");
    }
    if (phase_re.size() != phase_im.size()) {
        throw std::runtime_error("phase_re/phase_im length mismatch");
    }
    if (dense_re.size() != dense_im.size()) {
        throw std::runtime_error("dense_re/dense_im length mismatch");
    }
    if ((terminal_measurements.size() % 2) != 0) {
        throw std::runtime_error("terminal_measurements must be flattened (qubit, bit) pairs");
    }

    auto validate_slice = [](int32_t offset, int32_t length, std::size_t total, const char* name) {
        if (offset < 0 || length < 0) {
            throw std::runtime_error(std::string(name) + " has negative offset/length");
        }
        const std::size_t end = static_cast<std::size_t>(offset) + static_cast<std::size_t>(length);
        if (end > total) {
            throw std::runtime_error(std::string(name) + " slice out of bounds");
        }
    };

    std::vector<LocalOp> ops;
    ops.reserve(op_count);
    for (std::size_t i = 0; i < op_count; ++i) {
        const int32_t kind = op_kinds[i];
        const int32_t target_offset = op_target_offsets[i];
        const int32_t target_len = op_target_lens[i];
        validate_slice(target_offset, target_len, target_pool.size(), "target_pool");

        LocalOp op;
        op.targets.assign(
            target_pool.begin() + target_offset,
            target_pool.begin() + target_offset + target_len
        );

        const int32_t coeff_offset = op_coeff_offsets[i];
        const int32_t coeff_len = op_coeff_lens[i];
        const int32_t aux_offset = op_aux_offsets[i];
        const int32_t aux_len = op_aux_lens[i];

        if (kind == kCanonicalKindDiagonal) {
            validate_slice(coeff_offset, coeff_len, diag_re.size(), "diag");
            op.kind = LocalKind::Diagonal;
            op.diagonal.reserve(static_cast<std::size_t>(coeff_len));
            for (int32_t j = 0; j < coeff_len; ++j) {
                const std::size_t idx = static_cast<std::size_t>(coeff_offset + j);
                op.diagonal.emplace_back(diag_re[idx], diag_im[idx]);
            }
            ops.push_back(std::move(op));
            continue;
        }

        if (kind == kCanonicalKindPermutation) {
            validate_slice(coeff_offset, coeff_len, perm_pool.size(), "perm_pool");
            op.kind = LocalKind::Permutation;
            op.permutation.assign(
                perm_pool.begin() + coeff_offset,
                perm_pool.begin() + coeff_offset + coeff_len
            );

            if (aux_len > 0) {
                if (aux_offset < 0) {
                    throw std::runtime_error("Permutation phase slice has negative offset");
                }
                validate_slice(aux_offset, aux_len, phase_re.size(), "phase");
                op.phase.reserve(static_cast<std::size_t>(aux_len));
                for (int32_t j = 0; j < aux_len; ++j) {
                    const std::size_t idx = static_cast<std::size_t>(aux_offset + j);
                    op.phase.emplace_back(phase_re[idx], phase_im[idx]);
                }
            }
            ops.push_back(std::move(op));
            continue;
        }

        if (kind == kCanonicalKindDense) {
            validate_slice(coeff_offset, coeff_len, dense_re.size(), "dense");
            op.kind = LocalKind::Dense;
            op.dense.reserve(static_cast<std::size_t>(coeff_len));
            for (int32_t j = 0; j < coeff_len; ++j) {
                const std::size_t idx = static_cast<std::size_t>(coeff_offset + j);
                op.dense.emplace_back(dense_re[idx], dense_im[idx]);
            }
            ops.push_back(std::move(op));
            continue;
        }

        throw std::runtime_error("Unknown canonical op kind");
    }

    return compile_local_ops_to_program(
        std::move(ops),
        std::move(terminal_measurements),
        n_qubits,
        n_bits
    );
}

std::int64_t compile_static_packed(py::bytes payload) {
    const std::string blob = static_cast<std::string>(payload);
    std::size_t cursor = 0;

    const uint32_t magic = read_scalar<uint32_t>(blob, cursor, "magic");
    if (magic != kPackedStaticAbiMagic) {
        throw std::runtime_error("Unsupported packed static circuit payload (magic mismatch)");
    }

    const uint32_t version = read_scalar<uint32_t>(blob, cursor, "version");
    if (version != kPackedStaticAbiVersion) {
        throw std::runtime_error("Unsupported packed static circuit payload version");
    }

    const int32_t n_qubits = read_scalar<int32_t>(blob, cursor, "n_qubits");
    const int32_t n_bits = read_scalar<int32_t>(blob, cursor, "n_bits");
    const uint32_t op_count = read_scalar<uint32_t>(blob, cursor, "op_count");
    const uint32_t terminal_pair_count = read_scalar<uint32_t>(blob, cursor, "terminal_pair_count");

    std::vector<LocalOp> ops;
    ops.reserve(static_cast<std::size_t>(op_count));
    for (uint32_t op_idx = 0; op_idx < op_count; ++op_idx) {
        const int32_t kind = read_scalar<int32_t>(blob, cursor, "op.kind");
        const int32_t target_len_i32 = read_scalar<int32_t>(blob, cursor, "op.target_len");
        const int32_t coeff_len_i32 = read_scalar<int32_t>(blob, cursor, "op.coeff_len");
        const int32_t aux_len_i32 = read_scalar<int32_t>(blob, cursor, "op.aux_len");

        if (target_len_i32 < 0 || coeff_len_i32 < 0 || aux_len_i32 < 0) {
            throw std::runtime_error("Packed op metadata contains negative lengths");
        }

        const std::size_t target_len = static_cast<std::size_t>(target_len_i32);
        const std::size_t coeff_len = static_cast<std::size_t>(coeff_len_i32);
        const std::size_t aux_len = static_cast<std::size_t>(aux_len_i32);

        LocalOp op;
        op.targets = read_array<int32_t>(blob, cursor, target_len, "op.targets");

        if (kind == kCanonicalKindDiagonal) {
            if (aux_len != 0) {
                throw std::runtime_error("Packed diagonal op cannot carry aux payload");
            }
            op.kind = LocalKind::Diagonal;
            op.diagonal = read_complex_array(blob, cursor, coeff_len, "op.diagonal_coeff");
            ops.push_back(std::move(op));
            continue;
        }

        if (kind == kCanonicalKindPermutation) {
            op.kind = LocalKind::Permutation;
            op.permutation = read_array<int32_t>(blob, cursor, coeff_len, "op.permutation");
            op.phase = read_complex_array(blob, cursor, aux_len, "op.permutation_phase");
            if (!op.phase.empty() && op.phase.size() != op.permutation.size()) {
                throw std::runtime_error("Permutation phase length must match permutation length");
            }
            ops.push_back(std::move(op));
            continue;
        }

        if (kind == kCanonicalKindDense) {
            if (aux_len != 0) {
                throw std::runtime_error("Packed dense op cannot carry aux payload");
            }
            op.kind = LocalKind::Dense;
            op.dense = read_complex_array(blob, cursor, coeff_len, "op.dense_coeff");
            ops.push_back(std::move(op));
            continue;
        }

        throw std::runtime_error("Unknown packed op kind");
    }

    std::vector<int32_t> terminal_measurements;
    terminal_measurements.reserve(static_cast<std::size_t>(terminal_pair_count) * 2);
    for (uint32_t i = 0; i < terminal_pair_count; ++i) {
        terminal_measurements.push_back(read_scalar<int32_t>(blob, cursor, "terminal.qubit"));
        terminal_measurements.push_back(read_scalar<int32_t>(blob, cursor, "terminal.bit"));
    }

    if (cursor != blob.size()) {
        throw std::runtime_error("Malformed packed static circuit: trailing bytes");
    }

    return compile_local_ops_to_program(
        std::move(ops),
        std::move(terminal_measurements),
        n_qubits,
        n_bits
    );
}

std::unordered_map<std::string, std::int64_t> run_static_circuit(
    py::object circuit,
    int n_qubits,
    int n_bits,
    std::int64_t num_shots,
    py::object seed_obj
) {
    std::optional<std::uint64_t> seed = std::nullopt;
    if (!seed_obj.is_none()) {
        seed = seed_obj.cast<std::uint64_t>();
    }

    const std::int64_t handle = compile_static_circuit(
        std::move(circuit),
        n_qubits,
        n_bits
    );
    try {
        auto counts = quantum_native::execute_static_program(handle, num_shots, seed);
        quantum_native::free_program(handle);
        return counts;
    } catch (...) {
        try {
            quantum_native::free_program(handle);
        } catch (...) {
            // best-effort cleanup
        }
        throw;
    }
}

std::unordered_map<std::string, std::int64_t> run_static_packed(
    py::bytes payload,
    std::int64_t num_shots,
    py::object seed_obj
) {
    std::optional<std::uint64_t> seed = std::nullopt;
    if (!seed_obj.is_none()) {
        seed = seed_obj.cast<std::uint64_t>();
    }

    const std::int64_t handle = compile_static_packed(std::move(payload));
    try {
        auto counts = quantum_native::execute_static_program(handle, num_shots, seed);
        quantum_native::free_program(handle);
        return counts;
    } catch (...) {
        try {
            quantum_native::free_program(handle);
        } catch (...) {
            // best-effort cleanup
        }
        throw;
    }
}

}  // namespace

PYBIND11_MODULE(quantum_native_runtime, m) {
    m.doc() = "Native Metal runtime for quantum static execution";

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

    m.def(
        "compile_static_program",
        [](int n_qubits,
           int n_bits,
           const py::object& op_table,
           const py::object& dispatch_table,
           const py::object& target_pool,
           const py::object& diag_re,
           const py::object& diag_im,
           const py::object& perm_pool,
           const py::object& phase_re,
           const py::object& phase_im,
           const py::object& dense_re,
           const py::object& dense_im,
           py::bytes monomial_blob,
           const py::object& terminal_measurements) {
            quantum_native::StaticProgramData data{};
            data.n_qubits = n_qubits;
            data.n_bits = n_bits;
            data.op_table = vector_from_array<std::int32_t>(op_table, "op_table");
            data.dispatch_table = vector_from_array<std::int32_t>(dispatch_table, "dispatch_table");
            data.target_pool = vector_from_array<std::int32_t>(target_pool, "target_pool");
            data.diag_pool_re = vector_from_array<float>(diag_re, "diag_re");
            data.diag_pool_im = vector_from_array<float>(diag_im, "diag_im");
            data.perm_pool = vector_from_array<std::int32_t>(perm_pool, "perm_pool");
            data.phase_pool_re = vector_from_array<float>(phase_re, "phase_re");
            data.phase_pool_im = vector_from_array<float>(phase_im, "phase_im");
            data.dense_pool_re = vector_from_array<float>(dense_re, "dense_re");
            data.dense_pool_im = vector_from_array<float>(dense_im, "dense_im");
            data.terminal_measurements = vector_from_array<std::int32_t>(terminal_measurements, "terminal_measurements");

            std::string blob = monomial_blob;
            data.monomial_blob.assign(blob.begin(), blob.end());

            return quantum_native::compile_static_program(data);
        },
        py::arg("n_qubits"),
        py::arg("n_bits"),
        py::arg("op_table"),
        py::arg("dispatch_table"),
        py::arg("target_pool"),
        py::arg("diag_re"),
        py::arg("diag_im"),
        py::arg("perm_pool"),
        py::arg("phase_re"),
        py::arg("phase_im"),
        py::arg("dense_re"),
        py::arg("dense_im"),
        py::arg("monomial_blob"),
        py::arg("terminal_measurements")
    );

    m.def(
        "compile_static_circuit",
        &compile_static_circuit,
        py::arg("circuit"),
        py::arg("n_qubits"),
        py::arg("n_bits")
    );

    m.def(
        "run_static_circuit",
        &run_static_circuit,
        py::arg("circuit"),
        py::arg("n_qubits"),
        py::arg("n_bits"),
        py::arg("num_shots"),
        py::arg("seed") = py::none()
    );

    m.def(
        "compile_static_canonical",
        &compile_static_canonical,
        py::arg("n_qubits"),
        py::arg("n_bits"),
        py::arg("op_kinds"),
        py::arg("op_target_offsets"),
        py::arg("op_target_lens"),
        py::arg("op_coeff_offsets"),
        py::arg("op_coeff_lens"),
        py::arg("op_aux_offsets"),
        py::arg("op_aux_lens"),
        py::arg("target_pool"),
        py::arg("diag_re"),
        py::arg("diag_im"),
        py::arg("perm_pool"),
        py::arg("phase_re"),
        py::arg("phase_im"),
        py::arg("dense_re"),
        py::arg("dense_im"),
        py::arg("terminal_measurements")
    );

    m.def(
        "compile_static_packed",
        &compile_static_packed,
        py::arg("payload")
    );

    m.def(
        "run_static_packed",
        &run_static_packed,
        py::arg("payload"),
        py::arg("num_shots"),
        py::arg("seed") = py::none()
    );

    m.def(
        "execute_static_program",
        [](std::int64_t handle, std::int64_t num_shots, py::object seed_obj) {
            std::optional<std::uint64_t> seed = std::nullopt;
            if (!seed_obj.is_none()) {
                seed = seed_obj.cast<std::uint64_t>();
            }
            return quantum_native::execute_static_program(handle, num_shots, seed);
        },
        py::arg("handle"),
        py::arg("num_shots"),
        py::arg("seed") = py::none()
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
