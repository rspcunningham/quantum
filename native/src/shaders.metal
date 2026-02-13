#include <metal_stdlib>
using namespace metal;

struct OpParams {
    uint n_qubits;
    uint dim;
    int target_offset;
    int target_len;
    int coeff_offset;
    int coeff_len;
    int aux0;
    int aux1;
};

struct MonomialParams {
    uint n_qubits;
    uint dim;
    uint gate_count;
};

struct DispatchGroupParams {
    uint n_qubits;
    uint dim;
    int op_start;
    int op_count;
};

struct SamplingProbParams {
    uint dim;
};

struct SamplingScanParams {
    uint dim;
    uint offset;
};

struct SamplingNormalizeParams {
    uint dim;
};

struct SamplingDrawParams {
    uint dim;
    uint n_qubits;
    uint n_bits;
    uint terminal_pairs;
    uint num_shots;
    uint identity_measure;
    uint _pad0;
    uint _pad1;
    ulong seed;
};

struct HistogramParams {
    uint num_shots;
    uint table_size;
};

constant int OPCODE_DIAG_SUBSET = 1;
constant int OPCODE_DIAG_FULL = 2;
constant int OPCODE_PERM_SUBSET = 3;
constant int OPCODE_PERM_FULL = 4;

inline int op_field(const device int* op_table, int op_index, int field) {
    return op_table[op_index * 8 + field];
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline ulong splitmix64(ulong x) {
    x += 0x9E3779B97F4A7C15ul;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ul;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBul;
    return x ^ (x >> 31);
}

inline float uniform01_from_u64(ulong x) {
    return float((x >> 40) & 0xFFFFFFul) * (1.0f / 16777216.0f);
}

inline uint subindex_for_targets(
    uint idx,
    const device int* targets,
    int target_offset,
    uint k,
    uint n_qubits
) {
    uint sub = 0;
    for (uint j = 0; j < k; ++j) {
        uint bitpos = n_qubits - 1u - uint(targets[target_offset + int(j)]);
        uint bit = (idx >> bitpos) & 1u;
        sub |= bit << (k - 1u - j);
    }
    return sub;
}

inline uint source_index_for_targets(
    uint idx,
    uint source_subindex,
    const device int* targets,
    int target_offset,
    uint k,
    uint n_qubits
) {
    uint src = idx;
    for (uint j = 0; j < k; ++j) {
        uint bitpos = n_qubits - 1u - uint(targets[target_offset + int(j)]);
        uint mask = 1u << bitpos;
        src &= ~mask;
        uint bit = (source_subindex >> (k - 1u - j)) & 1u;
        src |= bit << bitpos;
    }
    return src;
}

kernel void diag_full(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const float* diag_re,
    device const float* diag_im,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint coeff_idx = uint(params.coeff_offset) + gid;
    float vr = in_re[gid];
    float vi = in_im[gid];
    float dr = diag_re[coeff_idx];
    float di = diag_im[coeff_idx];

    out_re[gid] = vr * dr - vi * di;
    out_im[gid] = vr * di + vi * dr;
}

kernel void diag_subset(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* target_pool,
    device const float* diag_re,
    device const float* diag_im,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint k = uint(params.target_len);
    uint sub = subindex_for_targets(gid, target_pool, params.target_offset, k, params.n_qubits);
    uint coeff_idx = uint(params.coeff_offset) + sub;

    float vr = in_re[gid];
    float vi = in_im[gid];
    float dr = diag_re[coeff_idx];
    float di = diag_im[coeff_idx];

    out_re[gid] = vr * dr - vi * di;
    out_im[gid] = vr * di + vi * dr;
}

kernel void perm_full(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* perm_pool,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint src = uint(perm_pool[params.coeff_offset + int(gid)]);
    out_re[gid] = in_re[src];
    out_im[gid] = in_im[src];
}

kernel void perm_full_phase(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* perm_pool,
    device const float* phase_re,
    device const float* phase_im,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint src = uint(perm_pool[params.coeff_offset + int(gid)]);
    float vr = in_re[src];
    float vi = in_im[src];
    float pr = phase_re[params.aux0 + int(gid)];
    float pi = phase_im[params.aux0 + int(gid)];

    out_re[gid] = vr * pr - vi * pi;
    out_im[gid] = vr * pi + vi * pr;
}

kernel void perm_subset(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* target_pool,
    device const int* perm_pool,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint k = uint(params.target_len);
    uint sub = subindex_for_targets(gid, target_pool, params.target_offset, k, params.n_qubits);
    uint mapped_sub = uint(perm_pool[params.coeff_offset + int(sub)]);
    uint src = source_index_for_targets(gid, mapped_sub, target_pool, params.target_offset, k, params.n_qubits);

    out_re[gid] = in_re[src];
    out_im[gid] = in_im[src];
}

kernel void perm_subset_phase(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* target_pool,
    device const int* perm_pool,
    device const float* phase_re,
    device const float* phase_im,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint k = uint(params.target_len);
    uint sub = subindex_for_targets(gid, target_pool, params.target_offset, k, params.n_qubits);
    uint mapped_sub = uint(perm_pool[params.coeff_offset + int(sub)]);
    uint src = source_index_for_targets(gid, mapped_sub, target_pool, params.target_offset, k, params.n_qubits);

    float vr = in_re[src];
    float vi = in_im[src];
    float pr = phase_re[params.aux0 + int(sub)];
    float pi = phase_im[params.aux0 + int(sub)];

    out_re[gid] = vr * pr - vi * pi;
    out_im[gid] = vr * pi + vi * pr;
}

kernel void diag_full_group(
    device const float* in_re [[buffer(0)]],
    device const float* in_im [[buffer(1)]],
    device float* out_re [[buffer(2)]],
    device float* out_im [[buffer(3)]],
    device const int* op_table [[buffer(4)]],
    device const int* target_pool [[buffer(5)]],
    device const float* diag_re [[buffer(6)]],
    device const float* diag_im [[buffer(7)]],
    device const int* perm_pool [[buffer(8)]],
    device const float* phase_re [[buffer(9)]],
    device const float* phase_im [[buffer(10)]],
    constant DispatchGroupParams& params [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    (void)target_pool;
    (void)perm_pool;
    (void)phase_re;
    (void)phase_im;
    if (gid >= params.dim) {
        return;
    }

    float vr = in_re[gid];
    float vi = in_im[gid];
    for (int i = 0; i < params.op_count; ++i) {
        int op_index = params.op_start + i;
        if (op_field(op_table, op_index, 0) != OPCODE_DIAG_FULL) {
            return;
        }
        int coeff_offset = op_field(op_table, op_index, 3);
        uint coeff_idx = uint(coeff_offset) + gid;
        float dr = diag_re[coeff_idx];
        float di = diag_im[coeff_idx];
        float next_r = vr * dr - vi * di;
        float next_i = vr * di + vi * dr;
        vr = next_r;
        vi = next_i;
    }

    out_re[gid] = vr;
    out_im[gid] = vi;
}

kernel void diag_subset_group(
    device const float* in_re [[buffer(0)]],
    device const float* in_im [[buffer(1)]],
    device float* out_re [[buffer(2)]],
    device float* out_im [[buffer(3)]],
    device const int* op_table [[buffer(4)]],
    device const int* target_pool [[buffer(5)]],
    device const float* diag_re [[buffer(6)]],
    device const float* diag_im [[buffer(7)]],
    device const int* perm_pool [[buffer(8)]],
    device const float* phase_re [[buffer(9)]],
    device const float* phase_im [[buffer(10)]],
    constant DispatchGroupParams& params [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    (void)perm_pool;
    (void)phase_re;
    (void)phase_im;
    if (gid >= params.dim) {
        return;
    }

    float vr = in_re[gid];
    float vi = in_im[gid];
    for (int i = 0; i < params.op_count; ++i) {
        int op_index = params.op_start + i;
        if (op_field(op_table, op_index, 0) != OPCODE_DIAG_SUBSET) {
            return;
        }
        int target_offset = op_field(op_table, op_index, 1);
        uint k = uint(op_field(op_table, op_index, 2));
        int coeff_offset = op_field(op_table, op_index, 3);
        uint sub = subindex_for_targets(gid, target_pool, target_offset, k, params.n_qubits);
        uint coeff_idx = uint(coeff_offset) + sub;
        float dr = diag_re[coeff_idx];
        float di = diag_im[coeff_idx];
        float next_r = vr * dr - vi * di;
        float next_i = vr * di + vi * dr;
        vr = next_r;
        vi = next_i;
    }

    out_re[gid] = vr;
    out_im[gid] = vi;
}

kernel void perm_full_group(
    device const float* in_re [[buffer(0)]],
    device const float* in_im [[buffer(1)]],
    device float* out_re [[buffer(2)]],
    device float* out_im [[buffer(3)]],
    device const int* op_table [[buffer(4)]],
    device const int* target_pool [[buffer(5)]],
    device const float* diag_re [[buffer(6)]],
    device const float* diag_im [[buffer(7)]],
    device const int* perm_pool [[buffer(8)]],
    device const float* phase_re [[buffer(9)]],
    device const float* phase_im [[buffer(10)]],
    constant DispatchGroupParams& params [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    (void)target_pool;
    (void)diag_re;
    (void)diag_im;
    if (gid >= params.dim) {
        return;
    }

    uint src = gid;
    float2 phase = float2(1.0, 0.0);
    for (int i = params.op_count - 1; i >= 0; --i) {
        int op_index = params.op_start + i;
        if (op_field(op_table, op_index, 0) != OPCODE_PERM_FULL) {
            return;
        }
        int coeff_offset = op_field(op_table, op_index, 3);
        int aux0 = op_field(op_table, op_index, 6);
        int aux1 = op_field(op_table, op_index, 7);

        if (aux0 >= 0 && aux1 > 0) {
            float pr = phase_re[aux0 + int(src)];
            float pi = phase_im[aux0 + int(src)];
            phase = cmul(phase, float2(pr, pi));
        }
        src = uint(perm_pool[coeff_offset + int(src)]);
    }

    float2 v = float2(in_re[src], in_im[src]);
    float2 out = cmul(phase, v);
    out_re[gid] = out.x;
    out_im[gid] = out.y;
}

kernel void perm_subset_group(
    device const float* in_re [[buffer(0)]],
    device const float* in_im [[buffer(1)]],
    device float* out_re [[buffer(2)]],
    device float* out_im [[buffer(3)]],
    device const int* op_table [[buffer(4)]],
    device const int* target_pool [[buffer(5)]],
    device const float* diag_re [[buffer(6)]],
    device const float* diag_im [[buffer(7)]],
    device const int* perm_pool [[buffer(8)]],
    device const float* phase_re [[buffer(9)]],
    device const float* phase_im [[buffer(10)]],
    constant DispatchGroupParams& params [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    (void)diag_re;
    (void)diag_im;
    if (gid >= params.dim) {
        return;
    }

    uint src = gid;
    float2 phase = float2(1.0, 0.0);
    for (int i = params.op_count - 1; i >= 0; --i) {
        int op_index = params.op_start + i;
        if (op_field(op_table, op_index, 0) != OPCODE_PERM_SUBSET) {
            return;
        }

        int target_offset = op_field(op_table, op_index, 1);
        uint k = uint(op_field(op_table, op_index, 2));
        int coeff_offset = op_field(op_table, op_index, 3);
        int aux0 = op_field(op_table, op_index, 6);
        int aux1 = op_field(op_table, op_index, 7);

        uint sub = subindex_for_targets(src, target_pool, target_offset, k, params.n_qubits);
        uint mapped_sub = uint(perm_pool[coeff_offset + int(sub)]);
        if (aux0 >= 0 && aux1 > 0) {
            float pr = phase_re[aux0 + int(sub)];
            float pi = phase_im[aux0 + int(sub)];
            phase = cmul(phase, float2(pr, pi));
        }
        src = source_index_for_targets(src, mapped_sub, target_pool, target_offset, k, params.n_qubits);
    }

    float2 v = float2(in_re[src], in_im[src]);
    float2 out = cmul(phase, v);
    out_re[gid] = out.x;
    out_im[gid] = out.y;
}

kernel void dense1(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* target_pool,
    device const float* dense_re,
    device const float* dense_im,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint bitpos = params.n_qubits - 1u - uint(target_pool[params.target_offset]);
    uint mask = 1u << bitpos;
    uint idx0 = gid & ~mask;
    uint idx1 = idx0 | mask;
    uint row = (gid >> bitpos) & 1u;

    uint cbase = uint(params.coeff_offset) + row * 2u;
    float2 c0 = float2(dense_re[cbase + 0u], dense_im[cbase + 0u]);
    float2 c1 = float2(dense_re[cbase + 1u], dense_im[cbase + 1u]);

    float2 v0 = float2(in_re[idx0], in_im[idx0]);
    float2 v1 = float2(in_re[idx1], in_im[idx1]);
    float2 out = cmul(c0, v0) + cmul(c1, v1);

    out_re[gid] = out.x;
    out_im[gid] = out.y;
}

kernel void dense2(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* target_pool,
    device const float* dense_re,
    device const float* dense_im,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint bitpos0 = params.n_qubits - 1u - uint(target_pool[params.target_offset + 0]);
    uint bitpos1 = params.n_qubits - 1u - uint(target_pool[params.target_offset + 1]);
    uint mask0 = 1u << bitpos0;
    uint mask1 = 1u << bitpos1;

    uint idx00 = gid & ~mask0 & ~mask1;
    uint idx01 = idx00 | mask1;
    uint idx10 = idx00 | mask0;
    uint idx11 = idx00 | mask0 | mask1;

    uint b0 = (gid >> bitpos0) & 1u;
    uint b1 = (gid >> bitpos1) & 1u;
    uint row = (b0 << 1u) | b1;

    uint cbase = uint(params.coeff_offset) + row * 4u;
    float2 c0 = float2(dense_re[cbase + 0u], dense_im[cbase + 0u]);
    float2 c1 = float2(dense_re[cbase + 1u], dense_im[cbase + 1u]);
    float2 c2 = float2(dense_re[cbase + 2u], dense_im[cbase + 2u]);
    float2 c3 = float2(dense_re[cbase + 3u], dense_im[cbase + 3u]);

    float2 v00 = float2(in_re[idx00], in_im[idx00]);
    float2 v01 = float2(in_re[idx01], in_im[idx01]);
    float2 v10 = float2(in_re[idx10], in_im[idx10]);
    float2 v11 = float2(in_re[idx11], in_im[idx11]);

    float2 out = cmul(c0, v00) + cmul(c1, v01) + cmul(c2, v10) + cmul(c3, v11);
    out_re[gid] = out.x;
    out_im[gid] = out.y;
}

kernel void densek(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* target_pool,
    device const float* dense_re,
    device const float* dense_im,
    constant OpParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint k = uint(params.target_len);
    if (k == 0u || k > 6u) {
        // Invalid arity should be rejected at compile-time; emit poison if violated.
        out_re[gid] = NAN;
        out_im[gid] = NAN;
        return;
    }

    uint dense_dim = 1u << k;
    uint row = subindex_for_targets(gid, target_pool, params.target_offset, k, params.n_qubits);

    float2 acc = float2(0.0, 0.0);
    for (uint col = 0u; col < dense_dim; ++col) {
        uint src = source_index_for_targets(gid, col, target_pool, params.target_offset, k, params.n_qubits);
        uint coeff_idx = uint(params.coeff_offset) + row * dense_dim + col;
        float2 coeff = float2(dense_re[coeff_idx], dense_im[coeff_idx]);
        float2 v = float2(in_re[src], in_im[src]);
        acc += cmul(coeff, v);
    }

    out_re[gid] = acc.x;
    out_im[gid] = acc.y;
}

kernel void monomial_stream(
    device const float* in_re,
    device const float* in_im,
    device float* out_re,
    device float* out_im,
    device const int* gate_ks,
    device const int* gate_targets,
    device const int* gate_permutations,
    device const float* gate_factors_re,
    device const float* gate_factors_im,
    constant MonomialParams& params,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }

    uint src = gid;
    float2 phase = float2(1.0, 0.0);

    for (int gi = int(params.gate_count) - 1; gi >= 0; --gi) {
        uint k = uint(gate_ks[gi]);
        uint target_offset = uint(gi) * 2u;
        uint sub = subindex_for_targets(src, gate_targets, int(target_offset), k, params.n_qubits);
        uint mapped_sub = uint(gate_permutations[uint(gi) * 4u + sub]);
        float2 factor = float2(gate_factors_re[uint(gi) * 4u + sub], gate_factors_im[uint(gi) * 4u + sub]);
        phase = cmul(phase, factor);
        src = source_index_for_targets(src, mapped_sub, gate_targets, int(target_offset), k, params.n_qubits);
    }

    float2 v = float2(in_re[src], in_im[src]);
    float2 out = cmul(phase, v);
    out_re[gid] = out.x;
    out_im[gid] = out.y;
}

kernel void compute_probabilities(
    device const float* state_re [[buffer(0)]],
    device const float* state_im [[buffer(1)]],
    device float* probs [[buffer(2)]],
    constant SamplingProbParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }
    float re = state_re[gid];
    float im = state_im[gid];
    probs[gid] = re * re + im * im;
}

kernel void inclusive_scan_step(
    device const float* in_values [[buffer(0)]],
    device float* out_values [[buffer(1)]],
    constant SamplingScanParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }
    float value = in_values[gid];
    if (gid >= params.offset) {
        value += in_values[gid - params.offset];
    }
    out_values[gid] = value;
}

kernel void normalize_cdf(
    device float* cdf [[buffer(0)]],
    constant SamplingNormalizeParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.dim) {
        return;
    }
    float total = cdf[params.dim - 1u];
    if (!(total > 0.0f)) {
        cdf[gid] = (gid == params.dim - 1u) ? 1.0f : 0.0f;
        return;
    }
    float value = cdf[gid] / total;
    if (gid == params.dim - 1u) {
        value = 1.0f;
    }
    cdf[gid] = value;
}

kernel void sample_indices_to_codes(
    device const float* cdf [[buffer(0)]],
    device const int* terminal_measurements [[buffer(1)]],
    device uint* out_codes [[buffer(2)]],
    constant SamplingDrawParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_shots) {
        return;
    }

    ulong rng = splitmix64(params.seed ^ ulong(gid + 1u));
    float u = uniform01_from_u64(rng);

    uint lo = 0u;
    uint hi = params.dim - 1u;
    while (lo < hi) {
        uint mid = lo + ((hi - lo) >> 1);
        if (cdf[mid] < u) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    uint basis = lo;
    if (basis >= params.dim) {
        basis = params.dim - 1u;
    }

    uint code = 0u;
    if (params.terminal_pairs == 0u) {
        code = 0u;
    } else if (params.identity_measure != 0u) {
        code = basis;
    } else {
        for (uint t = 0u; t < params.terminal_pairs; ++t) {
            int q = terminal_measurements[t * 2u + 0u];
            int b = terminal_measurements[t * 2u + 1u];
            uint qubit_shift = params.n_qubits - 1u - uint(q);
            uint bit_shift = params.n_bits - 1u - uint(b);
            uint measured = (basis >> qubit_shift) & 1u;
            code |= measured << bit_shift;
        }
    }
    out_codes[gid] = code;
}

kernel void histogram_codes(
    device const uint* sampled_codes [[buffer(0)]],
    device atomic_uint* hist_keys [[buffer(1)]],
    device atomic_uint* hist_counts [[buffer(2)]],
    constant HistogramParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_shots) {
        return;
    }

    constexpr uint EMPTY_KEY = 0xFFFFFFFFu;
    uint code = sampled_codes[gid];
    uint mask = params.table_size - 1u;
    uint idx = (code * 2654435761u) & mask;

    for (uint probe = 0u; probe < params.table_size; ++probe) {
        uint expected = EMPTY_KEY;
        if (atomic_compare_exchange_weak_explicit(
                &hist_keys[idx], &expected, code, memory_order_relaxed, memory_order_relaxed)) {
            atomic_fetch_add_explicit(&hist_counts[idx], 1u, memory_order_relaxed);
            return;
        }

        uint key = expected;
        if (key == code) {
            atomic_fetch_add_explicit(&hist_counts[idx], 1u, memory_order_relaxed);
            return;
        }

        idx = (idx + 1u) & mask;
    }
}
