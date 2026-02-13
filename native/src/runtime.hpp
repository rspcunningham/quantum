#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace quantum_native {

struct StaticProgramData {
    int n_qubits;
    int n_bits;

    std::vector<int32_t> op_table;           // rows of 8
    std::vector<int32_t> dispatch_table;     // rows of 4
    std::vector<int32_t> target_pool;

    std::vector<float> diag_pool_re;
    std::vector<float> diag_pool_im;

    std::vector<int32_t> perm_pool;

    std::vector<float> phase_pool_re;
    std::vector<float> phase_pool_im;

    std::vector<float> dense_pool_re;
    std::vector<float> dense_pool_im;

    std::vector<uint8_t> monomial_blob;
    std::vector<int32_t> terminal_measurements;  // flattened (qubit, bit)
};

void set_module_file_path(const std::string& module_file_path);
void set_metallib_path_override(const std::string& metallib_path);

std::int64_t compile_static_program(const StaticProgramData& data);
std::unordered_map<std::string, std::int64_t> execute_static_program(
    std::int64_t handle,
    std::int64_t num_shots,
    std::optional<std::uint64_t> seed
);
std::unordered_map<std::string, std::int64_t> get_program_stats(std::int64_t handle);
void free_program(std::int64_t handle);

}  // namespace quantum_native
