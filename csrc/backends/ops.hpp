#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace turbogator {

void geometric_product_baseline(const float* a, const float* b, float* out, size_t n);
void geometric_product_opt_v1(const float* a, const float* b, float* out, size_t n);
void geometric_product_vectorized(const float* a, const float* b, float* out, size_t n);

void equi_join_baseline(const float* a, const float* b, const float* ref, float* out, size_t n);
void equi_join_optimized_hardcoded(const float* a, const float* b, const float* ref, float* out, size_t n);
void equi_join_optimized_sparse(const float* a, const float* b, const float* ref, float* out, size_t n);
void equi_join_optimized_precompute_ab(const float* a, const float* b, const float* ref, float* out, size_t n);
void equi_join_optimized_unroll_k(const float* a, const float* b, const float* ref, float* out, size_t n);
void equi_join_restrict_unswitch(const float* a, const float* b, const float* ref, float* out, size_t n);
void equi_geometric_attention_baseline(
    const float* q, const float* k, const float* v, float* out,
    int64_t B, int64_t H, int64_t T, int64_t C,
    const std::vector<std::string>& kinds,
    const std::vector<const float*>& weights,
    const float* attn_mask,
    bool is_causal
);
void equi_geometric_attention_optimized1(
    const float* q, const float* k, const float* v, float* out,
    int64_t B, int64_t H, int64_t T, int64_t C,
    const std::vector<std::string>& kinds,
    const std::vector<const float*>& weights,
    const float* attn_mask,
    bool is_causal
);
void equi_geometric_attention_optimized2(
    const float* q, const float* k, const float* v, float* out,
    int64_t B, int64_t H, int64_t T, int64_t C,
    const std::vector<std::string>& kinds,
    const std::vector<const float*>& weights,
    const float* attn_mask,
    bool is_causal
);
void scaler_gated_gelu_baseline(const float* x, float* out, size_t n);
void equi_linear_baseline(const float* x, const float* weight, const float* bias, float* out,
                          size_t batch, size_t in_channels, size_t out_channels,
                          bool normalize_basis = true);
void equi_linear_opt_v1(const float* x, const float* weight, const float* bias, float* out,
                        size_t batch, size_t in_channels, size_t out_channels,
                        bool normalize_basis = true);
void equi_linear_opt_v2(const float* x, const float* weight, const float* bias, float* out,
                        size_t batch, size_t in_channels, size_t out_channels,
                        bool normalize_basis = true);
void equi_linear_opt_vectorized(const float* x, const float* weight, const float* bias, float* out,
                                size_t batch, size_t in_channels, size_t out_channels,
                                bool normalize_basis = true);
void equi_rms_norm_baseline(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps);
void equi_rms_norm_branchless_clamp(const float* x, const float* weight, float* out,
                                    size_t batch, size_t n_channels, float eps);
void equi_rms_norm_restrict(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps);
void equi_rms_norm_unrolled_selector(const float* x, const float* weight,
                                     float* out, size_t batch, size_t n_channels, float eps);
void equi_rms_norm_reciprocal_div(const float* x, const float* weight, float* out,
                                  size_t batch, size_t n_channels, float eps);
void equi_rms_norm_prefetch(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps);
void equi_rms_norm_unrolled_channels_4(const float* x, const float* weight, float* out,
                                       size_t batch, size_t n_channels, float eps);
void equi_rms_norm_assume_aligned(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps);
void equi_rms_norm_combined(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps);

}  // namespace turbogator
