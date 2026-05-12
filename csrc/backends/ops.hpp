#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace turbogator {

void geometric_product_baseline(const float* a, const float* b, float* out, size_t n);
void geometric_product_vectorized(const float* a, const float* b, float* out, size_t n);

void equi_join_baseline(const float* a, const float* b, const float* ref, float* out, size_t n);
void equi_geometric_attention_baseline(
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
void equi_rms_norm_baseline(const float* x, const float* weight, float* out, size_t n);

}  // namespace turbogator
