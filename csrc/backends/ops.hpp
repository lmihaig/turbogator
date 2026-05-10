#pragma once

#include <cstddef>

namespace tg {

void geometric_product_baseline(const float* a, const float* b, float* out, size_t n);
void geometric_product_vectorized(const float* a, const float* b, float* out, size_t n);

void equi_join_baseline(const float* a, const float* b, const float* ref, float* out, size_t n);
void equi_geometric_attention_baseline(const float* q, const float* k, const float* v, float* out, size_t n);
void scaler_gated_gelu_baseline(const float* x, float* out, size_t n);

}  // namespace tg
