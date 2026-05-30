#include <immintrin.h>

#include <cmath>

#include "ops.hpp"

namespace turbogator {

void scaler_gated_gelu_vectorized(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    if (n % 16 != 0) __builtin_unreachable();
    size_t num_mvs = n / 16;
    for (size_t i = 0; i < num_mvs; ++i) {
        const float* mv_in = x + i * 16;
        float* mv_out      = out + i * 16;

        float g     = mv_in[0];
        float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        float gate  = 0.5f * g * (1.0f + std::tanh(inner));

        __m256 vgate = _mm256_set1_ps(gate);
        _mm256_storeu_ps(mv_out, _mm256_mul_ps(_mm256_loadu_ps(mv_in), vgate));
        _mm256_storeu_ps(mv_out + 8, _mm256_mul_ps(_mm256_loadu_ps(mv_in + 8), vgate));
    }
}

}  // namespace turbogator
