#include <immintrin.h>

#include <cmath>

#include "ops.hpp"

namespace turbogator {

__attribute__((always_inline)) static inline float hsum256_ps(__m256 v) {
    __m128 lo   = _mm256_castps256_ps128(v);
    __m128 hi   = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x1));
    return _mm_cvtss_f32(sum1);
}

void equi_rms_norm_vectorized(
    const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out,
    size_t batch, size_t n_channels, float eps) {
    if (n_channels % 2 != 0) __builtin_unreachable();
    const float* __restrict__ x_r   = x;
    const float* __restrict__ w_r   = weight;
    float* __restrict__       out_r = out;

    const float inv_n_channels = 1.0f / (float)n_channels;
    const __m256 mask0         = _mm256_setr_ps(1.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f);
    const __m256 mask1         = _mm256_setr_ps(1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f);

    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x_r + b * n_channels * 16;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            __m256 v0       = _mm256_loadu_ps(mv);
            __m256 v1       = _mm256_loadu_ps(mv + 8);
            acc0            = _mm256_fmadd_ps(v0, v0, acc0);
            acc1            = _mm256_fmadd_ps(v1, v1, acc1);
        }

        float norm = (hsum256_ps(_mm256_mul_ps(acc0, mask0)) + hsum256_ps(_mm256_mul_ps(acc1, mask1))) * inv_n_channels;

        float clamped = std::fmax(norm, eps);
        float scale   = 1.0f / std::sqrt(clamped);

        float* out_b = out_r + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w          = (w_r != nullptr) ? w_r[c] : 1.0f;
            float cscale     = scale * w;
            __m256 scale_v   = _mm256_set1_ps(cscale);
            const float* x_c = x_b + c * 16;
            float* o_c       = out_b + c * 16;
            __m256 v0        = _mm256_loadu_ps(x_c);
            __m256 v1        = _mm256_loadu_ps(x_c + 8);
            _mm256_storeu_ps(o_c, _mm256_mul_ps(v0, scale_v));
            _mm256_storeu_ps(o_c + 8, _mm256_mul_ps(v1, scale_v));
        }
    }
}

}  // namespace turbogator
