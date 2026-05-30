#include <immintrin.h>

#include <cmath>
#include <unordered_map>
#include <vector>

#include "ops.hpp"

namespace turbogator {
alignas(32) static const int32_t PERM_LO[8] = {0, 0, 0, 0, 0, 2, 3, 4};
alignas(32) static const int32_t PERM_HI[8] = {0, 0, 0, 0, 1, 2, 0, 6};

__attribute__((always_inline)) static inline void pack_weight(float* __restrict__ dst, const float* __restrict__ w) {
    dst[0] = w[0];
    dst[1] = dst[2] = dst[3] = dst[4] = w[1];
    dst[5] = dst[6] = dst[7] = w[2];

    dst[8] = dst[9] = dst[10] = w[2];
    dst[11] = dst[12] = dst[13] = dst[14] = w[3];
    dst[15]                               = w[4];

    dst[16] = 0.f;
    dst[17] = w[5];
    dst[18] = dst[19] = dst[20] = 0.f;
    dst[21] = dst[22] = dst[23] = w[6];

    dst[24] = dst[25] = dst[26] = 0.f;
    dst[27] = dst[28] = dst[29] = w[7];
    dst[30]                     = 0.f;
    dst[31]                     = w[8];
}

namespace {
struct PackedWeights {
    std::vector<float> data;
    bool normalize = false;
};

// shuffle + fma row
__attribute__((always_inline)) static inline void fma_row(__m256& acc_lo,
                                                          __m256& acc_hi,
                                                          __m256 w0,
                                                          __m256 w1,
                                                          __m256 w2,
                                                          __m256 w3,
                                                          const float* __restrict__ xp,
                                                          __m256i perm_lo,
                                                          __m256i perm_hi) {
    __m256 xlo = _mm256_loadu_ps(xp);
    __m256 xhi = _mm256_loadu_ps(xp + 8);

    acc_lo = _mm256_fmadd_ps(w0, xlo, acc_lo);
    acc_hi = _mm256_fmadd_ps(w1, xhi, acc_hi);

    acc_lo = _mm256_fmadd_ps(w2, _mm256_permutevar8x32_ps(xlo, perm_lo), acc_lo);
    acc_hi = _mm256_fmadd_ps(w3, _mm256_permutevar8x32_ps(xhi, perm_hi), acc_hi);
}
}  // namespace

void equi_linear_vectorized(const float* __restrict__ x,
                            const float* __restrict__ weight,
                            const float* __restrict__ bias,
                            float* __restrict__ out,
                            size_t batch,
                            size_t in_channels,
                            size_t out_channels,
                            bool normalize_basis) {
    float scales[9] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    if (normalize_basis) {
        scales[1] = .5f;
        scales[2] = 1.0f / std::sqrt(6.0f);
        scales[3] = 1.0f / std::sqrt(4.0f);
        scales[6] = 1.0f / std::sqrt(3.0f);
        scales[7] = 1.0f / std::sqrt(3.0f);
    }

    // cache reuseeable weights
    static thread_local std::unordered_map<const float*, PackedWeights> w_cache;
    const float* weight_key    = weight;
    PackedWeights& pk          = w_cache[weight_key];
    const size_t packed_floats = out_channels * in_channels * 32;
    if (pk.data.size() != packed_floats || pk.normalize != normalize_basis) {
        pk.data.resize(packed_floats);
        pk.normalize = normalize_basis;
        for (size_t ch = 0; ch < out_channels * in_channels; ++ch) {
            const float* src = weight + ch * 9;
            float w[9];
            for (int i = 0; i < 9; ++i)
                w[i] = src[i] * scales[i];
            pack_weight(pk.data.data() + ch * 32, w);
        }
    }
    const float* w_packed = pk.data.data();

    const __m256i perm_lo = _mm256_load_si256((const __m256i*)PERM_LO);
    const __m256i perm_hi = _mm256_load_si256((const __m256i*)PERM_HI);

    const size_t x_stride = in_channels * 16;

    // load weights once for 4 rows
    size_t b0 = 0;
    for (; b0 + 4 <= batch; b0 += 4) {
        const float* xr0 = x + (b0 + 0) * x_stride;
        const float* xr1 = x + (b0 + 1) * x_stride;
        const float* xr2 = x + (b0 + 2) * x_stride;
        const float* xr3 = x + (b0 + 3) * x_stride;
        for (size_t oc = 0; oc < out_channels; ++oc) {
            __m256 lo0 = _mm256_setzero_ps(), hi0 = _mm256_setzero_ps();
            __m256 lo1 = _mm256_setzero_ps(), hi1 = _mm256_setzero_ps();
            __m256 lo2 = _mm256_setzero_ps(), hi2 = _mm256_setzero_ps();
            __m256 lo3 = _mm256_setzero_ps(), hi3 = _mm256_setzero_ps();

            const float* wp = w_packed + oc * in_channels * 32;
            for (size_t ic = 0; ic < in_channels; ++ic) {
                const float* w = wp + ic * 32;
                __m256 w0      = _mm256_loadu_ps(w);
                __m256 w1      = _mm256_loadu_ps(w + 8);
                __m256 w2      = _mm256_loadu_ps(w + 16);
                __m256 w3      = _mm256_loadu_ps(w + 24);
                const size_t o = ic * 16;
                fma_row(lo0, hi0, w0, w1, w2, w3, xr0 + o, perm_lo, perm_hi);
                fma_row(lo1, hi1, w0, w1, w2, w3, xr1 + o, perm_lo, perm_hi);
                fma_row(lo2, hi2, w0, w1, w2, w3, xr2 + o, perm_lo, perm_hi);
                fma_row(lo3, hi3, w0, w1, w2, w3, xr3 + o, perm_lo, perm_hi);
            }

            float* o0 = out + ((b0 + 0) * out_channels + oc) * 16;
            float* o1 = out + ((b0 + 1) * out_channels + oc) * 16;
            float* o2 = out + ((b0 + 2) * out_channels + oc) * 16;
            float* o3 = out + ((b0 + 3) * out_channels + oc) * 16;
            _mm256_storeu_ps(o0, lo0);
            _mm256_storeu_ps(o0 + 8, hi0);
            _mm256_storeu_ps(o1, lo1);
            _mm256_storeu_ps(o1 + 8, hi1);
            _mm256_storeu_ps(o2, lo2);
            _mm256_storeu_ps(o2 + 8, hi2);
            _mm256_storeu_ps(o3, lo3);
            _mm256_storeu_ps(o3 + 8, hi3);
        }
    }

    // tail
    for (; b0 < batch; ++b0) {
        const float* xb = x + b0 * x_stride;
        for (size_t oc = 0; oc < out_channels; ++oc) {
            __m256 lo       = _mm256_setzero_ps();
            __m256 hi       = _mm256_setzero_ps();
            const float* wp = w_packed + oc * in_channels * 32;
            for (size_t ic = 0; ic < in_channels; ++ic) {
                const float* w = wp + ic * 32;
                fma_row(lo,
                        hi,
                        _mm256_loadu_ps(w),
                        _mm256_loadu_ps(w + 8),
                        _mm256_loadu_ps(w + 16),
                        _mm256_loadu_ps(w + 24),
                        xb + ic * 16,
                        perm_lo,
                        perm_hi);
            }
            float* out_bo = out + (b0 * out_channels + oc) * 16;
            _mm256_storeu_ps(out_bo, lo);
            _mm256_storeu_ps(out_bo + 8, hi);
        }
    }

    if (bias != nullptr) {
        for (size_t b = 0; b < batch; ++b) {
            for (size_t oc = 0; oc < out_channels; ++oc) {
                out[b * out_channels * 16 + oc * 16] += bias[oc];
            }
        }
    }
}

}  // namespace turbogator
