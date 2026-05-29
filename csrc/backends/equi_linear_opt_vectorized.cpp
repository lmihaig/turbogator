#include "ops.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace turbogator
{
    // PERM_LO: brings x[0] -> lane 1, x[2..4] -> lanes 5..7
    alignas(32) static const int32_t PERM_LO[8] = {0, 0, 0, 0, 0, 2, 3, 4};
    // PERM_HI: brings x[8..10] -> lanes 3..5, x[14] -> lane 7
    alignas(32) static const int32_t PERM_HI[8] = {0, 0, 0, 0, 1, 2, 0, 6};

    // Packing the 9 weights into a 32-float array for efficient access during the FMA operations.
    static void pack_weight(float *dst, const float *w)
    {
        dst[0]  = w[0];
        dst[1]  = dst[2] = dst[3] = dst[4] = w[1];
        dst[5]  = dst[6] = dst[7] = w[2];

        dst[8]  = dst[9] = dst[10] = w[2];
        dst[11] = dst[12] = dst[13] = dst[14] = w[3];
        dst[15] = w[4];

        dst[16] = 0.f; dst[17] = w[5];
        dst[18] = dst[19] = dst[20] = 0.f;
        dst[21] = dst[22] = dst[23] = w[6];

        dst[24] = dst[25] = dst[26] = 0.f;
        dst[27] = dst[28] = dst[29] = w[7];
        dst[30] = 0.f; dst[31] = w[8];
    }

    void equi_linear_opt_vectorized(
        const float *x, const float *weight, const float *bias, float *out,
        size_t batch, size_t in_channels, size_t out_channels,
        bool normalize_basis
    )
    {
        float scales[9] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        if (normalize_basis)
        {
            scales[1] = .5f;
            scales[2] = 1.0f / std::sqrt(6.0f);
            scales[3] = 1.0f / std::sqrt(4.0f);
            scales[6] = 1.0f / std::sqrt(3.0f);
            scales[7] = 1.0f / std::sqrt(3.0f);
        }

        std::vector<float> w_packed(out_channels * in_channels * 32);
        for (size_t ch = 0; ch < out_channels * in_channels; ++ch)
        {
            const float *src = weight + ch * 9;
            float w[9];
            for (int i = 0; i < 9; ++i) w[i] = src[i] * scales[i];
            pack_weight(w_packed.data() + ch * 32, w);
        }

        const __m256i perm_lo = _mm256_load_si256((const __m256i *)PERM_LO);
        const __m256i perm_hi = _mm256_load_si256((const __m256i *)PERM_HI);

        for (size_t b = 0; b < batch; ++b)
        {
            for (size_t oc = 0; oc < out_channels; ++oc)
            {
                __m256 acc_lo_0 = _mm256_setzero_ps();
                __m256 acc_lo_1 = _mm256_setzero_ps();
                __m256 acc_lo_2 = _mm256_setzero_ps();
                __m256 acc_lo_3 = _mm256_setzero_ps();
                __m256 acc_hi_0 = _mm256_setzero_ps();
                __m256 acc_hi_1 = _mm256_setzero_ps();
                __m256 acc_hi_2 = _mm256_setzero_ps();
                __m256 acc_hi_3 = _mm256_setzero_ps();

                const float *w_base = w_packed.data() + oc * in_channels * 32;
                const float *x_base = x + b * in_channels * 16;

#define EQUI_FMAS(acc_lo, acc_hi, wp, xp)                                                                      \
    do {                                                                                                       \
        __m256 xlo_ = _mm256_loadu_ps(xp);                                                                     \
        __m256 xhi_ = _mm256_loadu_ps((xp) + 8);                                                               \
        acc_lo = _mm256_fmadd_ps(_mm256_loadu_ps(wp),        xlo_, acc_lo);                                    \
        acc_hi = _mm256_fmadd_ps(_mm256_loadu_ps((wp) + 8),  xhi_, acc_hi);                                    \
        acc_lo = _mm256_fmadd_ps(_mm256_loadu_ps((wp) + 16), _mm256_permutevar8x32_ps(xlo_, perm_lo), acc_lo); \
        acc_hi = _mm256_fmadd_ps(_mm256_loadu_ps((wp) + 24), _mm256_permutevar8x32_ps(xhi_, perm_hi), acc_hi); \
    } while (0)

                size_t ic = 0;
                for (; ic + 3 < in_channels; ic += 4)
                {
                    EQUI_FMAS(acc_lo_0, acc_hi_0, w_base + ic * 32,       x_base + ic * 16);
                    EQUI_FMAS(acc_lo_1, acc_hi_1, w_base + (ic+1) * 32,   x_base + (ic+1) * 16);
                    EQUI_FMAS(acc_lo_2, acc_hi_2, w_base + (ic+2) * 32,   x_base + (ic+2) * 16);
                    EQUI_FMAS(acc_lo_3, acc_hi_3, w_base + (ic+3) * 32,   x_base + (ic+3) * 16);
                }
                for (; ic < in_channels; ++ic)
                    EQUI_FMAS(acc_lo_0, acc_hi_0, w_base + ic * 32, x_base + ic * 16);

#undef EQUI_FMAS

                __m256 final_lo = _mm256_add_ps(_mm256_add_ps(acc_lo_0, acc_lo_1),
                                                _mm256_add_ps(acc_lo_2, acc_lo_3));
                __m256 final_hi = _mm256_add_ps(_mm256_add_ps(acc_hi_0, acc_hi_1),
                                                _mm256_add_ps(acc_hi_2, acc_hi_3));

                float *out_bo = out + (b * out_channels + oc) * 16;
                _mm256_storeu_ps(out_bo,     final_lo);
                _mm256_storeu_ps(out_bo + 8, final_hi);
            }
        }

        if (bias != nullptr)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t oc = 0; oc < out_channels; ++oc)
                {
                    out[b * out_channels * 16 + oc * 16] += bias[oc];
                }
            }
        }
    }

} // namespace turbogator
