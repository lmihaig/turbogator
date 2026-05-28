#include "ops.hpp"
#include <vector>
#include <cmath>
#include <cstring>

namespace turbogator
{
    // opt v3: oc-tiling (blocking by 4) on top of v2

    static constexpr size_t OC_BLOCK = 4;

    static inline void equi_linear_v3_block4(const float *x_b, const float *w_base,
                                             float *out_base, size_t in_channels,
                                             float b0, float b1, float b2, float b3)
    {
        float acc0[16], acc1[16], acc2[16], acc3[16];

        // bias fused into acc init
        acc0[0] = b0;
        acc1[0] = b1;
        acc2[0] = b2;
        acc3[0] = b3;
        for (int d = 1; d < 16; ++d)
        {
            acc0[d] = 0.0f;
            acc1[d] = 0.0f;
            acc2[d] = 0.0f;
            acc3[d] = 0.0f;
        }

        const size_t row_stride = in_channels * 9;

        for (size_t ic = 0; ic < in_channels; ++ic)
        {
            const float *x_bi = x_b + ic * 16;

            // cache the 16 source-blade values once per (b, ic); reused 4x
            const float x0 = x_bi[0], x1 = x_bi[1], x2 = x_bi[2], x3 = x_bi[3];
            const float x4 = x_bi[4], x5 = x_bi[5], x6 = x_bi[6], x7 = x_bi[7];
            const float x8 = x_bi[8], x9 = x_bi[9], x10 = x_bi[10], x11 = x_bi[11];
            const float x12 = x_bi[12], x13 = x_bi[13], x14 = x_bi[14], x15 = x_bi[15];

            const float *w0r = w_base + 0 * row_stride + ic * 9;
            const float *w1r = w_base + 1 * row_stride + ic * 9;
            const float *w2r = w_base + 2 * row_stride + ic * 9;
            const float *w3r = w_base + 3 * row_stride + ic * 9;

#define EQ_FMA_V3(WR, ACC)                  \
    {                                       \
        const float w0 = WR[0], w1 = WR[1]; \
        const float w2 = WR[2], w3 = WR[3]; \
        const float w4 = WR[4], w5 = WR[5]; \
        const float w6 = WR[6], w7 = WR[7]; \
        const float w8 = WR[8];             \
        ACC[0] += w0 * x0;                  \
        ACC[1] += w1 * x1 + w5 * x0;        \
        ACC[2] += w1 * x2;                  \
        ACC[3] += w1 * x3;                  \
        ACC[4] += w1 * x4;                  \
        ACC[5] += w2 * x5 + w6 * x2;        \
        ACC[6] += w2 * x6 + w6 * x3;        \
        ACC[7] += w2 * x7 + w6 * x4;        \
        ACC[8] += w2 * x8;                  \
        ACC[9] += w2 * x9;                  \
        ACC[10] += w2 * x10;                \
        ACC[11] += w3 * x11 + w7 * x8;      \
        ACC[12] += w3 * x12 + w7 * x9;      \
        ACC[13] += w3 * x13 + w7 * x10;     \
        ACC[14] += w3 * x14;                \
        ACC[15] += w4 * x15 + w8 * x14;     \
    }

            EQ_FMA_V3(w0r, acc0)
            EQ_FMA_V3(w1r, acc1)
            EQ_FMA_V3(w2r, acc2)
            EQ_FMA_V3(w3r, acc3)
#undef EQ_FMA_V3
        }

        for (int d = 0; d < 16; ++d)
            out_base[0 * 16 + d] = acc0[d];
        for (int d = 0; d < 16; ++d)
            out_base[1 * 16 + d] = acc1[d];
        for (int d = 0; d < 16; ++d)
            out_base[2 * 16 + d] = acc2[d];
        for (int d = 0; d < 16; ++d)
            out_base[3 * 16 + d] = acc3[d];
    }

    void equi_linear_opt_v3(const float *x, const float *weight, const float *bias, float *out,
                            size_t batch, size_t in_channels, size_t out_channels,
                            bool normalize_basis)
    {
        // pre-scale weights;
        // scale[w] = 1/sqrt(group_size[w]) for normalize_basis, else 1;
        // group_size = {1, 4, 6, 4, 1, 1, 3, 3, 1}, so 4 of the 9 scales
        std::vector<float> w_use(out_channels * in_channels * 9);

        if (normalize_basis)
        {
            const float S1 = 0.5f;
            const float S2 = 1.0f / std::sqrt(6.0f);
            const float S3 = 0.5f;
            const float S6 = 1.0f / std::sqrt(3.0f);
            const float S7 = 1.0f / std::sqrt(3.0f);

            for (size_t channel = 0; channel < out_channels * in_channels; ++channel)
            {
                const float *w_orig = weight + channel * 9;
                float *w_scaled = w_use.data() + channel * 9;
                w_scaled[0] = w_orig[0];
                w_scaled[1] = w_orig[1] * S1;
                w_scaled[2] = w_orig[2] * S2;
                w_scaled[3] = w_orig[3] * S3;
                w_scaled[4] = w_orig[4];
                w_scaled[5] = w_orig[5];
                w_scaled[6] = w_orig[6] * S6;
                w_scaled[7] = w_orig[7] * S7;
                w_scaled[8] = w_orig[8];
            }
        }
        else
        {
            std::memcpy(w_use.data(), weight, out_channels * in_channels * 9 * sizeof(float));
        }

        const size_t oc_tiles = out_channels / OC_BLOCK;
        const size_t oc_tail_start = oc_tiles * OC_BLOCK;

        for (size_t b = 0; b < batch; ++b)
        {
            const float *x_b = x + b * in_channels * 16;
            float *out_b = out + b * out_channels * 16;

            for (size_t t = 0; t < oc_tiles; ++t)
            {
                const size_t oc_base = t * OC_BLOCK;
                const float *w_base = w_use.data() + oc_base * in_channels * 9;
                float *out_base = out_b + oc_base * 16;

                const float b0 = (bias != nullptr) ? bias[oc_base + 0] : 0.0f;
                const float b1 = (bias != nullptr) ? bias[oc_base + 1] : 0.0f;
                const float b2 = (bias != nullptr) ? bias[oc_base + 2] : 0.0f;
                const float b3 = (bias != nullptr) ? bias[oc_base + 3] : 0.0f;

                equi_linear_v3_block4(x_b, w_base, out_base, in_channels, b0, b1, b2, b3);
            }

            // scalar tail for out_channels not divisible by OC_BLOCK
            for (size_t oc = oc_tail_start; oc < out_channels; ++oc)
            {
                float acc[16];
                acc[0] = (bias != nullptr) ? bias[oc] : 0.0f;
                for (int d = 1; d < 16; ++d)
                    acc[d] = 0.0f;

                for (size_t ic = 0; ic < in_channels; ++ic)
                {
                    const float *w_oi = w_use.data() + (oc * in_channels + ic) * 9;
                    const float *x_bi = x_b + ic * 16;

                    // load weights for (oc, ic)
                    const float w0 = w_oi[0];
                    const float w1 = w_oi[1];
                    const float w2 = w_oi[2];
                    const float w3 = w_oi[3];
                    const float w4 = w_oi[4];
                    const float w5 = w_oi[5];
                    const float w6 = w_oi[6];
                    const float w7 = w_oi[7];
                    const float w8 = w_oi[8];

                    // w=0
                    acc[0] += w0 * x_bi[0];

                    // w=1
                    acc[1] += w1 * x_bi[1];
                    acc[2] += w1 * x_bi[2];
                    acc[3] += w1 * x_bi[3];
                    acc[4] += w1 * x_bi[4];

                    // w=2
                    acc[5] += w2 * x_bi[5];
                    acc[6] += w2 * x_bi[6];
                    acc[7] += w2 * x_bi[7];
                    acc[8] += w2 * x_bi[8];
                    acc[9] += w2 * x_bi[9];
                    acc[10] += w2 * x_bi[10];

                    // w=3
                    acc[11] += w3 * x_bi[11];
                    acc[12] += w3 * x_bi[12];
                    acc[13] += w3 * x_bi[13];
                    acc[14] += w3 * x_bi[14];

                    // w=4
                    acc[15] += w4 * x_bi[15];

                    // w=5
                    acc[1] += w5 * x_bi[0];

                    // w=6
                    acc[5] += w6 * x_bi[2];
                    acc[6] += w6 * x_bi[3];
                    acc[7] += w6 * x_bi[4];

                    // w=7
                    acc[11] += w7 * x_bi[8];
                    acc[12] += w7 * x_bi[9];
                    acc[13] += w7 * x_bi[10];

                    // w=8
                    acc[15] += w8 * x_bi[14];
                }

                float *out_bo = out_b + oc * 16;
                for (int d = 0; d < 16; ++d)
                    out_bo[d] = acc[d];
            }
        }
    }

} // namespace turbogator
