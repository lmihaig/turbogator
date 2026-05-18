#include "ops.hpp"
#include <vector>
#include <cmath>

namespace turbogator
{
    // opt v2: d/s loop unrolled with direct indexing;
    // compute weights * x directly in inner loop

    void equi_linear_opt_v2(const float *x, const float *weight, const float *bias, float *out,
                            size_t batch, size_t in_channels, size_t out_channels,
                            bool normalize_basis)
    {
        // number of non-zeros in basis[w]
        static const int group_size[9] = {1, 4, 6, 4, 1, 1, 3, 3, 1};

        // working weights: w_use[o,i,w] = weight[o,i,w] * scale[w]
        // TODO: for normalize_basis=false, scale[w] = 1 could skip this
        std::vector<float> w_use(out_channels * in_channels * 9);
        float scale[9];
        if (normalize_basis)
        {
            for (int w = 0; w < 9; ++w)
                scale[w] = 1.0f / std::sqrt(float(group_size[w]));
        }
        else
        {
            for (int w = 0; w < 9; ++w)
                scale[w] = 1.0f;
        }
        for (size_t channel = 0; channel < out_channels * in_channels; ++channel)
        {
            const float *w_orig = weight + channel * 9;
            float *w_scaled = w_use.data() + channel * 9;
            for (int w = 0; w < 9; ++w)
                w_scaled[w] = w_orig[w] * scale[w];
        }

        for (size_t b = 0; b < batch; ++b)
        {
            for (size_t oc = 0; oc < out_channels; ++oc)
            {
                float acc[16] = {0.0f};

                for (size_t ic = 0; ic < in_channels; ++ic)
                {
                    const float *w_oi = w_use.data() + (oc * in_channels + ic) * 9;
                    const float *x_bi = x + (b * in_channels + ic) * 16;

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

                float *out_bo = out + (b * out_channels + oc) * 16;
                for (size_t d = 0; d < 16; ++d)
                    out_bo[d] = acc[d];
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
