#include <cmath>

#include "ops.hpp"

namespace turbogator {

void equi_rms_norm_opt_v1(const float* x, const float* weight, float* out, size_t batch, size_t n_channels, float eps) {
    const float* __restrict__ x_a = (const float*)__builtin_assume_aligned(x, 64);
    const float* __restrict__ w_r = weight;
    float* __restrict__ out_a     = (float*)__builtin_assume_aligned(out, 64);
    const float inv_n_channels    = 1.0f / (float)n_channels;

    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x_a + b * n_channels * 16;

        if (b + 1 < batch) {
            const float* x_next = x_a + (b + 1) * n_channels * 16;
            for (size_t i = 0; i < n_channels * 16; i += 16)
                __builtin_prefetch(x_next + i, 0, 1);
        }

        float norm               = 0.0f;
        const size_t full_groups = n_channels / 4;
        for (size_t cg = 0; cg < full_groups; ++cg) {
            const float* mv0 = x_b + (cg * 4 + 0) * 16;
            const float* mv1 = x_b + (cg * 4 + 1) * 16;
            const float* mv2 = x_b + (cg * 4 + 2) * 16;
            const float* mv3 = x_b + (cg * 4 + 3) * 16;

            float ip0 = mv0[0] * mv0[0] + mv0[2] * mv0[2] + mv0[3] * mv0[3] + mv0[4] * mv0[4] + mv0[8] * mv0[8] +
                        mv0[9] * mv0[9] + mv0[10] * mv0[10] + mv0[14] * mv0[14];
            float ip1 = mv1[0] * mv1[0] + mv1[2] * mv1[2] + mv1[3] * mv1[3] + mv1[4] * mv1[4] + mv1[8] * mv1[8] +
                        mv1[9] * mv1[9] + mv1[10] * mv1[10] + mv1[14] * mv1[14];
            float ip2 = mv2[0] * mv2[0] + mv2[2] * mv2[2] + mv2[3] * mv2[3] + mv2[4] * mv2[4] + mv2[8] * mv2[8] +
                        mv2[9] * mv2[9] + mv2[10] * mv2[10] + mv2[14] * mv2[14];
            float ip3 = mv3[0] * mv3[0] + mv3[2] * mv3[2] + mv3[3] * mv3[3] + mv3[4] * mv3[4] + mv3[8] * mv3[8] +
                        mv3[9] * mv3[9] + mv3[10] * mv3[10] + mv3[14] * mv3[14];

            norm += ip0 + ip1 + ip2 + ip3;
        }
        for (size_t c = full_groups * 4; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            norm += mv[0] * mv[0] + mv[2] * mv[2] + mv[3] * mv[3] + mv[4] * mv[4] + mv[8] * mv[8] + mv[9] * mv[9] +
                    mv[10] * mv[10] + mv[14] * mv[14];
        }
        norm *= inv_n_channels;

        float scale = 1.0f / std::sqrt(std::fmax(norm, eps));

        float* out_b = out_a + b * n_channels * 16;
        if (w_r != nullptr) {
            for (size_t c = 0; c < n_channels; ++c) {
                float cscale     = scale * w_r[c];
                const float* x_c = x_b + c * 16;
                float* o_c       = out_b + c * 16;
                for (int d = 0; d < 16; ++d)
                    o_c[d] = x_c[d] * cscale;
            }
        } else {
            for (size_t c = 0; c < n_channels; ++c) {
                const float* x_c = x_b + c * 16;
                float* o_c       = out_b + c * 16;
                for (int d = 0; d < 16; ++d)
                    o_c[d] = x_c[d] * scale;
            }
        }
    }
}

}  // namespace turbogator
