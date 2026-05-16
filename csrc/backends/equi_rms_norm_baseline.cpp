#include "ops.hpp"

#include <cmath>

namespace turbogator {

// PGA inner product selects blades that don't contain e_0
static const int IP_SELECTOR[] = {0, 2, 3, 4, 8, 9, 10, 14};
static const int IP_SELECTOR_LEN = 8;

void equi_rms_norm_baseline(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps)
{
    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x + b * n_channels * 16;

        // mean over channels of selective inner_product(x, x) from PGA
        float norm = 0.0f;
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < IP_SELECTOR_LEN; ++s)
                ip += mv[IP_SELECTOR[s]] * mv[IP_SELECTOR[s]];
            norm += ip;
        }
        norm /= (float)n_channels;

        float scale = 1.0f / std::sqrt(norm > eps ? norm : eps);

        float* out_b = out + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (weight != nullptr) ? weight[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float*      o_c  = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

}  // namespace turbogator
