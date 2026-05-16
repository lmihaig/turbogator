#include <cmath>

#include "ops.hpp"

namespace turbogator {

void scaler_gated_gelu_baseline(const float* x, float* out, size_t n) {
    size_t num_mvs = n / 16;
    for (size_t i = 0; i < num_mvs; ++i) {
        const float* mv_in  = x   + i * 16;
        float*       mv_out = out + i * 16;

        float g     = mv_in[0];
        float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        float gate  = 0.5f * g * (1.0f + std::tanh(inner));

        for (int d = 0; d < 16; ++d)
            mv_out[d] = mv_in[d] * gate;
    }
}

}  // namespace turbogator
