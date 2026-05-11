#include <cmath>

#include "ops.hpp"

namespace turbogator {

void scaler_gated_gelu_baseline(const float* x, float* out, size_t n) {
    if (n == 0) {
        return;
    }

    const float gate_input = x[0];
    const float inner = 0.7978845608028654f * (gate_input + 0.044715f * gate_input * gate_input * gate_input);
    const float gate = 0.5f * gate_input * (1.0f + std::tanh(inner));

    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] * gate;
    }
}

}  // namespace turbogator
