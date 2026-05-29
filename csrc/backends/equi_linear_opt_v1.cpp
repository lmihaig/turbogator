#include <cmath>
#include <cstring>
#include <utility>
#include <variant>
#include <vector>

#include "ops.hpp"

using BasisElement = std::variant<int, std::pair<int, int>>;

namespace turbogator {
namespace {
struct LocalEquiLinearBasis {
    float unnorm_basis[9][16][16] = {};
    float norm_basis[9][16][16]   = {};

    LocalEquiLinearBasis() {
        const std::vector<std::vector<BasisElement>> basis_elements = {
            {0},
            {1, 2, 3, 4},
            {5, 6, 7, 8, 9, 10},
            {11, 12, 13, 14},
            {15},
            {std::pair<int, int>{1, 0}},
            {std::pair<int, int>{5, 2}, std::pair<int, int>{6, 3}, std::pair<int, int>{7, 4}},
            {std::pair<int, int>{11, 8}, std::pair<int, int>{12, 9}, std::pair<int, int>{13, 10}},
            {std::pair<int, int>{15, 14}},
        };

        for (size_t k = 0; k < basis_elements.size(); ++k) {
            float w[16][16] = {};

            for (const auto& element : basis_elements[k]) {
                if (std::holds_alternative<int>(element)) {
                    int index       = std::get<int>(element);
                    w[index][index] = 1.0f;
                } else {
                    auto [i, j] = std::get<std::pair<int, int>>(element);
                    w[i][j]     = 1.0f;
                }
            }

            std::memcpy(unnorm_basis[k], w, sizeof(w));

            float sum_sq = 0.0f;
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j)
                    sum_sq += w[i][j] * w[i][j];

            float norm = std::sqrt(sum_sq);
            if (norm > 0.0f) {
                for (int i = 0; i < 16; ++i)
                    for (int j = 0; j < 16; ++j)
                        w[i][j] /= norm;
            }
            std::memcpy(norm_basis[k], w, sizeof(w));
        }
    }
};
}  // namespace

void equi_linear_opt_v1(const float* x,
                        const float* weight,
                        const float* bias,
                        float* out,
                        size_t batch,
                        size_t in_channels,
                        size_t out_channels,
                        bool normalize_basis) {
    static const LocalEquiLinearBasis EQUI_BASIS;
    const float (*basis)[16][16] = normalize_basis ? EQUI_BASIS.norm_basis : EQUI_BASIS.unnorm_basis;

    // step 1: kernel[o, i, s, d] = sum_w weight[o,i,w] * basis[w,d,s]
    std::vector<float> kernel(out_channels * in_channels * 16 * 16);

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t s = 0; s < 16; ++s) {
                for (size_t d = 0; d < 16; ++d) {
                    float sum = 0.0f;
                    for (size_t w = 0; w < 9; ++w) {
                        sum += weight[(oc * in_channels + ic) * 9 + w] * basis[w][d][s];
                    }
                    kernel[((oc * in_channels + ic) * 16 + s) * 16 + d] = sum;
                }
            }
        }
    }

    // step 2: out[b,o,d] = sum_{ic,s} kernel[o,ic,s,d] * x[b,ic,s]
    // OPT1:
    // - register accumulator: acc[16] holds all d values for one (b, oc)
    // - x[b,ic,s] is loaded once per (ic,s) and reused across all 16 d
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            float acc[16] = {0.0f};

            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t s = 0; s < 16; ++s) {
                    float xv = x[b * in_channels * 16 + ic * 16 + s];
                    for (size_t d = 0; d < 16; ++d) {
                        acc[d] += kernel[((oc * in_channels + ic) * 16 + s) * 16 + d] * xv;
                    }
                }
            }

            for (size_t d = 0; d < 16; ++d) {
                out[b * out_channels * 16 + oc * 16 + d] = acc[d];
            }
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