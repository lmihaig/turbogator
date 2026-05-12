#include "ops.hpp"
#include <vector>
#include <utility>
#include <variant>
#include <cmath>
#include <cstring>

using BasisElement = std::variant<int, std::pair<int, int>>;

namespace turbogator

{
    void _compute_pin_equi_linear_basis(bool normalize_basis, float basis[9][16][16])
    {
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

        std::memset(basis, 0, sizeof(float) * 9 * 16 * 16);

        for (size_t k = 0; k < basis_elements.size(); ++k)
        {
            float w[16][16] = {};

            for (const auto &element : basis_elements[k])
            {
                if (std::holds_alternative<int>(element))
                {
                    int index = std::get<int>(element);
                    w[index][index] = 1.0f;
                }
                else
                {
                    auto [i, j] = std::get<std::pair<int, int>>(element);
                    w[i][j] = 1.0f;
                }
            }

            if (normalize_basis)
            {
                float sum_sq = 0.0f;
                for (int i = 0; i < 16; ++i)
                    for (int j = 0; j < 16; ++j)
                        sum_sq += w[i][j] * w[i][j];

                float norm = std::sqrt(sum_sq);
                if (norm > 0.0f)
                {
                    for (int i = 0; i < 16; ++i)
                        for (int j = 0; j < 16; ++j)
                            w[i][j] /= norm;
                }
            }

            std::memcpy(basis[k], w, sizeof(w));
        }
    }

    void equi_linear_baseline(const float *x, const float *weight, const float *bias, float *out,
                              size_t batch, size_t in_channels, size_t out_channels,
                              bool normalize_basis)
    {
        (void)x;
        (void)weight;
        (void)bias;

        float basis[9][16][16];
        memset(basis, 0, sizeof(basis));
        _compute_pin_equi_linear_basis(normalize_basis, basis);

        for (size_t i = 0; i < n; ++i)
        {
            out[i] = 0.0f;
        }
    }

} // namespace turbogator
