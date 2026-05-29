#pragma once
#include "equi_join_constants.hpp"

namespace turbogator {

struct JoinKernel {
    float data[16][16][16] = {};

    JoinKernel() {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                float x[16] = {0};
                float y[16] = {0};
                x[i]        = 1.0f;
                y[j]        = 1.0f;

                float dual_x[16];
                float dual_y[16];
                float prod[16];
                float final_out[16];

                equi_dual(x, dual_x);
                equi_dual(y, dual_y);
                outer_product_hardcoded(dual_x, dual_y, prod);
                equi_dual(prod, final_out);

                for (int k = 0; k < 16; k++)
                    data[k][i][j] = final_out[k];
            }
        }
    }

    inline void equi_dual(const float* in, float* out) {
        const int perm[16]   = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        const float sign[16] = {1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1};

        for (int i = 0; i < 16; i++)
            out[i] = sign[i] * in[perm[i]];
    }

    inline void outer_product_hardcoded(const float* x, const float* y, float* out) {
        for (int i = 0; i < 16; i++)
            out[i] = 0.0f;

        for (size_t idx = 0; idx < kOpBasisEntryCount; idx++) {
            const SparseEntry& e = kOpBasisEntries[idx];
            out[e.i] += e.v * x[e.j] * y[e.k];
        }
    }
};

inline const JoinKernel KERNEL;

}  // namespace turbogator
