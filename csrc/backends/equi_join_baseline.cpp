#include "ops.hpp"

/*

x = [a, b, c, d, ... n, o, p]
equi_dual(x)
    perm = [15, 14, 13, ... 0]
    sign = [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1]
    return x = [p, -o, n, ... d, -c, b, a]

*/

namespace turbogator
{

    struct JoinKernel
    {
        float data[16][16][16];

        JoinKernel()
        {
            for (size_t i = 0; i < 16; ++i)
                for (size_t j = 0; j < 16; ++j)
                    for (size_t k = 0; k < 16; ++k)
                        data[i][j][k] = 0.0f;

            const float dualisation_sign[16] = {1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1};

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    float x[16] = {0};
                    float y[16] = {0};
                    x[i] = 1.0f;
                    y[j] = 1.0f;
                }
            }
        }
    };

    void equi_join_baseline(const float *a, const float *b, const float *ref, float *out, size_t n)
    {
        // this breaks if D != 16
        (void)ref;
        for (size_t i = 0; i < n; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }

} // namespace turbogator
