#include "ops.hpp"

/*
// actually this math can be ignored, we can just hardcode the kernel
x = [a, b, c, d, ... n, o, p]
equi_dual(x)
    perm = [15, 14, 13, ... 0]
    sign = [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1]
    return x = [p, -o, n, ... d, -c, b, a]

outer_product(x, y)
    op_basis = [16x16x16 sparse]
    for i in 0..15:
       for j in 0..15:
          for k in 0..15:
             out[i] += op_basis[i][j][k] * x[j] * y[k]

compute_join_kernel() ...
equi_join(x,)
    kernel = [16x16x16]
    for i in 0..15:
       for j in 0..15:
          for k in 0..15:
                out[i] += kernel[i][j][k] * x[j] * y[k]

    scale by ref[14]
*/

namespace turbogator
{

    // incredibly sparse matrix
    // 16x16x16 = 4096 floats
    // 81 nonzero
    // 98% sparsity

    // IMPORTANT: see random/join_kernel.txt
    struct JoinKernel
    {
        // OPTIMISATION TODO:
        // this is a constant kernel, can be hardcoded using one of the sparse formats
        // it's also just -1,0,1
        float data[16][16][16] = {};

        JoinKernel()
        {
            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    float x[16] = {0};
                    float y[16] = {0};
                    x[i] = 1.0f;
                    y[j] = 1.0f;

                    float dual_x[16];
                    float dual_y[16];
                    float prod[16];
                    float final_out[16];

                    equi_dual(x, dual_x);
                    equi_dual(y, dual_y);
                    outer_prod(dual_x, dual_y, prod);
                    equi_dual(prod, final_out);

                    for (int k = 0; k < 16; k++)
                        data[k][i][j] = final_out[k];
                }
            }
        }

        inline void equi_dual(const float *in, float *out)
        {
            const int perm[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
            const float sign[16] = {1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1};

            for (int i = 0; i < 16; i++)
                out[i] = sign[i] * in[perm[i]];
        }

        inline void outer_prod(const float *x, const float *y, float *out)
        {
            // op bilinear basis, see random/
            float OP_BASIS[16][16][16] = {};
            // i did this to hide the code block lol
            if (1)
            {

                OP_BASIS[0][0][0] = 1.0f;
                OP_BASIS[1][0][1] = 1.0f;
                OP_BASIS[1][1][0] = 1.0f;
                OP_BASIS[2][0][2] = 1.0f;
                OP_BASIS[2][2][0] = 1.0f;
                OP_BASIS[3][0][3] = 1.0f;
                OP_BASIS[3][3][0] = 1.0f;
                OP_BASIS[4][0][4] = 1.0f;
                OP_BASIS[4][4][0] = 1.0f;
                OP_BASIS[5][0][5] = 1.0f;
                OP_BASIS[5][1][2] = 1.0f;
                OP_BASIS[5][2][1] = -1.0f;
                OP_BASIS[5][5][0] = 1.0f;
                OP_BASIS[6][0][6] = 1.0f;
                OP_BASIS[6][1][3] = 1.0f;
                OP_BASIS[6][3][1] = -1.0f;
                OP_BASIS[6][6][0] = 1.0f;
                OP_BASIS[7][0][7] = 1.0f;
                OP_BASIS[7][1][4] = 1.0f;
                OP_BASIS[7][4][1] = -1.0f;
                OP_BASIS[7][7][0] = 1.0f;
                OP_BASIS[8][0][8] = 1.0f;
                OP_BASIS[8][2][3] = 1.0f;
                OP_BASIS[8][3][2] = -1.0f;
                OP_BASIS[8][8][0] = 1.0f;
                OP_BASIS[9][0][9] = 1.0f;
                OP_BASIS[9][2][4] = 1.0f;
                OP_BASIS[9][4][2] = -1.0f;
                OP_BASIS[9][9][0] = 1.0f;
                OP_BASIS[10][0][10] = 1.0f;
                OP_BASIS[10][3][4] = 1.0f;
                OP_BASIS[10][4][3] = -1.0f;
                OP_BASIS[10][10][0] = 1.0f;
                OP_BASIS[11][0][11] = 1.0f;
                OP_BASIS[11][1][8] = 1.0f;
                OP_BASIS[11][2][6] = -1.0f;
                OP_BASIS[11][3][5] = 1.0f;
                OP_BASIS[11][5][3] = 1.0f;
                OP_BASIS[11][6][2] = -1.0f;
                OP_BASIS[11][8][1] = 1.0f;
                OP_BASIS[11][11][0] = 1.0f;
                OP_BASIS[12][0][12] = 1.0f;
                OP_BASIS[12][1][9] = 1.0f;
                OP_BASIS[12][2][7] = -1.0f;
                OP_BASIS[12][4][5] = 1.0f;
                OP_BASIS[12][5][4] = 1.0f;
                OP_BASIS[12][7][2] = -1.0f;
                OP_BASIS[12][9][1] = 1.0f;
                OP_BASIS[12][12][0] = 1.0f;
                OP_BASIS[13][0][13] = 1.0f;
                OP_BASIS[13][1][10] = 1.0f;
                OP_BASIS[13][3][7] = -1.0f;
                OP_BASIS[13][4][6] = 1.0f;
                OP_BASIS[13][6][4] = 1.0f;
                OP_BASIS[13][7][3] = -1.0f;
                OP_BASIS[13][10][1] = 1.0f;
                OP_BASIS[13][13][0] = 1.0f;
                OP_BASIS[14][0][14] = 1.0f;
                OP_BASIS[14][2][10] = 1.0f;
                OP_BASIS[14][3][9] = -1.0f;
                OP_BASIS[14][4][8] = 1.0f;
                OP_BASIS[14][8][4] = 1.0f;
                OP_BASIS[14][9][3] = -1.0f;
                OP_BASIS[14][10][2] = 1.0f;
                OP_BASIS[14][14][0] = 1.0f;
                OP_BASIS[15][0][15] = 1.0f;
                OP_BASIS[15][1][14] = 1.0f;
                OP_BASIS[15][2][13] = -1.0f;
                OP_BASIS[15][3][12] = 1.0f;
                OP_BASIS[15][4][11] = -1.0f;
                OP_BASIS[15][5][10] = 1.0f;
                OP_BASIS[15][6][9] = -1.0f;
                OP_BASIS[15][7][8] = 1.0f;
                OP_BASIS[15][8][7] = 1.0f;
                OP_BASIS[15][9][6] = -1.0f;
                OP_BASIS[15][10][5] = 1.0f;
                OP_BASIS[15][11][4] = 1.0f;
                OP_BASIS[15][12][3] = -1.0f;
                OP_BASIS[15][13][2] = 1.0f;
                OP_BASIS[15][14][1] = -1.0f;
                OP_BASIS[15][15][0] = 1.0f;
            }

            for (int i = 0; i < 16; i++)
                out[i] = 0.0f;

            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    for (int k = 0; k < 16; k++)
                        out[i] += OP_BASIS[i][j][k] * x[j] * y[k];
        }
    };

    inline static const JoinKernel KERNEL;
    // this breaks if D != 16 !!!!
    void equi_join_baseline(const float *a, const float *b, const float *ref, float *out, size_t n)
    {
        // ret = torch.einsum("ijk, ...j, ...k -> ...i", kernel, x, y)
        // if reference is not None:
        //     ret *= reference[..., [14]]
        // return ret

        for (size_t batch = 0; batch < n; batch++)
        {
            // curr multivector
            const float *cur_a = a + (batch * 16);
            const float *cur_b = b + (batch * 16);
            float *cur_out = out + (batch * 16);

            for (int i = 0; i < 16; i++)
                cur_out[i] = 0.0f;

            // OPTIMISATION TODO:
            // actual potentials for optimisations HERE
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    for (int k = 0; k < 16; k++)
                        cur_out[i] += KERNEL.data[i][j][k] * cur_a[j] * cur_b[k];

            // scale by ref?
            if (ref != nullptr)
            {
                const float *cur_ref = ref + (batch * 16);
                float scale = cur_ref[14];

                for (int i = 0; i < 16; i++)
                    cur_out[i] *= scale;
            }
        }
    }

} // namespace turbogator
