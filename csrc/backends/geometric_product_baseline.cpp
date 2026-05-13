#include "ops.hpp"

namespace turbogator
{

    // just does einsum with the fixed kernel
    void geometric_product_baseline(const float *a, const float *b, float *out, size_t n)
    {
        float GP_BASIS[16][16][16] = {};
        // i did this to hide the code block lol
        if (1)
        {
            GP_BASIS[0][0][0] = 1.0f;
            GP_BASIS[0][2][2] = 1.0f;
            GP_BASIS[0][3][3] = 1.0f;
            GP_BASIS[0][4][4] = 1.0f;
            GP_BASIS[0][8][8] = -1.0f;
            GP_BASIS[0][9][9] = -1.0f;
            GP_BASIS[0][10][10] = -1.0f;
            GP_BASIS[0][14][14] = -1.0f;
            GP_BASIS[1][0][1] = 1.0f;
            GP_BASIS[1][1][0] = 1.0f;
            GP_BASIS[1][2][5] = -1.0f;
            GP_BASIS[1][3][6] = -1.0f;
            GP_BASIS[1][4][7] = -1.0f;
            GP_BASIS[1][5][2] = 1.0f;
            GP_BASIS[1][6][3] = 1.0f;
            GP_BASIS[1][7][4] = 1.0f;
            GP_BASIS[1][8][11] = -1.0f;
            GP_BASIS[1][9][12] = -1.0f;
            GP_BASIS[1][10][13] = -1.0f;
            GP_BASIS[1][11][8] = -1.0f;
            GP_BASIS[1][12][9] = -1.0f;
            GP_BASIS[1][13][10] = -1.0f;
            GP_BASIS[1][14][15] = 1.0f;
            GP_BASIS[1][15][14] = -1.0f;
            GP_BASIS[2][0][2] = 1.0f;
            GP_BASIS[2][2][0] = 1.0f;
            GP_BASIS[2][3][8] = -1.0f;
            GP_BASIS[2][4][9] = -1.0f;
            GP_BASIS[2][8][3] = 1.0f;
            GP_BASIS[2][9][4] = 1.0f;
            GP_BASIS[2][10][14] = -1.0f;
            GP_BASIS[2][14][10] = -1.0f;
            GP_BASIS[3][0][3] = 1.0f;
            GP_BASIS[3][2][8] = 1.0f;
            GP_BASIS[3][3][0] = 1.0f;
            GP_BASIS[3][4][10] = -1.0f;
            GP_BASIS[3][8][2] = -1.0f;
            GP_BASIS[3][9][14] = 1.0f;
            GP_BASIS[3][10][4] = 1.0f;
            GP_BASIS[3][14][9] = 1.0f;
            GP_BASIS[4][0][4] = 1.0f;
            GP_BASIS[4][2][9] = 1.0f;
            GP_BASIS[4][3][10] = 1.0f;
            GP_BASIS[4][4][0] = 1.0f;
            GP_BASIS[4][8][14] = -1.0f;
            GP_BASIS[4][9][2] = -1.0f;
            GP_BASIS[4][10][3] = -1.0f;
            GP_BASIS[4][14][8] = -1.0f;
            GP_BASIS[5][0][5] = 1.0f;
            GP_BASIS[5][1][2] = 1.0f;
            GP_BASIS[5][2][1] = -1.0f;
            GP_BASIS[5][3][11] = 1.0f;
            GP_BASIS[5][4][12] = 1.0f;
            GP_BASIS[5][5][0] = 1.0f;
            GP_BASIS[5][6][8] = -1.0f;
            GP_BASIS[5][7][9] = -1.0f;
            GP_BASIS[5][8][6] = 1.0f;
            GP_BASIS[5][9][7] = 1.0f;
            GP_BASIS[5][10][15] = -1.0f;
            GP_BASIS[5][11][3] = 1.0f;
            GP_BASIS[5][12][4] = 1.0f;
            GP_BASIS[5][13][14] = -1.0f;
            GP_BASIS[5][14][13] = 1.0f;
            GP_BASIS[5][15][10] = -1.0f;
            GP_BASIS[6][0][6] = 1.0f;
            GP_BASIS[6][1][3] = 1.0f;
            GP_BASIS[6][2][11] = -1.0f;
            GP_BASIS[6][3][1] = -1.0f;
            GP_BASIS[6][4][13] = 1.0f;
            GP_BASIS[6][5][8] = 1.0f;
            GP_BASIS[6][6][0] = 1.0f;
            GP_BASIS[6][7][10] = -1.0f;
            GP_BASIS[6][8][5] = -1.0f;
            GP_BASIS[6][9][15] = 1.0f;
            GP_BASIS[6][10][7] = 1.0f;
            GP_BASIS[6][11][2] = -1.0f;
            GP_BASIS[6][12][14] = 1.0f;
            GP_BASIS[6][13][4] = 1.0f;
            GP_BASIS[6][14][12] = -1.0f;
            GP_BASIS[6][15][9] = 1.0f;
            GP_BASIS[7][0][7] = 1.0f;
            GP_BASIS[7][1][4] = 1.0f;
            GP_BASIS[7][2][12] = -1.0f;
            GP_BASIS[7][3][13] = -1.0f;
            GP_BASIS[7][4][1] = -1.0f;
            GP_BASIS[7][5][9] = 1.0f;
            GP_BASIS[7][6][10] = 1.0f;
            GP_BASIS[7][7][0] = 1.0f;
            GP_BASIS[7][8][15] = -1.0f;
            GP_BASIS[7][9][5] = -1.0f;
            GP_BASIS[7][10][6] = -1.0f;
            GP_BASIS[7][11][14] = -1.0f;
            GP_BASIS[7][12][2] = -1.0f;
            GP_BASIS[7][13][3] = -1.0f;
            GP_BASIS[7][14][11] = 1.0f;
            GP_BASIS[7][15][8] = -1.0f;
            GP_BASIS[8][0][8] = 1.0f;
            GP_BASIS[8][2][3] = 1.0f;
            GP_BASIS[8][3][2] = -1.0f;
            GP_BASIS[8][4][14] = 1.0f;
            GP_BASIS[8][8][0] = 1.0f;
            GP_BASIS[8][9][10] = -1.0f;
            GP_BASIS[8][10][9] = 1.0f;
            GP_BASIS[8][14][4] = 1.0f;
            GP_BASIS[9][0][9] = 1.0f;
            GP_BASIS[9][2][4] = 1.0f;
            GP_BASIS[9][3][14] = -1.0f;
            GP_BASIS[9][4][2] = -1.0f;
            GP_BASIS[9][8][10] = 1.0f;
            GP_BASIS[9][9][0] = 1.0f;
            GP_BASIS[9][10][8] = -1.0f;
            GP_BASIS[9][14][3] = -1.0f;
            GP_BASIS[10][0][10] = 1.0f;
            GP_BASIS[10][2][14] = 1.0f;
            GP_BASIS[10][3][4] = 1.0f;
            GP_BASIS[10][4][3] = -1.0f;
            GP_BASIS[10][8][9] = -1.0f;
            GP_BASIS[10][9][8] = 1.0f;
            GP_BASIS[10][10][0] = 1.0f;
            GP_BASIS[10][14][2] = 1.0f;
            GP_BASIS[11][0][11] = 1.0f;
            GP_BASIS[11][1][8] = 1.0f;
            GP_BASIS[11][2][6] = -1.0f;
            GP_BASIS[11][3][5] = 1.0f;
            GP_BASIS[11][4][15] = -1.0f;
            GP_BASIS[11][5][3] = 1.0f;
            GP_BASIS[11][6][2] = -1.0f;
            GP_BASIS[11][7][14] = 1.0f;
            GP_BASIS[11][8][1] = 1.0f;
            GP_BASIS[11][9][13] = -1.0f;
            GP_BASIS[11][10][12] = 1.0f;
            GP_BASIS[11][11][0] = 1.0f;
            GP_BASIS[11][12][10] = -1.0f;
            GP_BASIS[11][13][9] = 1.0f;
            GP_BASIS[11][14][7] = -1.0f;
            GP_BASIS[11][15][4] = 1.0f;
            GP_BASIS[12][0][12] = 1.0f;
            GP_BASIS[12][1][9] = 1.0f;
            GP_BASIS[12][2][7] = -1.0f;
            GP_BASIS[12][3][15] = 1.0f;
            GP_BASIS[12][4][5] = 1.0f;
            GP_BASIS[12][5][4] = 1.0f;
            GP_BASIS[12][6][14] = -1.0f;
            GP_BASIS[12][7][2] = -1.0f;
            GP_BASIS[12][8][13] = 1.0f;
            GP_BASIS[12][9][1] = 1.0f;
            GP_BASIS[12][10][11] = -1.0f;
            GP_BASIS[12][11][10] = 1.0f;
            GP_BASIS[12][12][0] = 1.0f;
            GP_BASIS[12][13][8] = -1.0f;
            GP_BASIS[12][14][6] = 1.0f;
            GP_BASIS[12][15][3] = -1.0f;
            GP_BASIS[13][0][13] = 1.0f;
            GP_BASIS[13][1][10] = 1.0f;
            GP_BASIS[13][2][15] = -1.0f;
            GP_BASIS[13][3][7] = -1.0f;
            GP_BASIS[13][4][6] = 1.0f;
            GP_BASIS[13][5][14] = 1.0f;
            GP_BASIS[13][6][4] = 1.0f;
            GP_BASIS[13][7][3] = -1.0f;
            GP_BASIS[13][8][12] = -1.0f;
            GP_BASIS[13][9][11] = 1.0f;
            GP_BASIS[13][10][1] = 1.0f;
            GP_BASIS[13][11][9] = -1.0f;
            GP_BASIS[13][12][8] = 1.0f;
            GP_BASIS[13][13][0] = 1.0f;
            GP_BASIS[13][14][5] = -1.0f;
            GP_BASIS[13][15][2] = 1.0f;
            GP_BASIS[14][0][14] = 1.0f;
            GP_BASIS[14][2][10] = 1.0f;
            GP_BASIS[14][3][9] = -1.0f;
            GP_BASIS[14][4][8] = 1.0f;
            GP_BASIS[14][8][4] = 1.0f;
            GP_BASIS[14][9][3] = -1.0f;
            GP_BASIS[14][10][2] = 1.0f;
            GP_BASIS[14][14][0] = 1.0f;
            GP_BASIS[15][0][15] = 1.0f;
            GP_BASIS[15][1][14] = 1.0f;
            GP_BASIS[15][2][13] = -1.0f;
            GP_BASIS[15][3][12] = 1.0f;
            GP_BASIS[15][4][11] = -1.0f;
            GP_BASIS[15][5][10] = 1.0f;
            GP_BASIS[15][6][9] = -1.0f;
            GP_BASIS[15][7][8] = 1.0f;
            GP_BASIS[15][8][7] = 1.0f;
            GP_BASIS[15][9][6] = -1.0f;
            GP_BASIS[15][10][5] = 1.0f;
            GP_BASIS[15][11][4] = 1.0f;
            GP_BASIS[15][12][3] = -1.0f;
            GP_BASIS[15][13][2] = 1.0f;
            GP_BASIS[15][14][1] = -1.0f;
            GP_BASIS[15][15][0] = 1.0f;
        }

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
                        cur_out[i] += GP_BASIS[i][j][k] * cur_a[j] * cur_b[k];
        }
    }

} // namespace turbogator
