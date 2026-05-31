#include <immintrin.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "ops.hpp"

namespace turbogator {

static constexpr int N_BLADES  = 16;
static constexpr int N_IPA     = 7;
static constexpr int N_DAA     = 5;
static constexpr float DAA_EPS = 1e-3f;
static constexpr int64_t BT    = 32;  // token tile: 32x32x4 = 4 KiB scores tile
// static constexpr int64_t BD    = 288;  // feature tile: 32x288x4 = 36 KiB K tile fits in P-core L1d (48 KiB)

// original vec_softmax better at T < 256
// then flash attention better
static constexpr int64_t FLASH_T_THRESHOLD = 256;

static constexpr int IPA_IDX[N_IPA] = {0, 2, 3, 4, 8, 9, 10};
static constexpr int TRI_IDX[4]     = {11, 12, 13, 14};

static constexpr int TILE_MR = 4;
static constexpr int TILE_NR = 2;

__attribute__((always_inline)) static inline void project_ipa(const float* __restrict__ mv, float* __restrict__ out) {
    for (int i = 0; i < N_IPA; ++i)
        out[i] = mv[IPA_IDX[i]];
}

__attribute__((always_inline)) static inline void normalize_tri(const float* __restrict__ mv, float* __restrict__ r) {
    float r3   = mv[TRI_IDX[3]];
    float norm = r3 / (r3 * r3 + DAA_EPS);
    for (int i = 0; i < 4; ++i)
        r[i] = mv[TRI_IDX[i]] * norm;
}

__attribute__((always_inline)) static inline void project_daa_bq(const float* __restrict__ mv,
                                                                 float* __restrict__ out) {
    float r[4];
    normalize_tri(mv, r);
    out[0] = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    out[1] = r[3] * r[3];
    out[2] = r[0] * r[3];
    out[3] = r[1] * r[3];
    out[4] = r[2] * r[3];
}

__attribute__((always_inline)) static inline void project_daa_bk(const float* __restrict__ mv,
                                                                 float* __restrict__ out) {
    float r[4];
    normalize_tri(mv, r);
    out[0] = -(r[3] * r[3]);
    out[1] = -(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    out[2] = 2.f * r[0] * r[3];
    out[3] = 2.f * r[1] * r[3];
    out[4] = 2.f * r[2] * r[3];
}

// Quad-accumulator AVX FMA dot product.
// 4 independent accumulators fully hide the 4-cycle FMA latency: each acc
template <int MR, int NR>
static inline void mkernel(const float* __restrict__ A,
                           int64_t lda,
                           const float* __restrict__ B,
                           int64_t ldb,
                           float* __restrict__ C,
                           int64_t ldc,
                           int64_t K) {
    __m256 c[MR][NR];
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NR; ++j)
            c[i][j] = _mm256_setzero_ps();

    for (int64_t k = 0; k < K; ++k) {
        const float* brow = B + k * ldb;
        __m256 bv[NR];
#pragma GCC unroll 8
        for (int j = 0; j < NR; ++j)
            bv[j] = _mm256_loadu_ps(brow + j * 8);
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            __m256 a = _mm256_broadcast_ss(A + (int64_t)i * lda + k);
#pragma GCC unroll 8
            for (int j = 0; j < NR; ++j)
                c[i][j] = _mm256_fmadd_ps(a, bv[j], c[i][j]);
        }
    }
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NR; ++j)
            _mm256_storeu_ps(C + (int64_t)i * ldc + j * 8, c[i][j]);
}

template <int MR, int NR>
static void tiled_mm(
    const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc, int64_t M, int64_t N, int64_t K) {
    constexpr int NB    = NR * 8;
    const int64_t Mfull = M - (M % MR);
    const int64_t Nfull = N - (N % NB);

    for (int64_t i0 = 0; i0 < Mfull; i0 += MR)
        for (int64_t j0 = 0; j0 < Nfull; j0 += NB)
            mkernel<MR, NR>(A + i0 * lda, lda, B + j0, ldb, C + i0 * ldc + j0, ldc, K);

    auto cell = [&](int64_t i, int64_t j) {
        float acc = 0.f;
        for (int64_t k = 0; k < K; ++k)
            acc += A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] = acc;
    };
    for (int64_t i = 0; i < Mfull; ++i)
        for (int64_t j = Nfull; j < N; ++j)
            cell(i, j);
    for (int64_t i = Mfull; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
            cell(i, j);
}

template <int MR, int NR>
static inline void mkernel_accum(const float* __restrict__ A,
                                 int64_t lda,
                                 const float* __restrict__ B,
                                 int64_t ldb,
                                 float* __restrict__ C,
                                 int64_t ldc,
                                 int64_t K,
                                 const float* __restrict__ row_scale) {
    __m256 c[MR][NR];
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i) {
        __m256 vs = _mm256_set1_ps(row_scale[i]);
        for (int j = 0; j < NR; ++j)
            c[i][j] = _mm256_mul_ps(_mm256_loadu_ps(C + (int64_t)i * ldc + j * 8), vs);
    }
    for (int64_t k = 0; k < K; ++k) {
        const float* brow = B + k * ldb;
        __m256 bv[NR];
#pragma GCC unroll 8
        for (int j = 0; j < NR; ++j)
            bv[j] = _mm256_loadu_ps(brow + j * 8);
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            __m256 a = _mm256_broadcast_ss(A + (int64_t)i * lda + k);
#pragma GCC unroll 8
            for (int j = 0; j < NR; ++j)
                c[i][j] = _mm256_fmadd_ps(a, bv[j], c[i][j]);
        }
    }
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NR; ++j)
            _mm256_storeu_ps(C + (int64_t)i * ldc + j * 8, c[i][j]);
}

template <int MR, int NR>
static void tiled_mm_accum(const float* A,
                           int64_t lda,
                           const float* B,
                           int64_t ldb,
                           float* C,
                           int64_t ldc,
                           int64_t M,
                           int64_t N,
                           int64_t K,
                           const float* row_scale) {
    constexpr int NB    = NR * 8;
    const int64_t Mfull = M - (M % MR);
    const int64_t Nfull = N - (N % NB);

    for (int64_t i0 = 0; i0 < Mfull; i0 += MR)
        for (int64_t j0 = 0; j0 < Nfull; j0 += NB)
            mkernel_accum<MR, NR>(A + i0 * lda, lda, B + j0, ldb, C + i0 * ldc + j0, ldc, K, row_scale + i0);

    auto cell = [&](int64_t i, int64_t j) {
        float acc = C[i * ldc + j] * row_scale[i];
        for (int64_t k = 0; k < K; ++k)
            acc += A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] = acc;
    };
    for (int64_t i = 0; i < Mfull; ++i)
        for (int64_t j = Nfull; j < N; ++j)
            cell(i, j);
    for (int64_t i = Mfull; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
            cell(i, j);
}

__attribute__((always_inline)) static inline float hmax256(__m256 v) {
    // 256 -> 128
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow         = _mm_max_ps(vlow, vhigh);
    // 128 -> 64
    vhigh = _mm_movehl_ps(vlow, vlow);
    vlow  = _mm_max_ps(vlow, vhigh);
    // 64 -> 32
    vhigh = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 1, 1, 1));
    return _mm_cvtss_f32(_mm_max_ss(vlow, vhigh));
}

__attribute__((always_inline)) static inline float hsum256(__m256 v) {
    // 256 -> 128
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow         = _mm_add_ps(vlow, vhigh);
    // 128 -> 64
    vhigh = _mm_movehl_ps(vlow, vlow);
    vlow  = _mm_add_ps(vlow, vhigh);
    // 64 -> 32
    vhigh = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 1, 1, 1));
    return _mm_cvtss_f32(_mm_add_ss(vlow, vhigh));
}

// funky e^x
__attribute__((always_inline)) static inline __m256 fast_exp_ps(__m256 x) {
    // clamp for underflowflow e^x = 0
    // -87.33654f < x <= 0.
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.33654f));

    // integer part n =  x * log2(e)
    __m256 y       = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504f));
    __m256i n      = _mm256_cvtps_epi32(y);
    __m256 float_n = _mm256_cvtepi32_ps(n);

    //  fractional part f = x - n * ln(2)
    x        = _mm256_fnmadd_ps(float_n, _mm256_set1_ps(0.69314575f), x);
    __m256 f = _mm256_fnmadd_ps(float_n, _mm256_set1_ps(1.4286068e-06f), x);

    // Horner's method
    // p(f) = 1 + f + f^2/2 + f^3/6 + f^4/24
    __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(0.04166667f), f, _mm256_set1_ps(0.16666667f));
    p        = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.5f));
    p        = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    p        = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));

    // 2^n
    __m256i exp_bits = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    __m256 two_to_n  = _mm256_castsi256_ps(exp_bits);

    // e^x = 2^n * e^f
    return _mm256_mul_ps(p, two_to_n);
}

#define SOFT_MAX_UNROLL 4
// Online Softmax
static inline void vec_softmax(float* scores,
                               int64_t T,
                               __m256 v_scale,
                               const float* attn_mask,
                               bool is_causal,
                               int64_t b,
                               int64_t H,
                               int64_t h,
                               __m256 v_neg_inf) {
    const int8_t VLEN = 256 / 32;
    const int STEP    = SOFT_MAX_UNROLL * VLEN;

    for (int64_t t1 = 0; t1 < T; ++t1) {
        float* row = scores + t1 * T;
        __m256 v_max[SOFT_MAX_UNROLL];
        __m256 v_sum[SOFT_MAX_UNROLL];

#pragma GCC unroll 4
        for (int i = 0; i < SOFT_MAX_UNROLL; ++i) {
            v_max[i] = v_neg_inf;
            v_sum[i] = _mm256_setzero_ps();
        }

        /// find max + SUM !!
        for (int64_t t2 = 0; t2 < T; t2 += STEP) {
#pragma GCC unroll 4
            for (int i = 0; i < SOFT_MAX_UNROLL; ++i) {
                int offset = i * VLEN;
                __m256 v_s = _mm256_loadu_ps(row + t2 + offset);
                v_s        = _mm256_mul_ps(v_s, v_scale);

                if (is_causal) {
                    __m256 v_t2 = _mm256_set_ps(t2 + offset + 7,
                                                t2 + offset + 6,
                                                t2 + offset + 5,
                                                t2 + offset + 4,
                                                t2 + offset + 3,
                                                t2 + offset + 2,
                                                t2 + offset + 1,
                                                t2 + offset);
                    __m256 v_t1 = _mm256_set1_ps((float)t1);
                    __m256 mask = _mm256_cmp_ps(v_t2, v_t1, _CMP_GT_OQ);
                    v_s         = _mm256_blendv_ps(v_s, v_neg_inf, mask);
                }

                _mm256_storeu_ps(row + t2 + offset, v_s);

                __m256 v_new_max    = _mm256_max_ps(v_max[i], v_s);
                __m256 v_correction = fast_exp_ps(_mm256_sub_ps(v_max[i], v_new_max));
                v_sum[i]            = _mm256_mul_ps(v_sum[i], v_correction);

                __m256 v_exp_s = fast_exp_ps(_mm256_sub_ps(v_s, v_new_max));
                v_sum[i]       = _mm256_add_ps(v_sum[i], v_exp_s);
                v_max[i]       = v_new_max;
            }
        }

        __m256 v_merged_max = v_max[0];
#pragma GCC unroll 4
        for (int i = 1; i < SOFT_MAX_UNROLL; ++i) {
            v_merged_max = _mm256_max_ps(v_merged_max, v_max[i]);
        }

        __m256 v_merged_sum = _mm256_setzero_ps();
#pragma GCC unroll 4
        for (int i = 0; i < SOFT_MAX_UNROLL; ++i) {
            __m256 v_corr = fast_exp_ps(_mm256_sub_ps(v_max[i], v_merged_max));
            v_merged_sum  = _mm256_add_ps(v_merged_sum, _mm256_mul_ps(v_sum[i], v_corr));
        }

        float global_max    = hmax256(v_merged_max);
        __m256 v_global_max = _mm256_set1_ps(global_max);

        __m256 v_sum_corr = fast_exp_ps(_mm256_sub_ps(v_merged_max, v_global_max));
        v_merged_sum      = _mm256_mul_ps(v_merged_sum, v_sum_corr);

        float total_sum  = hsum256(v_merged_sum);
        float inv_sum    = 1.0f / total_sum;
        __m256 v_inv_sum = _mm256_set1_ps(inv_sum);

        /// mult inv
        for (int64_t t2 = 0; t2 < T; t2 += STEP) {
#pragma GCC unroll 4
            for (int i = 0; i < SOFT_MAX_UNROLL; ++i) {
                int offset = i * VLEN;
                __m256 v_s = _mm256_loadu_ps(row + t2 + offset);

                v_s = _mm256_sub_ps(v_s, v_global_max);
                v_s = fast_exp_ps(v_s);
                v_s = _mm256_mul_ps(v_s, v_inv_sum);

                _mm256_storeu_ps(row + t2 + offset, v_s);
            }
        }
    }
}

static inline void scalar_softmax(float* scores,
                                  int64_t T,
                                  float scale,
                                  const float* attn_mask,
                                  bool is_causal,
                                  int64_t b,
                                  int64_t H,
                                  int64_t h,
                                  float neg_inf) {
    for (int64_t t1 = 0; t1 < T; ++t1) {
        float row_max = neg_inf;
        for (int64_t t2 = 0; t2 < T; ++t2) {
            float s;
            if (is_causal && t2 > t1) {
                s = neg_inf;
            } else {
                s = scores[t1 * T + t2] * scale;
                if (attn_mask)
                    s += attn_mask[((b * H + h) * T + t1) * T + t2];
            }
            scores[t1 * T + t2] = s;
            if (s > row_max)
                row_max = s;
        }
        float sum_exp = 0.f;
        for (int64_t t2 = 0; t2 < T; ++t2) {
            float e             = std::exp(scores[t1 * T + t2] - row_max);
            scores[t1 * T + t2] = e;
            sum_exp += e;
        }
        for (int64_t t2 = 0; t2 < T; ++t2)
            scores[t1 * T + t2] /= sum_exp;
    }
}

// tilled flash attention for a single (batch, head ) pair
// avoids full TxT scores by keeping BTxBT tiles in L1
__attribute__((always_inline)) static inline void flash_attention_bh(const float* __restrict__ q_flat,
                                                                     const float* __restrict__ kt_flat,
                                                                     const float* __restrict__ v_bh,
                                                                     float* __restrict__ o_bh,
                                                                     int64_t T,
                                                                     int64_t qk_dim,
                                                                     int64_t mv_stride,
                                                                     int64_t out_t_stride,
                                                                     int64_t qs_t,
                                                                     __m256 v_scale,
                                                                     __m256 v_neg_inf,
                                                                     float scalar_neg_inf,
                                                                     bool is_causal,
                                                                     float* __restrict__ score_tile,
                                                                     float* __restrict__ row_max_fa,
                                                                     float* __restrict__ row_sum_fa) {
    alignas(32) float corrections[BT];

    for (int64_t t1s = 0; t1s < T; t1s += BT) {
        for (int r = 0; r < BT; ++r) {
            row_max_fa[r] = scalar_neg_inf;
            row_sum_fa[r] = 0.f;
        }

        float* o_qt = o_bh + t1s * out_t_stride;
        for (int r = 0; r < BT; ++r) {
            float* orow = o_qt + r * out_t_stride;
            for (int64_t d = 0; d < mv_stride; d += 8)
                _mm256_storeu_ps(orow + d, _mm256_setzero_ps());
        }

        const int64_t t2_lim = is_causal ? (t1s + BT) : T;
        for (int64_t t2s = 0; t2s < t2_lim; t2s += BT) {
            // QK^T
            tiled_mm<TILE_MR, TILE_NR>(q_flat + t1s * qk_dim, qk_dim, kt_flat + t2s, T, score_tile, BT, BT, BT, qk_dim);

            // diag tile mask (t2 > t1 -> -inf)
            const bool diag = is_causal && (t2s == t1s);

            for (int r = 0; r < BT; ++r) {
                float* srow = score_tile + r * BT;

                __m256 vm = v_neg_inf;
                for (int c = 0; c < BT; c += 8) {
                    __m256 sv = _mm256_mul_ps(_mm256_loadu_ps(srow + c), v_scale);
                    if (diag) {
                        __m256 vt2  = _mm256_set_ps((float)(t2s + c + 7),
                                                    (float)(t2s + c + 6),
                                                    (float)(t2s + c + 5),
                                                    (float)(t2s + c + 4),
                                                    (float)(t2s + c + 3),
                                                    (float)(t2s + c + 2),
                                                    (float)(t2s + c + 1),
                                                    (float)(t2s + c + 0));
                        __m256 mask = _mm256_cmp_ps(vt2, _mm256_set1_ps((float)(t1s + r)), _CMP_GT_OQ);
                        sv          = _mm256_blendv_ps(sv, v_neg_inf, mask);
                    }
                    _mm256_storeu_ps(srow + c, sv);
                    vm = _mm256_max_ps(vm, sv);
                }
                float tile_max_r = hmax256(vm);

                float old_max    = row_max_fa[r];
                float new_max    = old_max > tile_max_r ? old_max : tile_max_r;
                float correction = (old_max >= new_max) ? 1.f : std::exp(old_max - new_max);
                corrections[r]   = correction;
                row_sum_fa[r] *= correction;
                row_max_fa[r] = new_max;

                __m256 vmax_r = _mm256_set1_ps(new_max);
                __m256 vsum   = _mm256_setzero_ps();
                for (int c = 0; c < BT; c += 8) {
                    __m256 sv = fast_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(srow + c), vmax_r));
                    _mm256_storeu_ps(srow + c, sv);
                    vsum = _mm256_add_ps(vsum, sv);
                }
                row_sum_fa[r] += hsum256(vsum);
            }

            tiled_mm_accum<TILE_MR, TILE_NR>(
                score_tile, BT, v_bh + t2s * qs_t, qs_t, o_qt, out_t_stride, BT, mv_stride, BT, corrections);
        }

        // norm rows
        for (int r = 0; r < BT; ++r) {
            float inv_sum = 1.f / row_sum_fa[r];
            float* orow   = o_qt + r * out_t_stride;
            __m256 vinv   = _mm256_set1_ps(inv_sum);
            for (int64_t d = 0; d < mv_stride; d += 8)
                _mm256_storeu_ps(orow + d, _mm256_mul_ps(_mm256_loadu_ps(orow + d), vinv));
        }
    }
}

void equi_geometric_attention_vectorized(const float* __restrict__ q,
                                         const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         float* __restrict__ out,
                                         int64_t B,
                                         int64_t H,
                                         int64_t T,
                                         int64_t C,
                                         int64_t qs_b,
                                         int64_t qs_h,
                                         int64_t qs_t,
                                         const std::vector<std::string>& kinds,
                                         const std::vector<const float*>& weights,
                                         const float* attn_mask,
                                         bool is_causal) {
    if (T % BT != 0)
        __builtin_unreachable();
    if (T % 32 != 0)
        __builtin_unreachable();

    // enable Flush to Zero (0x8000)
    // enable Denormals are Zero (0x40)
    // for vectorising std:exp and avoiding denormals
    unsigned old_csr = _mm_getcsr();
    _mm_setcsr(old_csr | 0x8040);

    int64_t qk_dim = 0;
    for (auto& kind : kinds) {
        if (kind == "ipa")
            qk_dim += C * N_IPA;
        else if (kind == "daa")
            qk_dim += C * N_DAA;
    }

    const int64_t mv_stride    = C * N_BLADES;
    const int64_t out_t_stride = H * mv_stride;

    const float scalar_scale   = 1.f / std::sqrt((float)qk_dim);
    const float scalar_neg_inf = -std::numeric_limits<float>::infinity();
    const __m256 v_scale       = _mm256_set1_ps(scalar_scale);
    const __m256 v_neg_inf     = _mm256_set1_ps(scalar_neg_inf);

    std::vector<float> q_flat(T * qk_dim);
    std::vector<float> k_flat(T * qk_dim);
    std::vector<float> kt_flat(qk_dim * T);
    std::vector<float> scores(T < FLASH_T_THRESHOLD ? T * T : 0);

    alignas(32) float score_tile[BT * BT];
    alignas(32) float row_max_fa[BT];
    alignas(32) float row_sum_fa[BT];

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float* q_bh = q + b * qs_b + h * qs_h;
            const float* k_bh = k + b * qs_b + h * qs_h;
            const float* v_bh = v + b * qs_b + h * qs_h;
            float* o_bh       = out + (b * T * H + h) * mv_stride;

            for (int64_t t = 0; t < T; ++t) {
                float* qf   = q_flat.data() + t * qk_dim;
                float* kf   = k_flat.data() + t * qk_dim;
                int64_t off = 0;
                for (size_t ki = 0; ki < kinds.size(); ++ki) {
                    const float* wptr = (ki < weights.size()) ? weights[ki] : nullptr;
                    if (kinds[ki] == "ipa") {
                        for (int64_t c = 0; c < C; ++c) {
                            const float* mv_q = q_bh + t * qs_t + c * N_BLADES;
                            const float* mv_k = k_bh + t * qs_t + c * N_BLADES;
                            float pq[N_IPA], pk[N_IPA];
                            project_ipa(mv_q, pq);
                            project_ipa(mv_k, pk);
                            float w = wptr ? wptr[h * C + c] : 1.f;
                            for (int j = 0; j < N_IPA; ++j) {
                                qf[off + c * N_IPA + j] = pq[j] * w;
                                kf[off + c * N_IPA + j] = pk[j];
                            }
                        }
                        off += C * N_IPA;
                    } else if (kinds[ki] == "daa") {
                        for (int64_t c = 0; c < C; ++c) {
                            const float* mv_q = q_bh + t * qs_t + c * N_BLADES;
                            const float* mv_k = k_bh + t * qs_t + c * N_BLADES;
                            float pq[N_DAA], pk[N_DAA];
                            project_daa_bq(mv_q, pq);
                            project_daa_bk(mv_k, pk);
                            float w = wptr ? wptr[h * C + c] : 1.f;
                            for (int j = 0; j < N_DAA; ++j) {
                                qf[off + c * N_DAA + j] = pq[j] * w;
                                kf[off + c * N_DAA + j] = pk[j];
                            }
                        }
                        off += C * N_DAA;
                    }
                }
            }

            for (int64_t t2 = 0; t2 < T; ++t2)
                for (int64_t d = 0; d < qk_dim; ++d)
                    kt_flat[d * T + t2] = k_flat[t2 * qk_dim + d];

            if (T < FLASH_T_THRESHOLD) {
                tiled_mm<TILE_MR, TILE_NR>(q_flat.data(), qk_dim, kt_flat.data(), T, scores.data(), T, T, T, qk_dim);
                vec_softmax(scores.data(), T, v_scale, attn_mask, is_causal, b, H, h, v_neg_inf);
                tiled_mm<TILE_MR, TILE_NR>(scores.data(), T, v_bh, qs_t, o_bh, out_t_stride, T, mv_stride, T);
            } else {
                flash_attention_bh(q_flat.data(),
                                   kt_flat.data(),
                                   v_bh,
                                   o_bh,
                                   T,
                                   qk_dim,
                                   mv_stride,
                                   out_t_stride,
                                   qs_t,
                                   v_scale,
                                   v_neg_inf,
                                   scalar_neg_inf,
                                   is_causal,
                                   score_tile,
                                   row_max_fa,
                                   row_sum_fa);
            }
        }
    }

    _mm_setcsr(old_csr);
}
}  // namespace turbogator
