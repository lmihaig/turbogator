#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "ops.hpp"

namespace turbogator {

static constexpr int N_BLADES  = 16;
static constexpr int N_IPA     = 7;
static constexpr int N_DAA     = 5;
static constexpr float DAA_EPS = 1e-3f;

static constexpr int64_t BT                = 64;
static constexpr int64_t FLASH_T_THRESHOLD = 256;

static constexpr int TILE_MR = 4;
static constexpr int TILE_NC = 4;

static constexpr int IPA_IDX[N_IPA] = {0, 2, 3, 4, 8, 9, 10};
static constexpr int TRI_IDX[4]     = {11, 12, 13, 14};

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

// funky e^x
__attribute__((always_inline)) static inline float fast_exp(float x) {
    // clamp for underflowflow e^x = 0
    // -87.33654f < x <= 0.
    x = x > -87.33654f ? x : -87.33654f;

    // integer part n =  x * log2(e)
    float y  = x * 1.44269504f;
    int n    = (int)std::lrint(y);
    float fn = (float)n;

    //  fractional part f = x - n * ln(2)
    x       = x - fn * 0.69314575f;
    float f = x - fn * 1.4286068e-06f;

    // Horner's method
    // p(f) = 1 + f + f^2/2 + f^3/6 + f^4/24
    float p = 0.04166667f * f + 0.16666667f;
    p       = p * f + 0.5f;
    p       = p * f + 1.0f;
    p       = p * f + 1.0f;

    // 2^n
    int exp_bits = (n + 127) << 23;
    float two_to_n;
    std::memcpy(&two_to_n, &exp_bits, sizeof(float));

    // e^x = 2^n * e^f
    return p * two_to_n;
}

template <int MR, int NC>
__attribute__((always_inline)) static inline void mkernel(const float* __restrict__ A,
                                                          int64_t lda,
                                                          const float* __restrict__ B,
                                                          int64_t ldb,
                                                          float* __restrict__ C,
                                                          int64_t ldc,
                                                          int64_t K) {
    float c[MR][NC];
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NC; ++j)
            c[i][j] = 0.f;

    for (int64_t k = 0; k < K; ++k) {
        const float* brow = B + k * ldb;
        float bv[NC];
#pragma GCC unroll 8
        for (int j = 0; j < NC; ++j)
            bv[j] = brow[j];
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            const float a = A[(int64_t)i * lda + k];
            for (int j = 0; j < NC; ++j)
                c[i][j] += a * bv[j];
        }
    }
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NC; ++j)
            C[(int64_t)i * ldc + j] = c[i][j];
}

template <int MR, int NC>
static void tiled_mm(
    const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc, int64_t M, int64_t N, int64_t K) {
    const int64_t Mfull = M - (M % MR);
    const int64_t Nfull = N - (N % NC);

    for (int64_t i0 = 0; i0 < Mfull; i0 += MR)
        for (int64_t j0 = 0; j0 < Nfull; j0 += NC)
            mkernel<MR, NC>(A + i0 * lda, lda, B + j0, ldb, C + i0 * ldc + j0, ldc, K);

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

template <int MR, int NC>
__attribute__((always_inline)) static inline void mkernel_accum(const float* __restrict__ A,
                                                                int64_t lda,
                                                                const float* __restrict__ B,
                                                                int64_t ldb,
                                                                float* __restrict__ C,
                                                                int64_t ldc,
                                                                int64_t K,
                                                                const float* __restrict__ row_scale) {
    float c[MR][NC];
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i) {
        const float vs = row_scale[i];
        for (int j = 0; j < NC; ++j)
            c[i][j] = C[(int64_t)i * ldc + j] * vs;
    }
    for (int64_t k = 0; k < K; ++k) {
        const float* brow = B + k * ldb;
        float bv[NC];
#pragma GCC unroll 8
        for (int j = 0; j < NC; ++j)
            bv[j] = brow[j];
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            const float a = A[(int64_t)i * lda + k];
            for (int j = 0; j < NC; ++j)
                c[i][j] += a * bv[j];
        }
    }
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NC; ++j)
            C[(int64_t)i * ldc + j] = c[i][j];
}

template <int MR, int NC>
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
    const int64_t Mfull = M - (M % MR);
    const int64_t Nfull = N - (N % NC);

    for (int64_t i0 = 0; i0 < Mfull; i0 += MR)
        for (int64_t j0 = 0; j0 < Nfull; j0 += NC)
            mkernel_accum<MR, NC>(A + i0 * lda, lda, B + j0, ldb, C + i0 * ldc + j0, ldc, K, row_scale + i0);

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

static constexpr int SM_UNROLL = 4;
static inline void softmax(float* scores,
                           int64_t T,
                           float scale,
                           const float* attn_mask,
                           bool is_causal,
                           int64_t b,
                           int64_t H,
                           int64_t h,
                           float neg_inf,
                           float* __restrict__ row_inv) {
    for (int64_t t1 = 0; t1 < T; ++t1) {
        float* row = scores + t1 * T;

        float lmax[SM_UNROLL];
        float lsum[SM_UNROLL];
#pragma GCC unroll 4
        for (int i = 0; i < SM_UNROLL; ++i) {
            lmax[i] = neg_inf;
            lsum[i] = 0.f;
        }

        int64_t t2 = 0;
        for (; t2 + SM_UNROLL <= T; t2 += SM_UNROLL) {
#pragma GCC unroll 4
            for (int i = 0; i < SM_UNROLL; ++i) {
                int64_t col = t2 + i;
                float s     = row[col] * scale;
                if (attn_mask)
                    s += attn_mask[((b * H + h) * T + t1) * T + col];
                if (is_causal && col > t1)
                    s = neg_inf;
                row[col] = s;

                float new_max = s > lmax[i] ? s : lmax[i];
                float corr    = fast_exp(lmax[i] - new_max);
                lsum[i]       = lsum[i] * corr + fast_exp(s - new_max);
                lmax[i]       = new_max;
            }
        }

        // merge lanes
        float gmax = neg_inf;
#pragma GCC unroll 4
        for (int i = 0; i < SM_UNROLL; ++i)
            gmax = lmax[i] > gmax ? lmax[i] : gmax;
        float gsum = 0.f;
#pragma GCC unroll 4
        for (int i = 0; i < SM_UNROLL; ++i)
            gsum += lsum[i] * fast_exp(lmax[i] - gmax);

        // serial tail for the remaining < SM_UNROLL columns
        for (; t2 < T; ++t2) {
            float s = row[t2] * scale;
            if (attn_mask)
                s += attn_mask[((b * H + h) * T + t1) * T + t2];
            if (is_causal && t2 > t1)
                s = neg_inf;
            row[t2] = s;

            float new_max = s > gmax ? s : gmax;
            float corr    = fast_exp(gmax - new_max);
            gsum          = gsum * corr + fast_exp(s - new_max);
            gmax          = new_max;
        }

        for (int64_t c = 0; c < T; ++c)
            row[c] = fast_exp(row[c] - gmax);
        row_inv[t1] = 1.0f / gsum;
    }
}

static inline void flash_attention_bh(const float* __restrict__ q_flat,
                                      const float* __restrict__ kt_flat,
                                      const float* __restrict__ v_bh,
                                      float* __restrict__ o_bh,
                                      int64_t T,
                                      int64_t qk_dim,
                                      int64_t mv_stride,
                                      int64_t out_t_stride,
                                      int64_t qs_t,
                                      float scale,
                                      float neg_inf,
                                      bool is_causal,
                                      float* __restrict__ score_tile,
                                      float* __restrict__ row_max_fa,
                                      float* __restrict__ row_sum_fa) {
    float corrections[BT];

    for (int64_t t1s = 0; t1s < T; t1s += BT) {
        for (int r = 0; r < BT; ++r) {
            row_max_fa[r] = neg_inf;
            row_sum_fa[r] = 0.f;
        }

        float* o_qt = o_bh + t1s * out_t_stride;
        for (int r = 0; r < BT; ++r) {
            float* orow = o_qt + r * out_t_stride;
            for (int64_t d = 0; d < mv_stride; ++d)
                orow[d] = 0.f;
        }

        const int64_t t2_lim = is_causal ? (t1s + BT) : T;
        for (int64_t t2s = 0; t2s < t2_lim; t2s += BT) {
            tiled_mm<TILE_MR, TILE_NC>(q_flat + t1s * qk_dim, qk_dim, kt_flat + t2s, T, score_tile, BT, BT, BT, qk_dim);

            const bool diag = is_causal && (t2s == t1s);

            for (int r = 0; r < BT; ++r) {
                float* srow = score_tile + r * BT;

                float tile_max = neg_inf;
                for (int c = 0; c < BT; ++c) {
                    float sv = srow[c] * scale;
                    if (diag && (t2s + c) > (t1s + r))
                        sv = neg_inf;
                    srow[c] = sv;
                    if (sv > tile_max)
                        tile_max = sv;
                }

                float old_max    = row_max_fa[r];
                float new_max    = old_max > tile_max ? old_max : tile_max;
                float correction = (old_max >= new_max) ? 1.f : fast_exp(old_max - new_max);
                corrections[r]   = correction;
                row_sum_fa[r] *= correction;
                row_max_fa[r] = new_max;

                float vsum = 0.f;
                for (int c = 0; c < BT; ++c) {
                    float e = fast_exp(srow[c] - new_max);
                    srow[c] = e;
                    vsum += e;
                }
                row_sum_fa[r] += vsum;
            }

            tiled_mm_accum<TILE_MR, TILE_NC>(
                score_tile, BT, v_bh + t2s * qs_t, qs_t, o_qt, out_t_stride, BT, mv_stride, BT, corrections);
        }

        for (int r = 0; r < BT; ++r) {
            float inv   = 1.f / row_sum_fa[r];
            float* orow = o_qt + r * out_t_stride;
            for (int64_t d = 0; d < mv_stride; ++d)
                orow[d] *= inv;
        }
    }
}

void equi_geometric_attention_opt_v2(const float* __restrict__ q,
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
    int64_t qk_dim = 0;
    for (auto& kind : kinds) {
        if (kind == "ipa")
            qk_dim += C * N_IPA;
        else if (kind == "daa")
            qk_dim += C * N_DAA;
    }

    const int64_t mv_stride    = C * N_BLADES;
    const int64_t out_t_stride = mv_stride;

    const float scale   = 1.f / std::sqrt((float)qk_dim);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    const bool use_flash = (T >= FLASH_T_THRESHOLD) && (T % BT == 0);

    static thread_local std::vector<float> q_flat, k_flat, kt_flat, scores, row_inv;
    q_flat.resize(T * qk_dim);
    k_flat.resize(T * qk_dim);
    kt_flat.resize(qk_dim * T);
    scores.resize(use_flash ? 0 : T * T);
    row_inv.resize(use_flash ? 0 : T);

    float score_tile[BT * BT];
    float row_max_fa[BT];
    float row_sum_fa[BT];

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float* q_bh = q + b * qs_b + h * qs_h;
            const float* k_bh = k + b * qs_b + h * qs_h;
            const float* v_bh = v + b * qs_b + h * qs_h;
            float* o_bh       = out + (b * H + h) * T * mv_stride;

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

            if (use_flash) {
                flash_attention_bh(q_flat.data(),
                                   kt_flat.data(),
                                   v_bh,
                                   o_bh,
                                   T,
                                   qk_dim,
                                   mv_stride,
                                   out_t_stride,
                                   qs_t,
                                   scale,
                                   neg_inf,
                                   is_causal,
                                   score_tile,
                                   row_max_fa,
                                   row_sum_fa);
            } else {
                tiled_mm<TILE_MR, TILE_NC>(q_flat.data(), qk_dim, kt_flat.data(), T, scores.data(), T, T, T, qk_dim);
                softmax(scores.data(), T, scale, attn_mask, is_causal, b, H, h, neg_inf, row_inv.data());
                tiled_mm<TILE_MR, TILE_NC>(scores.data(), T, v_bh, qs_t, o_bh, out_t_stride, T, mv_stride, T);
                for (int64_t t1 = 0; t1 < T; ++t1) {
                    float inv   = row_inv[t1];
                    float* orow = o_bh + t1 * out_t_stride;
                    for (int64_t d = 0; d < mv_stride; ++d)
                        orow[d] *= inv;
                }
            }
        }
    }
}

}  // namespace turbogator
