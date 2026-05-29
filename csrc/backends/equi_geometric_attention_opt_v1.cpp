#include <algorithm>
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
static constexpr int64_t BT    = 32;
static constexpr int64_t BD    = 288;

static constexpr int IPA_IDX[N_IPA] = {0, 2, 3, 4, 8, 9, 10};
static constexpr int TRI_IDX[4]     = {11, 12, 13, 14};

static void project_ipa(const float* mv, float* out) {
    for (int i = 0; i < N_IPA; ++i)
        out[i] = mv[IPA_IDX[i]];
}

static void normalize_tri(const float* mv, float* r) {
    float r3   = mv[TRI_IDX[3]];
    float norm = r3 / (r3 * r3 + DAA_EPS);
    for (int i = 0; i < 4; ++i)
        r[i] = mv[TRI_IDX[i]] * norm;
}

static void project_daa_bq(const float* mv, float* out) {
    float r[4];
    normalize_tri(mv, r);
    out[0] = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    out[1] = r[3] * r[3];
    out[2] = r[0] * r[3];
    out[3] = r[1] * r[3];
    out[4] = r[2] * r[3];
}

static void project_daa_bk(const float* mv, float* out) {
    float r[4];
    normalize_tri(mv, r);
    out[0] = -(r[3] * r[3]);
    out[1] = -(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    out[2] = 2.f * r[0] * r[3];
    out[3] = 2.f * r[1] * r[3];
    out[4] = 2.f * r[2] * r[3];
}

void equi_geometric_attention_opt_v1(const float* q,
                                     const float* k,
                                     const float* v,
                                     float* out,
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

    const int64_t mv_stride = C * N_BLADES;
    const float scale       = 1.f / std::sqrt((float)qk_dim);

    std::vector<float> q_flat(T * qk_dim);
    std::vector<float> k_flat(T * qk_dim);
    std::vector<float> scores(T * T);

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

            const float neg_inf = -std::numeric_limits<float>::infinity();
            for (int64_t t1 = 0; t1 < T; ++t1)
                for (int64_t t2 = 0; t2 < T; ++t2)
                    scores[t1 * T + t2] = (is_causal && t2 > t1) ? neg_inf : 0.f;

            for (int64_t t1b = 0; t1b < T; t1b += BT) {
                const int64_t t1_end = std::min(t1b + BT, T);
                for (int64_t t2b = 0; t2b < T; t2b += BT) {
                    if (is_causal && t2b >= t1_end)
                        break;
                    const int64_t t2_end = std::min(t2b + BT, T);
                    for (int64_t db = 0; db < qk_dim; db += BD) {
                        const int64_t d_len = std::min(db + BD, qk_dim) - db;
                        for (int64_t t1 = t1b; t1 < t1_end; ++t1) {
                            const float* qf1     = q_flat.data() + t1 * qk_dim + db;
                            const int64_t t2_lim = is_causal ? std::min(t2_end, t1 + 1) : t2_end;
                            for (int64_t t2 = t2b; t2 < t2_lim; ++t2) {
                                const float* kf2 = k_flat.data() + t2 * qk_dim + db;
                                float s          = 0.f;
                                for (int64_t d = 0; d < d_len; ++d)
                                    s += qf1[d] * kf2[d];
                                scores[t1 * T + t2] += s;
                            }
                        }
                    }
                }
            }

            for (int64_t t1 = 0; t1 < T; ++t1) {
                float row_max = neg_inf;
                for (int64_t t2 = 0; t2 < T; ++t2) {
                    float& s = scores[t1 * T + t2];
                    s *= scale;
                    if (attn_mask)
                        s += attn_mask[((b * H + h) * T + t1) * T + t2];
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

                float* o_t1 = o_bh + t1 * mv_stride;
                std::memset(o_t1, 0, mv_stride * sizeof(float));
                for (int64_t t2 = 0; t2 < T; ++t2) {
                    float a           = scores[t1 * T + t2];
                    const float* v_t2 = v_bh + t2 * qs_t;
                    for (int64_t d = 0; d < mv_stride; ++d)
                        o_t1[d] += a * v_t2[d];
                }
            }
        }
    }
}

}  // namespace turbogator
