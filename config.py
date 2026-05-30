ACTIVE_SERVER = "adam"

WARMUP = 5
STEPS = 9
# SIZES = [1]
# SIZES = [2, 4, 6, 8]
SIZES = [2, 4, 8, 16, 32]
# SIZES = [1, 4, 6, 16, 32, 48, 56, 64]
REPRESENTATIVE_N = 8

SHOW_CACHE_LINES = True


## Do not worry about things below this line :)
########################################################################
BATCH_SIZE = 8
VECTOR_DIM = 16

URL = "https://aos.licu.dev"
AUTH = "turbogator:TurboGator2026"


SERVERS = {
    "adam": {
        "ssh": "mihai@100.104.223.34",
        "root_dir": "/home/mihai/aos",
        "user_systemd": True,
        "listen_host": "127.0.0.1",
        "machine": "Intel Core 7 240H (P-core, 2.5GHz)",
        "cpu_freq": 2.5e9,
        "pinned_core": "4",
        "build_jobs": 12,
        "roofline_beta": 32,
        #  these are THEORETICAL, for measured see docs/hardware_specs
        "roofline_beta_levels": {
            "l1": 96,
            "l2": 64,
            "l3": 32,
            "dram": 9,
        },
        "roofline_pi_scalar": 5,
        "roofline_pi_vector": 40,
        "perf_events": [
            "cycles",
            "instructions",
            "FP_ARITH_INST_RETIRED.SCALAR_SINGLE",
            "FP_ARITH_INST_RETIRED.128B_PACKED_SINGLE",
            "FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE",
            "L1-dcache-loads",
            "L1-dcache-stores",
            "l2_rqsts.references",
            "l2_rqsts.miss",
            "mem_load_retired.l3_miss",
        ],
        "dram_events": [
            "unc_m_cas_count_rd",
            "unc_m_cas_count_wr",
        ],
    },
    "mihai": {
        "ssh": "root@192.168.1.130",
        "root_dir": "/opt/aos",
        "user_systemd": False,
        "listen_host": "0.0.0.0",
        "machine": "Intel N100 (E-Core, 800Mhz)",
        "cpu_freq": 8e8,
        "pinned_core": "2",
        "build_jobs": 4,
        "roofline_beta": 32,
        "roofline_pi_scalar": 3.33,
        "roofline_pi_vector": 16,
        "perf_events": [
            "cycles",
            "instructions",
            "branches",
            "branch-misses",
            "L1-dcache-loads",
            "LLC-load-misses",
            "dTLB-load-misses",
            "topdown-retiring",
            "topdown-bad-spec",
            "topdown-fe-bound",
            "topdown-be-bound",
        ],
        "dram_events": [],
    },
}

_ACTIVE = SERVERS[ACTIVE_SERVER]
MACHINE = _ACTIVE["machine"]
CPU_FREQ = _ACTIVE["cpu_freq"]
PINNED_CPU_CORE = _ACTIVE["pinned_core"]
PERF_EVENTS = _ACTIVE["perf_events"]
DRAM_EVENTS = _ACTIVE.get("dram_events", [])
ROOFLINE_BETA = _ACTIVE["roofline_beta"]
ROOFLINE_BETA_LEVELS = _ACTIVE.get("roofline_beta_levels", {})
ROOFLINE_PI_SCALAR = _ACTIVE["roofline_pi_scalar"]
ROOFLINE_PI_VECTOR = _ACTIVE["roofline_pi_vector"]
BUILD_JOBS = _ACTIVE["build_jobs"]


def roofline_beta_for(level):
    return ROOFLINE_BETA_LEVELS.get(level, ROOFLINE_BETA)


CACHE_SIZES_BYTES = {
    "L1d": 48 * 1024,
    "L2": 1280 * 1024,
    "L3": 24576 * 1024,
}

# whole model miss rate
# observed through experiments
# L1d: 48 Kb too small for working set??
# L2: 1280 Kb   -> miss rate flat ~27% (N=4) climbs to 36%    (N=8) -> 41% (N=16) -> 58% (N=32);
# L3: 24576 Kb  -> miss rate flat ~0% (N=48) climbs to 0.01% (N=56) -> 0.02% (N=64);
CACHE_LINE_SIZES = {
    "L2": 64 * 12**2,  # N ~12  (size 9216)
    "L3": 64 * 57**2,  # N ~57  (size 207936)
}

# expected theory says:
# ~9*T^2 attention dominates
# boundaries would be at
#  L1d  ~ N=2.3
#  L2   ~ N=12
#  L3   ~ N=52


def get_dimensions(N):
    # returns (T, C_in) for given size N
    T = 32 * N
    C_in = 2 * N
    return T, C_in


def calculate_total_flops(B, T, C_in, D):
    # W(n)

    C_hid = 32
    C_int = 32
    C_out = 1
    H = 4
    L = 4

    # -----
    def f_mean(b, t, c, d):
        return b * t * c * d

    def f_equi_linear(b, t, c_in, c_out, d):
        return 2 * b * t * c_in * c_out * d * d

    # -----
    def f_rmsnorm(b, t, c, d):
        return b * t * (c * (15 + 2 * d + 1) + 4)

    def f_geom_attn(b, h, t, c, d):
        # qs size of ipa - (b, h, t, c, 7)
        _linear_normalizer = b * h * t * c * 3  # linear square normalizer div pow add
        qk_daa_computation = b * h * t * c * 4  # trivector normalization muls
        qk_daa_computation += (
            5 * 4 * 4 * (b * h * t * c) * 3
        )  # dist_vec computation k times 4*4 times 2 muls and one add
        qk_daa_computation += _linear_normalizer

        qk_daa_computation *= 2  # we have q and k computations
        qk_ipa_weighting = b * h * t * c * 7  # q weighting
        qk_daa_weighting = b * h * t * c * 5  # q weighting

        # torch.cat(qs, dim=-1) shape - (b, h, t, c*12)
        # torch.cat(qs, dim=-1) shape - (b, h, t, c*12)
        scaled_dot_product_attention = (
            b * h * (t**2) * (24 * c - 1)
        )  # 12c multiplications and 12c-1 additions per output element
        scaled_dot_product_attention += b * h * (t**2)  # scaling scores
        scaled_dot_product_attention += (
            4 * b * h * (t**2)
        )  # softmax exp div 1 FLOP each
        scaled_dot_product_attention += 32 * b * h * (t**2) * c  # Attention output AV

        return (
            scaled_dot_product_attention
            + qk_daa_computation
            + qk_ipa_weighting
            + qk_daa_weighting
        )

    def f_add(b, t, c, d):
        return b * t * c * d

    def f_exp(b, t, c, d):
        return b * t * c * d

    # -----
    def f_einsum_bilinear(b, t, c, d):
        # torch.einsum("ijk, ...j, ...k -> ...i", basis, x, y)
        volume = b * t * c

        # out[i] += basis[i][j][k] * x[j] * y[k]
        flops_per_pair = 3 * (d**3)

        return volume * flops_per_pair

    def f_geom_prod(b, t, c, d):
        return f_einsum_bilinear(b, t, c, d)

    def f_equi_join(b, t, c, d):
        einsum_flops = f_einsum_bilinear(b, t, c, d)

        # ret *= reference[..., [14]]
        volume = b * t * c
        broadcast_muls = volume * d

        return einsum_flops + broadcast_muls

    # -----
    def f_gelu(b, t, c, d):
        return b * t * c * (4 + d)

    def attention_flops():
        norm = f_rmsnorm(B, T, C_hid, D)
        proj_qkv = f_equi_linear(B, T, C_hid, 3 * H * C_hid, D)
        attention_weights = 2 * f_exp(
            H, 1, C_hid, 1
        )  # we have 2 kinds of attention ipa and daa
        attn = f_geom_attn(B, H, T, C_hid, D)
        proj_out = f_equi_linear(B, T, H * C_hid, C_hid, D)
        residual = f_add(B, T, C_hid, D)

        return norm + proj_qkv + attn + proj_out + residual + attention_weights

    def bilinear_flops():
        proj_bil = f_equi_linear(B, T, C_hid, 4 * C_int, D)
        geom_prod = f_geom_prod(B, T, C_int, D)
        equi_join = f_equi_join(B, T, C_int, D)
        proj_out = f_equi_linear(B, T, 2 * C_int, C_hid, D)
        return proj_bil + geom_prod + equi_join + proj_out

    def mlp_flops():
        norm = f_rmsnorm(B, T, C_hid, D)
        bilinear = bilinear_flops()
        gelu = f_gelu(B, T, C_hid, D)
        proj_out = f_equi_linear(B, T, C_hid, C_hid, D)
        residual = f_add(B, T, C_hid, D)
        return norm + bilinear + gelu + proj_out + residual

    reference_flops = f_mean(B, T, C_in, D)
    embedding_flops = f_equi_linear(B, T, C_in, C_hid, D)

    flops_per_block = attention_flops() + mlp_flops()
    total_block_flops = L * flops_per_block

    head_flops = f_equi_linear(B, T, C_hid, C_out, D)

    total_flops = reference_flops + embedding_flops + total_block_flops + head_flops

    # print(f"W({B}, {T}, {C_in}, {D}): {total_flops}")
    return total_flops


##############################3
def calculate_total_bytes(B, T, C_in, D):
    # Q(n) — strict compulsory traffic (lower bound):
    #   input + output + all learnable params + all precomputed bases

    C_hid = 32
    C_int = 32
    C_out = 1
    H = 4
    L = 4

    BYTES = 4  # PyTorch default: float32

    # 1. input + final output
    io_bytes = B * T * (C_in + C_out) * D * BYTES

    # 2. learnable params
    def equi_linear_params(c_in, c_out):
        # weight: (c_out, c_in, 9), bias: (c_out,)
        return c_out * c_in * 9 + c_out

    def equi_rmsnorm_params(c):
        # channelwise rescale weight: (c,)
        return c

    embed_p = equi_linear_params(C_in, C_hid)
    head_p = equi_linear_params(C_hid, C_out)

    # one block
    attn_norm_p = equi_rmsnorm_params(C_hid)
    attn_mix_p = 2 * H * C_hid  # ipa + daa, each (H, 1, C_hid, 1)
    proj_qkv_p = equi_linear_params(C_hid, 3 * H * C_hid)
    attn_out_p = equi_linear_params(H * C_hid, C_hid)

    mlp_norm_p = equi_rmsnorm_params(C_hid)
    proj_bil_p = equi_linear_params(C_hid, 4 * C_int)
    bil_out_p = equi_linear_params(2 * C_int, C_hid)
    mlp_out_p = equi_linear_params(C_hid, C_hid)

    block_p = (
        attn_norm_p
        + attn_mix_p
        + proj_qkv_p
        + attn_out_p
        + mlp_norm_p
        + proj_bil_p
        + bil_out_p
        + mlp_out_p
    )

    param_bytes = (embed_p + L * block_p + head_p) * BYTES

    # 3. precomputed bases
    basis_bytes = (
        9 * D * D  # EquiLinear basis
        + D**3  # geometric_product basis
        + D**3  # equi_join kernel
        + 2 * 4 * 4 * 5  # daa bq, bk
    ) * BYTES

    total_bytes = io_bytes + param_bytes + basis_bytes
    # print(f"Q({B}, {T}, {C_in}, {D}): {total_bytes}")

    return total_bytes


############################
