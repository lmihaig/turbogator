MACHINE = "Intel N100 (1 Core, 800Mhz)"

# 800 Mhz
CPU_FREQ = 8e8

ROOFLINE_BETA = 32
ROOFLINE_PI_SCALAR = 3.33
ROOFLINE_PI_VECTOR = 8

# probably don't need to change these
BATCH_SIZE = 8
CHANNELS = 2
BUILD_JOBS = 4
PINNED_CPU_CORE = 2


# i think running only with perf is fine, overhead seems to be ~2% in testing
# but if needed we can run both or either
RUN_BENCHMARK = False
RUN_PERF = True

SIZES = [
    16,
    32,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
]

# this is used by the validation pipeline and also valgrind
REPRESENTATIVE_N = 128


PERF_EVENTS = [
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
]


LLVM_MCA_FLAGS = [
    "-mtriple=x86_64-unknown-unknown",
    "-mcpu=skylake",  # tool doesn't support gracemont particularly, but skylake e-cores should satisfy
    "-dispatch=6",
    "-lqueue=64",
    "-squeue=40",
    "-iterations=1000",
    "-bottleneck-analysis",
    "-timeline",
    "-timeline-max-cycles=100",
    "-all-stats",
    "-output-asm-variant=1",
]


def calculate_total_flops(N):
    # W(n)
    B = BATCH_SIZE
    T = N
    C_in = CHANNELS
    D = 16

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

    print(f"W({N}): {total_flops}")
    return total_flops


##############################3
# TODO: TEMP REPLACE
def calculate_total_bytes(N):
    # Q(n)
    B = BATCH_SIZE
    T = N
    C_in = CHANNELS
    D = 16

    C_hid = 32
    C_int = 32
    C_out = 1
    H = 4
    L = 4

    total_bytes = T * D * C_in * 8
    print(f"Q({N}): {total_bytes}")

    return total_bytes


############################
