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

# 800 Mhz
CPU_FREQ = 8e8

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


# TODO: REPLACE WITH SOME REAL CALCULATION FOR OUR ALGO
def calculate_total_flops(N):
    return BATCH_SIZE * ((5000 * N) + (100 * (N**2)))
