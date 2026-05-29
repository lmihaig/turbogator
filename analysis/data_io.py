import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as app_config

LEGACY_SERVER = "mihai"
BYTES_PER_LINE = 64


# rename events here for qol
PERF_RENAME = {
    "cycles": "cycles_perf",
    "instructions": "instructions",
    "FP_ARITH_INST_RETIRED.SCALAR_SINGLE": "fp_scalar",
    "FP_ARITH_INST_RETIRED.128B_PACKED_SINGLE": "fp_128b",
    "FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE": "fp_256b",
    "L1-dcache-loads": "l1_loads",
    "L1-dcache-stores": "l1_stores",
    "l2_rqsts.references": "l1_l2_lines",  # cache lines crossing L1<->L2
    "l2_rqsts.miss": "l2_l3_lines",  # cache lines crossing L2<->L3
    "mem_load_retired.l3_miss": "l3_miss",  # loads escaping L3 to DRAM
}
DRAM_RENAME = {
    "unc_m_cas_count_rd": "dram_rd",
    "unc_m_cas_count_wr": "dram_wr",
}

# Total FLOPs from PMCs: each event * its lane-per-event factor.
FLOPS_TERMS = {
    "fp_scalar": 1,
    "fp_128b": 4,
    "fp_256b": 8,
}

BYTES_LEVELS = {
    "l1": "l1_l2_lines",
    "l2": "l2_l3_lines",
    "l3": "l3_miss",
}

# flops_theory == config.py calculate funcs
FLOPS_SOURCE = "flops_pmc"


def _sanitize(name):
    return name.lower().replace("-", "_").replace(".", "_").replace("/", "_")


def read_jsonl_records(path):
    if not path.exists():
        return []
    records = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row.setdefault("server", LEGACY_SERVER)
            records.append(row)
    return records


def _flatten_point(point, run):
    x = point.get("X") or {}
    perf = point.get("perf") or {}
    dram = point.get("dram") or {}

    flat = {
        "description": run.get("description"),
        "server": run.get("server"),
        "job_id": run.get("job_id"),
        "user": run.get("user"),
        "N": point.get("N"),
        "B": x.get("B"),
        "T": x.get("T"),
        "C_in": x.get("C_in"),
        "D": x.get("D"),
        "cycles_wall": point.get("cycles"),
    }
    for raw, val in perf.items():
        flat[PERF_RENAME.get(raw, _sanitize(raw))] = val
    for raw, val in dram.items():
        flat[DRAM_RENAME.get(raw, _sanitize(raw))] = val
    return flat


def _theory(fn, r):
    try:
        return fn(r["B"], r["T"], r["C_in"], r["D"])
    except (TypeError, ValueError):
        return float("nan")


def _derive(df):
    if df.empty:
        return df

    df["size"] = df["T"] * df["C_in"]

    # flops from PMCs: sum of present terms, weighted
    flops_pmc = None
    for col, w in FLOPS_TERMS.items():
        if col not in df.columns:
            continue
        term = w * df[col].fillna(0)
        flops_pmc = term if flops_pmc is None else flops_pmc + term
    df["flops_pmc"] = flops_pmc if flops_pmc is not None else float("nan")

    df["flops_theory"] = df.apply(
        lambda r: _theory(app_config.calculate_total_flops, r), axis=1
    )
    df["bytes_theory"] = df.apply(
        lambda r: _theory(app_config.calculate_total_bytes, r), axis=1
    )

    df["flops"] = df[FLOPS_SOURCE]

    # cycles: prefer perf when present, fall back to wall
    if "cycles_perf" in df.columns:
        df["cycles"] = df["cycles_perf"].where(
            df["cycles_perf"].notna(), df["cycles_wall"]
        )
    else:
        df["cycles"] = df["cycles_wall"]
    df["gcycles"] = df["cycles"] / 1e9

    # bytes per cache level (only those whose miss-counter is present)
    for level, miss_col in BYTES_LEVELS.items():
        if miss_col in df.columns:
            df[f"bytes_{level}"] = df[miss_col] * BYTES_PER_LINE

    if {"dram_rd", "dram_wr"}.issubset(df.columns):
        df["bytes_dram"] = (
            df["dram_rd"].fillna(0) + df["dram_wr"].fillna(0)
        ) * BYTES_PER_LINE

    # operational intensity per byte source
    for level in BYTES_LEVELS:
        col = f"bytes_{level}"
        if col in df.columns:
            df[f"oi_{level}"] = df["flops"] / df[col]
    if "bytes_dram" in df.columns:
        df["oi_dram"] = df["flops"] / df["bytes_dram"]
    df["oi_theory"] = df["flops"] / df["bytes_theory"]

    df["perf"] = df["flops"] / df["cycles"]
    df["perf_theory"] = df["flops_theory"] / df["cycles"]

    if "instructions" in df.columns:
        df["ipc"] = df["instructions"] / df["cycles"]
    if "fp_256b" in df.columns:
        df["avx2_ratio"] = (8 * df["fp_256b"].fillna(0)) / df["flops"]

    # cache hit/miss rates - each measured at its proper boundary
    if {"l1_l2_lines", "l1_loads"}.issubset(df.columns):
        df["l1_load_miss_rate"] = df["l1_l2_lines"] / df["l1_loads"]
    if {"l2_l3_lines", "l1_l2_lines"}.issubset(df.columns):
        df["l2_miss_rate"] = df["l2_l3_lines"] / df["l1_l2_lines"]
        df["l2_hit_rate"] = 1 - df["l2_miss_rate"]
    if {"l3_miss", "l1_loads"}.issubset(df.columns):
        df["l3_demand_miss_rate"] = df["l3_miss"] / df["l1_loads"]

    if {"l1_stores", "l1_loads"}.issubset(df.columns):
        df["store_load_ratio"] = df["l1_stores"] / df["l1_loads"]

    if {"bytes_dram", "l3_miss"}.issubset(df.columns):
        # fraction of DRAM bandwidth not explained by demand L3 misses -> prefetch + writeback
        df["dram_prefetch_share"] = (
            1 - (df["l3_miss"] * BYTES_PER_LINE) / df["bytes_dram"]
        )
    if "bytes_dram" in df.columns:
        df["dram_per_gflop"] = df["bytes_dram"] / (df["flops"] * 1e-9)
    if "bytes_l1" in df.columns:
        df["bytes_per_flop_l1"] = df["bytes_l1"] / df["flops"]

    df["flops_efficiency"] = df["flops_pmc"] / df["flops_theory"]

    return df


def load_history(path):
    runs = read_jsonl_records(path)

    latest = {}
    for r in runs:
        if "data" not in r:
            continue
        latest[r.get("description")] = r

    rows = []
    for run in latest.values():
        for pt in run.get("data") or []:
            rows.append(_flatten_point(pt, run))

    return _derive(pd.DataFrame(rows))


def description_order(df):
    seen = []
    for d in df["description"].tolist():
        if d not in seen:
            seen.append(d)
    return seen
