import json
from pathlib import Path

import pandas as pd

# rows without server to default to mihai
# in case we want to keep/plot smth
LEGACY_SERVER = "mihai"


def _annotate_server(row):
    if isinstance(row, dict):
        row.setdefault("server", LEGACY_SERVER)
    return row


def read_jsonl_records(path: Path):
    if not path.exists():
        return []

    records = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(_annotate_server(row))
    return records


def load_history_dataframe(history_file: Path):
    return pd.DataFrame(
        row for row in read_jsonl_records(history_file) if "data" in row
    )


def load_baseline_dataframe(baseline_file: Path):
    if not baseline_file.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(baseline_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    if isinstance(payload, dict):
        payload.setdefault("server", LEGACY_SERVER)
        data = payload.get("data", [])
    else:
        data = []
    return pd.DataFrame(data if isinstance(data, list) else [])


def load_reference_dataframe(reference_file: Path):
    if not reference_file.exists():
        return pd.DataFrame()

    try:
        payload = json.loads(reference_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()

    if isinstance(payload, dict):
        payload.setdefault("server", LEGACY_SERVER)
        data = payload.get("data", [])
    else:
        data = []
    return pd.DataFrame(data if isinstance(data, list) else [])
