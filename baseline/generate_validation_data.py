# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0.0", "numpy>=1.24.0", "einops>=0.8.1"]
# ///
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.append(os.path.join(os.path.dirname(__file__), "ezgatr", "src"))
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel

import config as app_config


def generate_data():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "baseline")
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    size_channels_in = app_config.CHANNELS
    batch_size = app_config.BATCH_SIZE
    N = app_config.REPRESENTATIVE_N

    print(f"Generating local validation data for N={N}...")
    model_config = MVOnlyGATrConfig(size_channels_in=size_channels_in)
    net = MVOnlyGATrModel(model_config).to(device)
    net.eval()

    torch.manual_seed(42)
    x = torch.randn(batch_size, N, size_channels_in, 16).to(device)

    with torch.no_grad():
        output = net(x)

    x.numpy().astype(np.float32).tofile(os.path.join(output_dir, "input.bin"))
    output.numpy().astype(np.float32).tofile(os.path.join(output_dir, "expected.bin"))

    with open(
        os.path.join(output_dir, "validation_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({"N": N, "batch": batch_size, "channels": size_channels_in}, f)

    print("Validation data generated in results/baseline/")


if __name__ == "__main__":
    generate_data()
