import os

os.environ.setdefault("FORCE_COLOR", "1")

import torch
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel
from termcolor import colored

import config as app_config
from turbogator.engine import TurboGatorModel


def validate(seed=42):
    torch.manual_seed(seed)

    N = app_config.REPRESENTATIVE_N
    T, C_in = app_config.get_dimensions(N)
    batch_size = app_config.BATCH_SIZE
    vector_dim = app_config.VECTOR_DIM

    device = "cpu"
    config = MVOnlyGATrConfig(size_channels_in=C_in)
    x = torch.randn(batch_size, T, C_in, vector_dim).to(device)

    net = MVOnlyGATrModel(config).to(device)
    output = net(x)

    net_aslr = TurboGatorModel(config).to(device)
    net_aslr.load_state_dict(net.state_dict(), strict=True)
    output_aslr = net_aslr(x)

    diff = (output - output_aslr).abs().mean().item()
    # 0.01% relative error
    # 0.00001 absolute error
    try:
        torch.testing.assert_close(
            output_aslr,
            output,
            rtol=1e-4,
            atol=1e-5,
            msg="Output does not match reference!",
        )
    except AssertionError:
        print(
            f"Mean absolute difference between implementations: {colored(diff, 'red', force_color=True)}",
            flush=True,
        )
        print(colored("Validation FAILED", "red", force_color=True), flush=True)
        raise

    print(
        f"Mean absolute difference between implementations: {colored(diff, 'green', force_color=True)}",
        flush=True,
    )
    print(colored("Validation OK!", "green", force_color=True), flush=True)


if __name__ == "__main__":
    validate()
