import torch
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel

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
    # copy random weights to ensure identical
    net_aslr.load_state_dict(net.state_dict(), strict=True)
    output_aslr = net_aslr(x)
    print(f"Mean absolute difference between implementations: {(output - output_aslr).abs().mean().item()}")
    # 0.01% relative error
    # 0.00001 absolute error
    torch.testing.assert_close(
        output_aslr,
        output,
        rtol=1e-4,
        atol=1e-5,  # 0.00001 absolute error
        msg="Output does not match reference!",
    )

    print("Validation OK!")


if __name__ == "__main__":
    validate()
