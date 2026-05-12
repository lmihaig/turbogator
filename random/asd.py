from __future__ import annotations

from pathlib import Path

import torch
from ezgatr.nn.functional import dual as dual_mod
from ezgatr.nn.functional import linear as linear_mod


def _format_cpp_value(value):
    text = f"{float(value):.6g}"
    if text.lower() in {"nan", "inf", "-inf"}:
        return f"{text}f"
    if "." in text or "e" in text or "E" in text:
        return f"{text}f"
    return f"{text}.0f"


def _print_tensor(name, tensor, output, precision, display_tensor=None):
    torch.set_printoptions(precision=precision, sci_mode=False, linewidth=160)
    cpu_tensor = tensor.detach().cpu()
    shown_tensor = display_tensor if display_tensor is not None else cpu_tensor

    print(
        f"{name} shape={tuple(cpu_tensor.shape)} dtype={cpu_tensor.dtype} device={cpu_tensor.device}",
        file=output,
    )
    print(
        f"nonzero entries: {(cpu_tensor != 0).sum().item()} of {cpu_tensor.numel()}",
        file=output,
    )

    print("", file=output)
    nonzero = (cpu_tensor != 0).nonzero(as_tuple=False)
    for idx in nonzero:
        i, j, k = idx.tolist()
        value = cpu_tensor[i, j, k].item()
        print(f"data[{i}][{j}][{k}] = {_format_cpp_value(value)};", file=output)

    print("", file=output)
    for i in range(shown_tensor.shape[0]):
        print(f"{name}[{i}]", file=output)
        print(shown_tensor[i], file=output)
        print("", file=output)


if __name__ == "__main__":
    output_dir = Path("./")
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = output_dir / "join_kernel.txt"
    op_basis_path = output_dir / "op_bilinear_basis.txt"
    gp_basis_path = output_dir / "gp_bilinear_basis.txt"

    cpu = torch.device("cpu")

    with kernel_path.open("w", encoding="utf-8") as output:
        join_kernel = dual_mod._compute_efficient_join_kernel(cpu, torch.float32)
        print("join kernel:\n", file=output)
        _print_tensor(
            "kernel",
            join_kernel,
            output,
            precision=0,
            display_tensor=join_kernel.detach().cpu().to(torch.int8),
        )

    with op_basis_path.open("w", encoding="utf-8") as output:
        op_basis = linear_mod._load_bilinear_basis("op", cpu, torch.float32)
        print("OP bilinear basis:\n", file=output)
        _print_tensor("basis", op_basis, output, precision=3)

    with gp_basis_path.open("w", encoding="utf-8") as output:
        gp_basis = linear_mod._load_bilinear_basis("gp", cpu, torch.float32)
        print("GP bilinear basis:\n", file=output)
        _print_tensor("basis", gp_basis, output, precision=3)
