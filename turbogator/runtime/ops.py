import importlib
import os

import torch
from ezgatr.nn.functional import (
    equi_geometric_attention as _py_equi_geometric_attention,
)
from ezgatr.nn.functional import equi_join as _py_equi_join
from ezgatr.nn.functional import geometric_product as _py_geometric_product
from ezgatr.nn.functional import scaler_gated_gelu as _py_scaler_gated_gelu
from ezgatr.nn.functional.linear import equi_linear as _py_equi_linear
from ezgatr.nn.functional.norm import equi_rms_norm as _py_equi_rms_norm

try:
    _ext = importlib.import_module("turbogator_ext")
except Exception:
    _ext = None


_IMPLS = {
    "geometric_product": {"py": _py_geometric_product},
    "equi_join": {"py": _py_equi_join},
    "equi_geometric_attention": {"py": _py_equi_geometric_attention},
    "scaler_gated_gelu": {"py": _py_scaler_gated_gelu},
    "equi_linear": {"py": _py_equi_linear},
    "equi_rms_norm": {"py": _py_equi_rms_norm},
}


def _baseline_geometric_product(a, b):
    return torch.zeros_like(a)


def _baseline_equi_join(a, b, reference=None):
    return torch.zeros_like(a)


def _baseline_equi_geometric_attention(*args, **kwargs):
    q = args[0]
    return torch.zeros_like(q), None


def _baseline_scaler_gated_gelu(x, approximate="tanh"):
    return torch.zeros_like(x)


def _baseline_equi_linear(x, weight, bias=None, normalize_basis=True):
    out_channels = weight.shape[0]
    return torch.zeros(
        *x.shape[:-2],
        out_channels,
        x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )


def _baseline_equi_rms_norm(x, weight=None, eps=None):
    return torch.zeros_like(x)


if _ext is not None:
    if hasattr(_ext, "geometric_product"):
        _IMPLS["geometric_product"]["cpp"] = _ext.geometric_product
    if hasattr(_ext, "equi_join"):
        _IMPLS["equi_join"]["cpp"] = _ext.equi_join
    if hasattr(_ext, "equi_geometric_attention"):
        _IMPLS["equi_geometric_attention"]["cpp"] = _ext.equi_geometric_attention
    if hasattr(_ext, "scaler_gated_gelu"):
        _IMPLS["scaler_gated_gelu"]["cpp"] = _ext.scaler_gated_gelu

    if hasattr(_ext, "geometric_product_baseline"):
        _IMPLS["geometric_product"]["baseline"] = _ext.geometric_product_baseline
    if hasattr(_ext, "geometric_product_vectorized"):
        _IMPLS["geometric_product"]["vectorized"] = _ext.geometric_product_vectorized
    if hasattr(_ext, "equi_join_baseline"):
        _IMPLS["equi_join"]["baseline"] = _ext.equi_join_baseline
    if hasattr(_ext, "equi_geometric_attention_baseline"):
        _IMPLS["equi_geometric_attention"]["baseline"] = (
            _ext.equi_geometric_attention_baseline
        )
    if hasattr(_ext, "scaler_gated_gelu_baseline"):
        _IMPLS["scaler_gated_gelu"]["baseline"] = _ext.scaler_gated_gelu_baseline
    if hasattr(_ext, "equi_linear_baseline"):
        _IMPLS["equi_linear"]["baseline"] = _ext.equi_linear_baseline
    if hasattr(_ext, "equi_rms_norm_baseline"):
        _IMPLS["equi_rms_norm"]["baseline"] = _ext.equi_rms_norm_baseline

if "baseline" not in _IMPLS["geometric_product"]:
    _IMPLS["geometric_product"]["baseline"] = _baseline_geometric_product
if "baseline" not in _IMPLS["equi_join"]:
    _IMPLS["equi_join"]["baseline"] = _baseline_equi_join
if "baseline" not in _IMPLS["equi_geometric_attention"]:
    _IMPLS["equi_geometric_attention"]["baseline"] = _baseline_equi_geometric_attention
if "baseline" not in _IMPLS["scaler_gated_gelu"]:
    _IMPLS["scaler_gated_gelu"]["baseline"] = _baseline_scaler_gated_gelu
if "baseline" not in _IMPLS["equi_linear"]:
    _IMPLS["equi_linear"]["baseline"] = _baseline_equi_linear
if "baseline" not in _IMPLS["equi_rms_norm"]:
    _IMPLS["equi_rms_norm"]["baseline"] = _baseline_equi_rms_norm


_SELECTED = {}
for _op, _choices in _IMPLS.items():
    env_key = "TURBOGATOR_IMPL_" + _op.upper()
    env_choice = os.getenv(env_key)
    if env_choice in _choices:
        _SELECTED[_op] = env_choice
    elif "cpp" in _choices:
        _SELECTED[_op] = "cpp"
    else:
        _SELECTED[_op] = "py"


def available_ops():
    return sorted(_IMPLS.keys())


def available_impls(op_name):
    return sorted(_IMPLS[op_name].keys())


def current_impls():
    return dict(_SELECTED)


def get_impl(op_name):
    return _SELECTED[op_name]


def set_impl(op_name, impl_name):
    if op_name not in _IMPLS:
        raise KeyError("unknown op: " + str(op_name))
    if impl_name not in _IMPLS[op_name]:
        raise KeyError(
            "unknown impl for "
            + str(op_name)
            + ": "
            + str(impl_name)
            + ", available="
            + ",".join(available_impls(op_name))
        )
    _SELECTED[op_name] = impl_name


def set_impls(mapping):
    for op_name, impl_name in mapping.items():
        set_impl(op_name, impl_name)


def register_impl(op_name, impl_name, fn):
    if op_name not in _IMPLS:
        _IMPLS[op_name] = {}
    _IMPLS[op_name][impl_name] = fn
    if op_name not in _SELECTED:
        _SELECTED[op_name] = impl_name


def _dispatch(op_name, *args, **kwargs):
    return _IMPLS[op_name][_SELECTED[op_name]](*args, **kwargs)


def geometric_product(a, b):
    return _dispatch("geometric_product", a, b)


def equi_join(a, b, reference=None):
    return _dispatch("equi_join", a, b, reference)


def equi_geometric_attention(*args, **kwargs):
    return _dispatch("equi_geometric_attention", *args, **kwargs)


def scaler_gated_gelu(x, approximate="tanh"):
    return _dispatch("scaler_gated_gelu", x, approximate)


def equi_linear(x, weight, bias=None, normalize_basis=True):
    return _dispatch("equi_linear", x, weight, bias, normalize_basis)


def equi_rms_norm(x, weight=None, eps=None):
    return _dispatch("equi_rms_norm", x, weight, eps)
