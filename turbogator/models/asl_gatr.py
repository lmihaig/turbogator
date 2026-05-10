from ezgatr.nets import mv_only_gatr as base
from ezgatr.nets.mv_only_gatr import MVOnlyGATrModel
from ezgatr.nn.functional import linear as linear_func
from ezgatr.nn.functional import norm as norm_func

from turbogator.runtime import ops


def patch_ops(impls=None):
    if impls:
        ops.set_impls(impls)

    base.geometric_product = ops.geometric_product
    base.equi_join = ops.equi_join
    base.equi_geometric_attention = ops.equi_geometric_attention
    base.scaler_gated_gelu = ops.scaler_gated_gelu
    linear_func.equi_linear = ops.equi_linear
    norm_func.equi_rms_norm = ops.equi_rms_norm


def set_model_impl(op_name, impl_name):
    ops.set_impl(op_name, impl_name)


def set_model_impls(mapping):
    ops.set_impls(mapping)


def get_model_impls():
    return ops.current_impls()


def register_model_impl(op_name, impl_name, fn):
    ops.register_impl(op_name, impl_name, fn)


class TurboGatorModel(MVOnlyGATrModel):
    def __init__(self, config, impls=None):
        patch_ops(impls)
        super().__init__(config)
