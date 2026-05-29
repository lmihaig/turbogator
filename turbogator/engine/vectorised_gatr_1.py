from __future__ import annotations

import math
from functools import reduce

import torch
import torch.nn as nn
from einops import rearrange

# still use the ezgatr config
from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig
from turbogator import cpp_bindings as c_ops


class EquiLinear(nn.Module):
    __constants__ = ["in_channels", "out_channels", "normalize_basis"]

    in_channels: int
    out_channels: int
    normalize_basis: bool
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        normalize_basis: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_basis = normalize_basis
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels, 9), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return c_ops.equi_linear_vectorized(
            x, self.weight, self.bias, self.normalize_basis
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"bias={self.bias is not None}, normalize_basis={self.normalize_basis}"
        )


class EquiRMSNorm(nn.Module):
    __constants__ = ["eps", "channelwise_rescale"]

    in_channels: int
    eps: float | None
    channelwise_rescale: bool
    weight: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        eps: float | None = None,
        channelwise_rescale: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.channelwise_rescale = channelwise_rescale
        if channelwise_rescale:
            self.weight = nn.Parameter(torch.empty(in_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.channelwise_rescale and (self.weight is not None):
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return c_ops.equi_rms_norm_vectorized(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"eps={self.eps}, "
            f"channelwise_rescale={self.channelwise_rescale}"
        )


class MVOnlyGATrEmbedding(nn.Module):
    config: MVOnlyGATrConfig
    embedding: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = EquiLinear(
            config.size_channels_in, config.size_channels_hidden
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.config.size_channels_in:
            raise ValueError(
                f"Input tensor has {x.shape[-2]} channels, "
                f"expected {self.config.size_channels_in}."
            )
        return self.embedding(x)


class MVOnlyGATrBilinear(nn.Module):
    config: MVOnlyGATrConfig
    proj_bil: EquiLinear
    proj_out: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.proj_bil = EquiLinear(
            config.size_channels_hidden, config.size_channels_intermediate * 4
        )
        self.proj_out = EquiLinear(
            config.size_channels_intermediate * 2, config.size_channels_hidden
        )

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor | None = None
    ) -> torch.Tensor:
        size_inter = self.config.size_channels_intermediate
        lg, rg, lj, rj = torch.split(self.proj_bil(x), size_inter, dim=-2)

        x = torch.cat(
            [
                c_ops.geometric_product_vectorized(lg, rg),
                c_ops.equi_join_vectorized(lj, rj, reference),
            ],
            dim=-2,
        )
        return self.proj_out(x)


class MVOnlyGATrMLP(nn.Module):
    config: MVOnlyGATrConfig
    layer_norm: EquiRMSNorm
    equi_bil: MVOnlyGATrBilinear
    proj_out: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.layer_norm = EquiRMSNorm(
            config.size_channels_hidden,
            eps=config.norm_eps,
            channelwise_rescale=config.norm_channelwise_rescale,
        )
        self.equi_bil = MVOnlyGATrBilinear(config)
        self.proj_out = EquiLinear(
            config.size_channels_hidden, config.size_channels_hidden
        )

    def forward(
        self, x: torch.Tensor, reference: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x

        x = self.layer_norm(x)
        x = self.equi_bil(x, reference)
        x = self.proj_out(
            c_ops.scaler_gated_gelu_vectorized(x, self.config.gelu_approximate)
        )

        return x + residual


class MVOnlyGATrAttention(nn.Module):
    config: MVOnlyGATrConfig
    layer_norm: EquiRMSNorm
    attn_mix: dict[str, torch.Tensor]
    proj_qkv: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.layer_norm = EquiRMSNorm(
            config.size_channels_hidden,
            eps=config.norm_eps,
            channelwise_rescale=config.norm_channelwise_rescale,
        )

        # The two dummy dimensions are for the sequence length
        # and blade dimension, respectively.
        attn_mix_shape = (config.attn_num_heads, 1, config.size_channels_hidden, 1)
        self.attn_mix = {}
        for kind in config.attn_kinds.keys():
            param = nn.Parameter(torch.zeros(attn_mix_shape, dtype=torch.float32))
            self.attn_mix[kind] = param
            self.register_parameter(f"attn_mix_{kind}", param)

        self.proj_qkv = EquiLinear(
            config.size_channels_hidden,
            config.size_channels_hidden * config.attn_num_heads * 3,
        )
        self.proj_out = EquiLinear(
            config.size_channels_hidden * config.attn_num_heads,
            config.size_channels_hidden,
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x

        x = self.layer_norm(x)
        q, k, v = rearrange(
            self.proj_qkv(x),
            "b t (qkv h c) k -> qkv b h t c k",
            qkv=3,
            h=self.config.attn_num_heads,
            c=self.config.size_channels_hidden,
        )
        x, _ = c_ops.equi_geometric_attention_vectorized(
            q,
            k,
            v,
            kinds=self.config.attn_kinds,
            weight=[w.exp() for w in self.attn_mix.values()],
            attn_mask=attn_mask,
            is_causal=self.config.attn_is_causal,
            dropout_p=self.config.attn_dropout_p,
            scale=self.config.attn_scale,
        )
        x = rearrange(x, "b h t c k -> b t (h c) k", h=self.config.attn_num_heads)
        x = self.proj_out(x)

        return x + residual


class MVOnlyGATrBlock(nn.Module):
    config: MVOnlyGATrConfig
    layer_id: int
    mlp: MVOnlyGATrMLP
    attn: MVOnlyGATrAttention

    def __init__(self, config: MVOnlyGATrConfig, layer_id: int) -> None:
        super().__init__()

        self.config = config
        self.layer_id = layer_id

        self.mlp = MVOnlyGATrMLP(config)
        self.attn = MVOnlyGATrAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        reference: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mlp(self.attn(x, attn_mask), reference)


class VectorisedGATrModel(nn.Module):
    config: MVOnlyGATrConfig
    embedding: MVOnlyGATrEmbedding
    blocks: nn.ModuleList
    head: EquiLinear

    def __init__(self, config: MVOnlyGATrConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = MVOnlyGATrEmbedding(config)
        self.blocks = nn.ModuleList(
            MVOnlyGATrBlock(config, i) for i in range(config.num_layers)
        )
        self.head = EquiLinear(config.size_channels_hidden, config.size_channels_out)
        self.apply(self._init_params)

    def _init_params(self, module: nn.Module):
        if isinstance(module, EquiLinear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            module.weight.data /= math.sqrt(self.config.num_layers)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        reference: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference = reference or torch.mean(
            x,
            dim=tuple(range(1, len(x.shape) - 1)),
            keepdim=True,
        )
        x = reduce(
            lambda x, block: block(x, reference, attn_mask),
            self.blocks,
            self.embedding(x),
        )
        return self.head(x)
