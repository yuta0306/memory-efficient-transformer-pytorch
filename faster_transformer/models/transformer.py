import copy
from typing import Literal, Optional

import torch
import torch.nn as nn
from faster_transformer.models.attention import FasterMultiHeadAttention


class FasterTransformer(nn.Module):
    """Memory-efficient Transformer.
    This uses `FasterMultiHeadAttention` based on \
    `Self-attention Does Not Need $O(n^2)$ Memory <https://arxiv.org/ags/2112.05682>`
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`

    Attributes
    ----------
    d_model : int
    kdim : int
    vdim : int
    _qkv_same_d_model : bool
    nheads : int
    dropout : nn.Dropout
    batch_first : bool
    head_dim : bool
    add_bias_kv : bool
    q_proj : nn.Linear
    k_proj : nn.Linear
    v_proj : nn.Linear
    out_proj : nn.Linear
    add_zero_attn : bool
    query_chunk_size : int
    key_chunk_size : int
    """

    def __init__(
        self,
        d_model: int = 512,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "gelu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(FasterTransformer, self).__init__()

        encoder_layer = FasterTransformerEncoderLayer(
            d_model,
            nheads,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            pre_norm,
            **factory_kwargs
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = FasterTransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = FasterTransformerDecoderLayer(
            d_model,
            nheads,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            pre_norm,
            **factory_kwargs
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = FasterTransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nheads = nheads

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class FasterTransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False
    ):
        super(FasterTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src
            the sequence to the encoder (required).
        mask
            the mask for the src sequence (optional).
        src_key_padding_mask
            the mask for the src keys per batch (optional).
        """
        output = src
        convert_to_nested = False

        for mod in self.layers:
            if convert_to_nested:
                output = mod(output, src_mask=mask)
            else:
                output = mod(
                    output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
                )

        if convert_to_nested:
            # output = output.to_padded_tensor(0.0)
            pass

        if self.norm is not None:
            output = self.norm(output)

        return output


class FasterTransformerEncoderLayer(nn.Module):
    __constants__ = ["pre_norm"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "gelu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(FasterTransformerEncoderLayer, self).__init__()
        self.self_attn = FasterMultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
            return_attention_weights=False,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation == "gelu":
            self.activation: nn.Module = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = nn.ReLU()
        super(FasterTransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src
            the sequence to the encoder layer (required).
        src_mask
            the mask for the src sequence (optional).
        src_key_padding_mask
            the mask for the src keys per batch (optional).
        """
        x = src
        if self.pre_norm:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
