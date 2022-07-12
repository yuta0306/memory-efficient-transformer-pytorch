import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ..utils import dynamic_slice, map_, scan


class ScaledDotProductAttention(nn.Module):
    """ScaledDotProductAttention"""

    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        k_features = key.size(-1)
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(k_features)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.softmax(scores, dim=-1)
        return attention.matmul(value)


class FasterMultiHeadAttention(nn.Module):
    """Memory-efficient multi-head dot product attention.
    `Self-attention Does Not Need $O(n^2)$ Memory <https://arxiv.org/ags/2112.05682>`
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`

    Attributes
    ----------
    embed_dim : int
    kdim : int
    vdim : int
    _qkv_same_embed_dim : bool
    num_heads : int
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

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,  # False as default in torch.nn.MultiHeadAttention
        device: Optional[str] = None,
        dtype: Optional[type] = None,
        query_chunk_size: int = 1024,
        key_chunk_size: int = 4096,
        return_attention_weights: bool = False,
    ) -> None:
        """Memory-efficient multi-head dot product attention.

        Parameters
        ----------
        embed_dim : int
        num_heads : int
        dropout : float = 0.0
        bias : bool, default=True
        add_bias_kv : bool, default=False
        add_zero_attn : bool, default=False
        kdim : optional, int
        vdim : optional, int
        batch_first : bool, default=True
        device : optional, str
        dtype : optional, type
        query_chunk_size : int, default=1024
        key_chunk_size : int, default=4096
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(FasterMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        self.add_bias_kv = add_bias_kv
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # if self._qkv_same_embed_dim is False:
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(
            embed_dim, self.kdim, bias=add_bias_kv, **factory_kwargs
        )
        self.v_proj = nn.Linear(
            embed_dim, self.vdim, bias=add_bias_kv, **factory_kwargs
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

        # faster-transformer settings
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size

        self.return_attention_weights = return_attention_weights

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.add_bias_kv:
            nn.init.xavier_normal_(self.k_proj.bias)
            nn.init.xavier_normal_(self.v_proj.bias)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, None]]:
        """
        Parameters
        ----------
        query : torch.Tensor
            Query embeddings of shape :math:`(N, L, E_q)`, where :math:`N` is the batch size,\
            :math:`L` is the target sequence length, and :math:`E_q` is the query embedding dimension ``embed_dim``.
        key : torch.Tensor
            Key embeddings of shape :math:`(N, L, E_k)` where :math:`N` is the batch size,\
            :math:`L` is the source sequence length, and :math:`E_k` is the key embedding dimension ``embed_dim``.
        value : torch.Tensor
            Key embeddings of shape :math:`(N, L, E_v)` where :math:`N` is the batch size,\
            :math:`L` is the source sequence length, and :math:`E_v` is the value embedding dimension ``embed_dim``.
        mask : optional, torch.Tensor
            If specified, a 2D preventing attention to certain positions. Must be of shape\
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,\
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be\
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.

        """
        query = self.q_proj(query)  # query projection
        key = self.k_proj(key)  # key projection
        value = self.v_proj(value)  # value projection

        B, num_q, _ = query.size()
        query = query.contiguous().view(B, -1, self.num_heads, self.head_dim)
        key = key.contiguous().view(B, -1, self.num_heads, self.head_dim)
        value = value.contiguous().view(B, -1, self.num_heads, self.head_dim)

        def _chunk_scanner(chunk_idx, _):
            query_chunk = dynamic_slice(
                query,
                (0, chunk_idx, 0, 0),
                sizes=(
                    B,
                    min(self.query_chunk_size, num_q),
                    self.num_heads,
                    self.head_dim,
                ),
            )
            return (
                chunk_idx + self.query_chunk_size,
                self._query_chunk_attention(query_chunk, key, value),
            )

        _, res = scan(
            _chunk_scanner,
            init=0,
            xs=None,
            length=math.ceil(num_q / self.query_chunk_size),
        )
        outputs = res.transpose(0, 1).contiguous().view(B, num_q, -1)
        outputs = self.out_proj(outputs)
        if self.return_attention_weights:
            return outputs, None
        return outputs

    def _query_chunk_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Multi-head dot product attention with a limited number of queries."""
        B, num_kv, num_heads, k_features = key.shape
        v_features = value.size(-1)
        key_chunk_size = min(self.key_chunk_size, num_kv)
        query = query / torch.sqrt(torch.tensor(k_features))

        # @functools.partial(checkpoint.checkpoint, preserve_rng_state=True)
        def summarize_chunk(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ):
            # B, L = query.size(0), query.size(1)
            attn_weights = torch.einsum("...qhd,...khd->...qhk", query, key)
            if mask is not None:
                big_neg = torch.tensor(
                    torch.finfo(attn_weights.dtype).min,
                    device=mask.device,
                    dtype=torch.float32,
                )
                mask = torch.einsum("...hqk->...qhk", mask)
                attn_weights = torch.where(mask, attn_weights, big_neg)
            max_score, _ = torch.max(attn_weights, dim=-1, keepdim=True)
            max_score = max_score.detach()
            exp_weights = torch.exp(attn_weights - max_score)
            exp_values = torch.einsum("...vhf,...qhv->...qhf", value, exp_weights)
            max_score = torch.einsum("...qhk->...qh", max_score)
            return exp_values, exp_weights.sum(dim=-1), max_score

        def chunk_scanner(chunk_idx):
            key_chunk = dynamic_slice(
                key,
                (0, chunk_idx, 0, 0),
                sizes=(B, key_chunk_size, num_heads, k_features),
            )
            value_chunk = dynamic_slice(
                value,
                (0, chunk_idx, 0, 0),
                sizes=(B, key_chunk_size, num_heads, v_features),
            )
            return checkpoint.checkpoint(summarize_chunk, query, key_chunk, value_chunk)

        chunk_values, chunk_weights, chunk_max = map_(
            chunk_scanner,
            torch.arange(0, num_kv, key_chunk_size),
        )

        global_max, _ = torch.max(chunk_max, dim=0, keepdim=True)
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values *= torch.unsqueeze(max_diffs, dim=-1)
        chunk_weights *= max_diffs

        all_values = chunk_values.sum(dim=0)
        all_weights = torch.unsqueeze(chunk_weights, dim=-1).sum(dim=0)
        return all_values / all_weights
