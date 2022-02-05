import functools

import torch
import torch.nn as nn
import torch.utils as utils


class Attention(nn.Module):
    """Memory-efficient multi-head dot product attention.

    Attributes
    ----------
    query_chunk_size : int, default=1024
    key_chunk_size : int, default=4096
    """

    def __init__(
        self,
        *,
        query_chunk_size: int = 1024,
        key_chunk_size: int = 4096,
    ) -> None:
        """Memory-efficient multi-head dot product attention.

        Parameters
        ----------
        query_chunk_size : int, default=1024
        key_chunk_size : int, default=4096
        """
        super(Attention, self).__init__()
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.TensorType,
        value: torch.TensorType,
    ):
        num_q, num_heads, q_features = query.size()

        def _chunk_scanner(chunk_idx, _):
            query_chunk_size = chunk_idx + min(self.query_chunk_size, num_q)
            query_chunk = query[
                chunk_idx:query_chunk_size,
                :num_heads,
                :q_features,
            ]
            return chunk_idx + self.query_chunk_size, self._query_chunk_attention(
                query_chunk, key, value
            )

    def _query_chunk_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        """Multi-head dot product attention with a limited number of queries."""
        num_kv, num_heads, k_features = key.shape
        v_features = value.shape[-1]
        key_chunk_size = min(self.key_chunk_size, num_kv)
        query = query / torch.sqrt(torch.tensor(k_features))

        @functools.partial(utils.checkpoint, prevent_cse=False)
        def summarize_chunk(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ):
            attn_weights: torch.Tensor = torch.einsum("qhd,khd->qhk", query, key)
            max_score = torch.max(attn_weights, dim=-1, keepdim=True)
            print(max_score)
            max_score = max_score.detach()
            exp_weights = torch.exp(attn_weights - max_score)
            exp_values = torch.einsum("vhf,qhv->qhf", value, exp_weights)
            return (
                exp_values,
                exp_weights.sum(dim=-1),
                max_score.reshape((query.size(0), num_heads)),
            )

        def chunk_scanner(chunk_idx, key_chunk_size: int):
            key_chunk_size = chunk_idx + key_chunk_size
            key_chunk = key[
                chunk_idx:key_chunk_size,
                :num_heads,
                :k_features,
            ]
            value_chunk = value[
                chunk_idx:key_chunk_size,
                :num_heads,
                :v_features,
            ]
            return summarize_chunk(query, key_chunk, value_chunk)

        chunk_values, chunk_weights, chunk_max = map(
            lambda chunk_idx: chunk_scanner(
                chunk_idx=chunk_idx, key_chunk_size=key_chunk_size
            ),
            torch.arange(0, num_kv, key_chunk_size),
        )

        global_max = torch.max(chunk_max, dim=0, keepdim=True)
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values *= torch.unsqueeze(max_diffs, dim=-1)
        chunk_weights *= max_diffs

        all_values = chunk_values.sum(dim=0)
        all_weights = torch.unsqueeze(chunk_weights, dim=-1).sum(dim=0)
        return all_values / all_weights

    # def chunk_scanner(chunk_idx, _):
    #     query_chunk = lax.dynamic_slice(
    #     query, (chunk_idx, 0, 0),
    #     slice_sizes=(min(query_chunk_size, num_q), num_heads, q_features))
    #     return (chunk_idx + query_chunk_size,
    #     _query_chunk_attention(query_chunk, key, value, precision=precision))
