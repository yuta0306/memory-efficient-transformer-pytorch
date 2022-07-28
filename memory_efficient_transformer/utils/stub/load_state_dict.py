from collections import OrderedDict

import torch.nn as nn

__all__ = ["load_state_dict"]


def load_state_dict(model: nn.Module, state_dict: OrderedDict) -> nn.Module:
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        if "in_proj_weight" in k:
            base = k.replace(".in_proj_weight", "")
            Wq, Wk, Wv = v.chunk(3, 0)
            state_dict_new[f"{base}.q_proj.weight"] = Wq
            state_dict_new[f"{base}.k_proj.weight"] = Wk
            state_dict_new[f"{base}.v_proj.weight"] = Wv
        elif "in_proj_bias" in k:
            base = k.replace(".in_proj_bias", "")
            bias_q, bias_k, bias_v = v.chunk(3, 0)
            if model.get_submodule(f"{base}.q_proj").bias is not None:
                state_dict_new[f"{base}.q_proj.bias"] = bias_q
            if model.get_submodule(f"{base}.k_proj").bias is not None:
                state_dict_new[f"{base}.k_proj.bias"] = bias_k
            if model.get_submodule(f"{base}.v_proj").bias is not None:
                state_dict_new[f"{base}.v_proj.bias"] = bias_v
        else:
            state_dict_new[k] = v

    model.load_state_dict(state_dict_new, strict=True)
    return model
