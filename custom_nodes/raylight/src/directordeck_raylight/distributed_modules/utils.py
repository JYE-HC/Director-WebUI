import torch
from torch.distributed._tensor import DTensor


def detect_dtype_mismatch(module, ref_dtype):
    ignored_param = set()
    for name, param in module.named_parameters(recurse=True):
        if param.dtype != ref_dtype:
            ignored_param.add(param)

    return ignored_param


def ensure_no_scalar(module):
    for mod in module.modules():
        for name, param in list(mod._parameters.items()):
            if param is None:
                continue
            if param.ndim == 0:
                new_param = torch.nn.Parameter(param.detach().unsqueeze(0))
                new_param.requires_grad = param.requires_grad
                mod._parameters[name] = new_param

        for name, buf in list(mod._buffers.items()):
            if buf is None:
                continue
            if isinstance(buf, torch.Tensor) and buf.ndim == 0:
                new_buf = buf.detach().unsqueeze(0)
                mod._buffers[name] = new_buf

    return module


def adjust_state_dict_scalars(state_dict):
    for k, v in list(state_dict.items()):
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            state_dict[k] = v.unsqueeze(0)
    return state_dict


def inspect_tensor(t):
    if isinstance(t, DTensor):
        print("=== DTensor Info ===")
        print("Global shape:", t.shape)
        print("Local shape:", t.to_local().shape)
        print("Device mesh:", t.device_mesh)
        print("Placement:", t.placements)
        print("====================")
    else:
        print("Regular Tensor:", t.shape, t.device)
