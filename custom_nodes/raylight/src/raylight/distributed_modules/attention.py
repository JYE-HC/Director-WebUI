# Modified for Director Web; see DIRECTOR_MODIFICATIONS.md.
from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.globals import PROCESS_GROUP
from yunchang.kernels import AttnType

from .attention_backends import COMFY_KITCHEN_INT8, validate_attention_backend_config
from .sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

_ATTN_TYPE = None
_SYNC_ULYSSES = None
_RING_DEGREE = None


def set_attn_type(attn):
    global _ATTN_TYPE
    _ATTN_TYPE = attn


def get_attn_type():
    if _ATTN_TYPE is None:
        raise RuntimeError("_ATTN_TYPE is not initialized")
    else:
        return _ATTN_TYPE


def set_sync_ulysses(is_sync):
    global _SYNC_ULYSSES
    _SYNC_ULYSSES = is_sync


def get_sync_ulysses():
    if _SYNC_ULYSSES is None:
        raise RuntimeError("_SYNC_ULYSSES variable is not initialized")
    else:
        return _SYNC_ULYSSES


def set_ring_degree(ring_degree):
    global _RING_DEGREE
    _RING_DEGREE = int(ring_degree)


def get_ring_degree():
    if _RING_DEGREE is None:
        raise RuntimeError("_RING_DEGREE is not initialized")
    return _RING_DEGREE


def _load_comfy_kitchen_int8_attention():
    """Lazy-load CK so legacy attention backends do not depend on its API."""
    try:
        from comfy_kitchen import int8_attention, int8_attention_is_available
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "COMFY_KITCHEN_INT8 requires a comfy-kitchen build that exports "
            "int8_attention and int8_attention_is_available; no fallback was used"
        ) from exc
    return int8_attention, int8_attention_is_available


def _current_worker_cuda_device():
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    return device, torch.cuda.get_device_capability(device)


def _require_neutral_ck_argument(name, value, neutral_values):
    if value not in neutral_values:
        raise ValueError(
            f"COMFY_KITCHEN_INT8 does not support {name}={value!r}; "
            "no fallback was used"
        )


class ComfyKitchenInt8UlyssesAttention:
    """Ulysses-only adapter from xFuser's NHD layout to CK's BHSD layout."""

    def __init__(self, use_sync, ring_degree):
        validate_attention_backend_config(COMFY_KITCHEN_INT8, ring_degree)
        int8_attention, is_available = _load_comfy_kitchen_int8_attention()
        try:
            device, capability = _current_worker_cuda_device()
            available = is_available(device)
        except Exception as exc:
            raise RuntimeError(
                "COMFY_KITCHEN_INT8 availability check failed on this Ray worker/device; "
                "no fallback was used"
            ) from exc
        sm = f"SM{capability[0]}{capability[1]}"
        if not available:
            raise RuntimeError(
                f"COMFY_KITCHEN_INT8 is unavailable on Ray worker device {device} ({sm}); "
                "no fallback was used"
            )
        if PROCESS_GROUP.ULYSSES_PG is None:
            raise RuntimeError(
                "COMFY_KITCHEN_INT8 requires an initialized Ulysses process group; "
                "no fallback was used"
            )
        self.int8_attention = int8_attention
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG
        self.use_sync = use_sync
        print(f"COMFY_KITCHEN_INT8 available on Ray worker device {device} ({sm})")

    def __call__(
        self,
        attn,
        query,
        key,
        value,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
    ):
        _require_neutral_ck_argument("attention module", attn, (None,))
        _require_neutral_ck_argument("dropout_p", dropout_p, (None, 0, 0.0))
        _require_neutral_ck_argument("causal", causal, (None, False))
        _require_neutral_ck_argument("window_size", window_size, (None, (-1, -1)))
        _require_neutral_ck_argument("alibi_slopes", alibi_slopes, (None,))
        _require_neutral_ck_argument("deterministic", deterministic, (None, False))
        _require_neutral_ck_argument("return_attn_probs", return_attn_probs, (None, False))
        _require_neutral_ck_argument("joint_strategy", joint_strategy, (None, "none"))
        for name, tensor in (
            ("joint_tensor_query", joint_tensor_query),
            ("joint_tensor_key", joint_tensor_key),
            ("joint_tensor_value", joint_tensor_value),
        ):
            _require_neutral_ck_argument(name, tensor, (None,))

        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.ndim != 4:
                raise ValueError(
                    f"COMFY_KITCHEN_INT8 expected {name} in NHD layout with 4 dimensions, "
                    f"got shape {tuple(tensor.shape)}"
                )

        # [B, local sequence, heads, dim] -> [B, full sequence, local heads, dim]
        query = SeqAllToAll4D.apply(self.ulysses_pg, query, 2, 1, self.use_sync)
        key = SeqAllToAll4D.apply(self.ulysses_pg, key, 2, 1, self.use_sync)
        value = SeqAllToAll4D.apply(self.ulysses_pg, value, 2, 1, self.use_sync)

        # comfy-kitchen INT8 attention consumes and returns [B, heads, sequence, dim].
        query_bhsd = query.transpose(1, 2).contiguous()
        key_bhsd = key.transpose(1, 2).contiguous()
        value_bhsd = value.transpose(1, 2).contiguous()
        output = self.int8_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            scale=softmax_scale,
        )
        expected_contract = (tuple(query_bhsd.shape), query_bhsd.dtype, query_bhsd.device)
        actual_contract = (
            tuple(getattr(output, "shape", ())),
            getattr(output, "dtype", None),
            getattr(output, "device", None),
        )
        if actual_contract != expected_contract:
            raise RuntimeError(
                "COMFY_KITCHEN_INT8 returned an invalid BHSD output contract; "
                f"expected shape/dtype/device={expected_contract}, got {actual_contract}; "
                "no fallback was used"
            )

        # [B, full sequence, local heads, dim] -> [B, local sequence, heads, dim]
        output = output.transpose(1, 2).contiguous()
        return SeqAllToAll4D.apply(self.ulysses_pg, output, 1, 2, self.use_sync)


def _validate_ck_wrapper_contract(join_q, join_k, join_v, mask, attn_precision, args, kwargs):
    for name, value in (("join_q", join_q), ("join_k", join_k), ("join_v", join_v), ("mask", mask)):
        _require_neutral_ck_argument(name, value, (None,))
    _require_neutral_ck_argument("attn_precision", attn_precision, (None,))
    if args:
        raise ValueError(
            "COMFY_KITCHEN_INT8 does not support extra positional attention arguments; "
            "no fallback was used"
        )

    neutral_kwargs = {
        "attn_mask": (None,),
        "causal": (None, False),
        "is_causal": (None, False),
        "dropout": (None, 0, 0.0),
        "dropout_p": (None, 0, 0.0),
        "window_size": (None, (-1, -1)),
        "alibi_slopes": (None,),
        "deterministic": (None, False),
        "return_attn_probs": (None, False),
        "joint_strategy": (None, "none"),
        "joint_tensor_query": (None,),
        "joint_tensor_key": (None,),
        "joint_tensor_value": (None,),
        "softcap": (None, 0, 0.0),
        "enable_gqa": (None, False),
    }
    for name, neutral_values in neutral_kwargs.items():
        if name in kwargs:
            _require_neutral_ck_argument(name, kwargs[name], neutral_values)

    allowed_kwargs = {"scale", "transformer_options", *neutral_kwargs}
    unsupported = sorted(set(kwargs) - allowed_kwargs)
    if unsupported:
        raise ValueError(
            "COMFY_KITCHEN_INT8 does not support attention arguments "
            f"{unsupported}; no fallback was used"
        )


def make_xfuser_attention(attn_type, sync_ulysses, ring_degree=None):
    print(f"Using XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}")
    if attn_type == COMFY_KITCHEN_INT8:
        effective_ring_degree = get_ring_degree() if ring_degree is None else ring_degree
        xfuser_attn = ComfyKitchenInt8UlyssesAttention(sync_ulysses, effective_ring_degree)
    else:
        attn = AttnType[attn_type]
        if attn_type == "SAGE_FP8_CUDA":
            ensure_hf_fp8_cuda_kernel()
        elif attn_type == "SAGE_FP8_SM90":
            ensure_hf_sm90_kernel

        xfuser_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=attn)

    def _attention_xfuser_unmask(
            q,
            k,
            v,
            heads,
            join_q=None,
            join_k=None,
            join_v=None,
            mask=None,
            attn_precision=None,
            skip_reshape=False,
            skip_output_reshape=False,
            *args,
            **kwargs):

        if attn_type == COMFY_KITCHEN_INT8:
            _validate_ck_wrapper_contract(join_q, join_k, join_v, mask, attn_precision, args, kwargs)

        if skip_reshape:
            b, _, _, dim_head = q.shape
            if join_q is not None:
                j_b, _, _, j_dim_head = join_q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )
            if join_q is not None:
                j_b, _, j_dim_head = join_q.shape
                j_dim_head //= heads
                join_q, join_k, join_v = map(
                    lambda t: t.view(j_b, -1, heads, j_dim_head).transpose(1, 2),
                    (join_q, join_k, join_v),
                )

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        # Check if using join attention, for MMDiT model
        if join_q is not None:
            out = xfuser_attn(
                None,
                query,
                key,
                value,
                joint_strategy="rear",
                joint_tensor_query=join_q.transpose(1, 2),
                joint_tensor_key=join_k.transpose(1, 2),
                joint_tensor_value=join_v.transpose(1, 2),
                softmax_scale=kwargs.get("scale", None),
            ).transpose(1, 2)
        else:
            out = xfuser_attn(
                None,
                query,
                key,
                value,
                softmax_scale=kwargs.get("scale", None),
            ).transpose(1, 2)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
        return out

    return _attention_xfuser_unmask
