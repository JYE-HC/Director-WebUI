# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
from yunchang.kernels import AttnType


COMFY_KITCHEN_INT8 = "COMFY_KITCHEN_INT8"


def attention_backend_choices() -> list[str]:
    """Return all supported RayLight attention backend names.

    Keep the xFuser/yunchang enum order stable so existing serialized workflow
    values retain exactly the same meaning. RayLight-owned backends are
    appended after the upstream choices.
    """

    return [member.name for member in AttnType] + [COMFY_KITCHEN_INT8]


def validate_attention_backend_config(attention: str, ring_degree: int) -> None:
    choices = attention_backend_choices()
    if attention not in choices:
        raise ValueError(
            f"Unknown RayLight attention backend {attention!r}; "
            f"expected one of {choices}"
        )
    if attention == COMFY_KITCHEN_INT8 and int(ring_degree) != 1:
        raise ValueError(
            "COMFY_KITCHEN_INT8 supports Ulysses sequence parallelism only; "
            f"ring_degree must be exactly 1, got {ring_degree}"
        )
