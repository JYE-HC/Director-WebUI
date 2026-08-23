import types

from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


def _chunk_cfg_batch(value, cfg_world_size, cfg_rank):
    if value is not None and value.ndim > 0 and value.shape[0] == cfg_world_size:
        return value.chunk(cfg_world_size, dim=0)[cfg_rank]
    return value


def cfg_parallel_forward(
    self,
    hidden_states,
    timestep,
    context=None,
    encoder_attention_mask=None,
    attention_mask=None,
    transformer_options={},
    **kwargs,
):
    cfg_world_size = get_classifier_free_guidance_world_size()
    if hidden_states.ndim == 0 or hidden_states.shape[0] != cfg_world_size:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")

    cfg_rank = get_classifier_free_guidance_rank()
    result = self._raylight_cfg_original_forward(
        _chunk_cfg_batch(hidden_states, cfg_world_size, cfg_rank),
        _chunk_cfg_batch(timestep, cfg_world_size, cfg_rank),
        context=_chunk_cfg_batch(context, cfg_world_size, cfg_rank),
        encoder_attention_mask=_chunk_cfg_batch(encoder_attention_mask, cfg_world_size, cfg_rank),
        attention_mask=_chunk_cfg_batch(attention_mask, cfg_world_size, cfg_rank),
        transformer_options=transformer_options,
        **kwargs,
    )
    return get_cfg_group().all_gather(result, dim=0)


def patch_cfg_forward(model):
    if getattr(model.forward, "__func__", None) is cfg_parallel_forward:
        return
    model._raylight_cfg_original_forward = model.forward
    model.forward = types.MethodType(cfg_parallel_forward, model)
