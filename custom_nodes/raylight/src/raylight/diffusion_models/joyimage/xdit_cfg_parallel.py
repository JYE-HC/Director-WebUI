from raylight.distributed_modules.cfg_utils import cfg_parallel_forward


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    return cfg_parallel_forward(
        executor,
        *args,
        chunk_names=("hidden_states", "timestep", "context", "ref_latents"),
        validate_name="hidden_states",
        **kwargs,
    )
