import torch
import torch.nn as nn
import types
import os
from unittest.mock import patch
from contextlib import nullcontext

import sys
from pathlib import Path

RAYLIGHT_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = Path(
    os.environ.get("COMFYUI_ROOT", str(RAYLIGHT_ROOT.parents[1]))
).expanduser().resolve()
sys.path.insert(0, str(RAYLIGHT_ROOT / "src"))
sys.path.insert(0, str(COMFYUI_ROOT))

from directordeck_raylight.distributed_worker.ray_worker_vae import (
    ray_vae_decode_partial_impl,
    combine_dist_vae_partials,
    _get_upscale_func,
    _round_upscale,
    _get_pos_func,
    _compute_feather,
    load_vae_model,
    _normalize_worker_result,
    _validate_worker_passes,
)


def test_load_vae_model_rejects_invalid_vae():
    import tempfile
    import os
    import comfy.utils as comfy_utils

    state_dict = {"default": torch.randn(1, 3, 3, 3)}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.safetensors")
        comfy_utils.save_torch_file(state_dict, path)
        from directordeck_raylight.distributed_worker.ray_worker_vae import load_vae_model
        try:
            vae = load_vae_model(path)
            assert False, "Expected RuntimeError for invalid VAE"
        except RuntimeError as e:
            assert "invalid" in str(e).lower()


def test_load_vae_model_sets_chunked_io_flag():
    import types as py_types

    metadata = {"config": "{}"}
    captured = {}
    fake_comfy = py_types.ModuleType("comfy")
    fake_comfy.sd = py_types.ModuleType("comfy.sd")
    fake_comfy.utils = py_types.ModuleType("comfy.utils")

    class FakeVAE:
        def __init__(self, sd=None, metadata=None):
            captured["sd"] = sd
            captured["metadata"] = metadata
            self.first_stage_model = types.SimpleNamespace(comfy_has_chunked_io=True)
            self.throw_exception_if_invalid = lambda: None

    fake_comfy.sd.VAE = FakeVAE
    fake_comfy.utils.load_torch_file = lambda path, return_metadata=False: ({"weight": torch.ones(1)}, metadata)

    modules = {
        "comfy": fake_comfy,
        "comfy.sd": fake_comfy.sd,
        "comfy.utils": fake_comfy.utils,
    }

    with patch.dict(sys.modules, modules):
        vae = load_vae_model("test_vae.safetensors")

    assert vae._raylight_comfy_has_chunked_io is True
    assert set(captured["sd"]) == {"weight"}
    assert captured["metadata"] is metadata


def test_normalize_seedvr2_latent_accepts_native_and_collapsed_layouts():
    from directordeck_raylight.distributed_worker.ray_worker_vae import normalize_seedvr2_latent

    native_5d = torch.randn(2, 4, 8, 64, 64)
    result = normalize_seedvr2_latent(native_5d, 4)
    assert result.shape == (2, 4, 8, 64, 64)

    collapsed_4d = torch.randn(2, 32, 64, 64)
    result = normalize_seedvr2_latent(collapsed_4d, 4)
    assert result.shape == (2, 4, 8, 64, 64)


def test_seedvr2_spatial_tile_ranges_match_owned_tiler_edges():
    from directordeck_raylight.distributed_worker.ray_worker_vae import seedvr2_spatial_tile_ranges

    ranges, overlap = seedvr2_spatial_tile_ranges(64, 64, 16, 4)
    assert len(ranges) > 0
    for y, y_end, x, x_end in ranges:
        assert 0 <= y < y_end <= 64
        assert 0 <= x < x_end <= 64
    assert overlap == 4


def test_combine_seedvr2_partials_reconstructs_two_ranks_without_temporal_blending():
    from directordeck_raylight.distributed_worker.ray_worker_vae import combine_seedvr2_vae_partials

    latent_h, latent_w = 8, 8
    spatial_scale = 8
    output_h = latent_h * spatial_scale
    output_w = latent_w * spatial_scale
    output_shape = (1, 3, 8, output_h, output_w)

    # Tiles overlap in output columns 24..40, matching SeedVR2 spatial feathering.
    tile_a = torch.ones((1, 3, 8, 64, 40)) * 1.0
    tile_b = torch.ones((1, 3, 8, 64, 40)) * 2.0

    partial_a = {
        "output_shape": output_shape,
        "latent_spatial_shape": (latent_h, latent_w),
        "spatial_scale": spatial_scale,
        "overlap": 2,
        "tiles": [(0, 0, 8, 0, 5, tile_a)],
    }
    partial_b = {
        "output_shape": output_shape,
        "latent_spatial_shape": (latent_h, latent_w),
        "spatial_scale": spatial_scale,
        "overlap": 2,
        "tiles": [(0, 0, 8, 3, 8, tile_b)],
    }

    combined = combine_seedvr2_vae_partials([partial_a, partial_b])
    assert combined.shape == (1, 3, 8, output_h, output_w)


def test_seedvr2_workers_keep_the_complete_temporal_sequence():
    import comfy.model_management as model_management
    from comfy.ldm.seedvr.vae import VideoAutoencoderKLWrapper

    model = VideoAutoencoderKLWrapper.__new__(VideoAutoencoderKLWrapper)
    nn.Module.__init__(model)
    model.spatial_downsample_factor = 2
    model.temporal_downsample_factor = 4
    decoded_shapes = []

    def decode(tile):
        decoded_shapes.append(tuple(tile.shape))
        return torch.ones((1, 3, tile.shape[2] * 4 - 3, tile.shape[3] * 2, tile.shape[4] * 2))

    model.decode = decode
    vae = types.SimpleNamespace(
        first_stage_model=model,
        latent_channels=32,
        output_channels=3,
        vae_dtype=torch.float32,
        device=torch.device("cpu"),
        patcher=object(),
        disable_offload=True,
        memory_used_decode=lambda shape, dtype: 1,
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 32, 3, 4, 6)}

    with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
        model_management, "cuda_device_context", lambda device: nullcontext()
    ):
        from directordeck_raylight.distributed_worker.ray_worker_vae import ray_seedvr2_vae_decode_partial_impl, combine_seedvr2_vae_partials
        partials = [
            ray_seedvr2_vae_decode_partial_impl(worker, samples, tile_size=8, overlap=4, job_rank=rank, job_world_size=2)
            for rank in range(2)
        ]

    assert decoded_shapes == [(1, 32, 3, 4, 4), (1, 32, 3, 4, 4)]
    assert combine_seedvr2_vae_partials(partials).shape == (1, 3, 9, 8, 12)


def test_generic_distvae_rejects_chunked_io_vaes():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(decode=lambda tile: tile),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
        _raylight_comfy_has_chunked_io=True,
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 3, 64)}

    try:
        with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
            model_management, "cuda_device_context", lambda device: nullcontext()
        ):
            ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=1)
        assert False, "Expected ValueError for chunked-IO VAE"
    except ValueError as e:
        assert "chunked" in str(e).lower()


def test_generic_distvae_rejects_handles_tiling():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(decode=lambda tile: tile),
        handles_tiling=True,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 3, 64)}

    try:
        with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
            model_management, "cuda_device_context", lambda device: nullcontext()
        ):
            ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=1)
        assert False, "Expected ValueError for handles_tiling VAE"
    except ValueError as e:
        assert "externally tile" in str(e).lower()


def test_generic_distvae_rejects_structured_nested_samples():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(decode=lambda tile: tile),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.nested.nested_tensor([torch.zeros(1, 3, 64)])}

    try:
        with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
            model_management, "cuda_device_context", lambda device: nullcontext()
        ):
            ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=1)
        assert False, "Expected ValueError for nested latents"
    except ValueError as e:
        assert "nested" in str(e).lower()


def test_generic_distvae_rejects_nested_torch_tensor():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(decode=lambda tile: tile),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.nested.nested_tensor([torch.zeros(1, 3, 64)])}

    try:
        with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
            model_management, "cuda_device_context", lambda device: nullcontext()
        ):
            ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=1)
        assert False, "Expected ValueError for nested torch tensor"
    except (ValueError, TypeError) as e:
        assert "nested" in str(e).lower()


def test_generic_distvae_rejects_invalid_rank():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(decode=lambda tile: tile),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 3, 64)}

    try:
        with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
            model_management, "cuda_device_context", lambda device: nullcontext()
        ):
            ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=2, job_world_size=2)
        assert False, "Expected ValueError for invalid rank"
    except ValueError as e:
        assert "rank" in str(e).lower()


def test_generic_distvae_3d_never_externally_temporal_tiles():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(
            decode=lambda tile: torch.ones((1, 3, tile.shape[2] * 4 - 3, tile.shape[3] * 8, tile.shape[4] * 8)),
        ),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=(lambda a: max(0, a * 4 - 3), 8, 8),
        upscale_index_formula=(4, 8, 8),
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 3, 8, 64, 64)}

    with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
        model_management, "cuda_device_context", lambda device: nullcontext()
    ):
        partials = ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=1)

    for pass_dict in partials:
        for tile in pass_dict["tiles"]:
            decoded = tile[-1]
            assert decoded.shape[2] == 8 * 4 - 3, "3D generic VAE must include full temporal sequence in every tile"


def test_generic_distvae_3d_coverage_matches_output():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(
            decode=lambda tile: torch.ones((1, 3, tile.shape[2] * 4 - 3, tile.shape[3] * 8, tile.shape[4] * 8)),
        ),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=(lambda a: max(0, a * 4 - 3), 8, 8),
        upscale_index_formula=(4, 8, 8),
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 3, 8, 64, 64)}

    with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
        model_management, "cuda_device_context", lambda device: nullcontext()
    ):
        partials = ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=1)

    combined = combine_dist_vae_partials(partials)
    assert combined.shape[0] == 1
    assert combined.shape[2] == 8 * 4 - 3


def test_generic_distvae_2d_three_pass_averaging():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(
            decode=lambda tile: torch.ones((1, 3, tile.shape[2] * 8, tile.shape[3] * 8)),
        ),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 3, 64, 64)}

    with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
        model_management, "cuda_device_context", lambda device: nullcontext()
    ):
        partials = ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=1)

    assert len(partials) == 3, "2D generic VAE must produce exactly 3 passes"
    combined = combine_dist_vae_partials([partials])
    assert combined.shape[0] == 1
    assert combined.shape[2] == 64 * 8


def test_generic_distvae_1d_extra_channel_decoder_input_shape():
    import comfy.model_management as model_management

    decoder_input_shapes = []

    def mock_decode(tile):
        decoder_input_shapes.append(tile.shape)
        return torch.ones((tile.shape[0], tile.shape[1] // 16, tile.shape[2]))

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(decode=mock_decode),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=16,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 2, 16, 64)}

    with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
        model_management, "cuda_device_context", lambda device: nullcontext()
    ):
        partials = ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=1)

    assert len(decoder_input_shapes) > 0, "Decoder must have been called"
    for shape in decoder_input_shapes:
        assert shape[1] == 2, f"Decoder input must preserve C=2 channels, got shape {shape}"
        assert shape[2] == 16, f"Decoder input must preserve extra_channels=16, got shape {shape}"


def test_sparse_reconstruction_two_ranks_overlapping_2d():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(
            decode=lambda tile: torch.ones((1, 3, tile.shape[2] * 8, tile.shape[3] * 8)),
        ),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=None,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 3, 64, 64)}

    with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
        model_management, "cuda_device_context", lambda device: nullcontext()
    ):
        partials_a = ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=0, job_world_size=2)
        partials_b = ray_vae_decode_partial_impl(worker, samples, tile_size=64, overlap=8, job_rank=1, job_world_size=2)

    all_partials = [partials_a, partials_b]
    combined = combine_dist_vae_partials(all_partials)
    assert combined.shape[0] == 1
    assert combined.shape[2] == 64 * 8
    assert combined.shape[3] == 64 * 8

    non_overlap_left = combined[0, 0, 0, 0].item()
    non_overlap_right = combined[0, 0, 0, -1].item()
    assert non_overlap_left == 1.0, f"Non-overlap left should be 1.0, got {non_overlap_left}"
    assert non_overlap_right == 1.0, f"Non-overlap right should be 1.0, got {non_overlap_right}"


def test_sparse_reconstruction_validates_holes():
    partial = {
        "output_shape": (1, 3, 64),
        "spatial_dims": 1,
        "pass_index": 0,
        "latent_spatial_shape": (64,),
        "overlap_latent": 1,
        "feather": (8,),
        "tiles": [],
    }

    try:
        combine_dist_vae_partials([partial])
        assert False, "Expected RuntimeError for hole in coverage"
    except RuntimeError as e:
        assert "cover" in str(e).lower()


def test_sparse_reconstruction_validates_shape_consistency():
    partial_a = {
        "output_shape": (1, 3, 64),
        "spatial_dims": 1,
        "pass_index": 0,
        "latent_spatial_shape": (64,),
        "overlap_latent": 1,
        "feather": (8,),
        "tiles": [(0, 0, torch.ones((1, 3, 64)))],
    }
    partial_b = {
        "output_shape": (1, 3, 32),
        "spatial_dims": 1,
        "pass_index": 0,
        "latent_spatial_shape": (32,),
        "overlap_latent": 1,
        "feather": (4,),
        "tiles": [(0, 0, torch.ones((1, 3, 32)))],
    }

    try:
        combine_dist_vae_partials([partial_a, partial_b])
        assert False, "Expected ValueError for inconsistent output shapes"
    except ValueError as e:
        assert "output shapes" in str(e).lower()


def test_sparse_reconstruction_validates_tile_ndim():
    partial = {
        "output_shape": (1, 3, 64),
        "spatial_dims": 1,
        "pass_index": 0,
        "latent_spatial_shape": (64,),
        "overlap_latent": 1,
        "feather": (8,),
        "tiles": [(0, 0, torch.ones((1, 3, 64, 64)))],
    }

    try:
        combine_dist_vae_partials([partial])
        assert False, "Expected ValueError for invalid tile ndim"
    except ValueError as e:
        assert "invalid 1d tile shape" in str(e).lower()


def test_sparse_reconstruction_validates_tile_batch():
    partial = {
        "output_shape": (1, 3, 64),
        "spatial_dims": 1,
        "pass_index": 0,
        "latent_spatial_shape": (64,),
        "overlap_latent": 1,
        "feather": (8,),
        "tiles": [(0, 0, torch.ones((2, 3, 64)))],
    }

    try:
        combine_dist_vae_partials([partial])
        assert False, "Expected ValueError for invalid batch dimension"
    except ValueError as e:
        assert "batch dimension" in str(e).lower()


def test_sparse_reconstruction_validates_tile_channels():
    partial = {
        "output_shape": (1, 3, 64),
        "spatial_dims": 1,
        "pass_index": 0,
        "latent_spatial_shape": (64,),
        "overlap_latent": 1,
        "feather": (8,),
        "tiles": [(0, 0, torch.ones((1, 1, 64)))],
    }

    try:
        combine_dist_vae_partials([partial])
        assert False, "Expected ValueError for invalid channel dimension"
    except ValueError as e:
        assert "channel dimension" in str(e).lower()


def test_sparse_reconstruction_validates_tile_temporal_3d():
    partial = {
        "output_shape": (1, 3, 8, 64, 64),
        "spatial_dims": 3,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 9, 64, 64)))],
    }

    try:
        combine_dist_vae_partials([partial])
        assert False, "Expected ValueError for invalid temporal dimension"
    except ValueError as e:
        assert "temporal dimension" in str(e).lower()


def test_sparse_reconstruction_validates_tile_spatial_3d():
    partial = {
        "output_shape": (1, 3, 8, 64, 64),
        "spatial_dims": 3,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 8, 64, 64)))],
    }

    combined = combine_dist_vae_partials([partial])
    assert combined.shape == (1, 3, 8, 64, 64)


def test_zero_work_worker_returns_pass_metadata():
    import comfy.model_management as model_management

    vae = types.SimpleNamespace(
        first_stage_model=types.SimpleNamespace(
            decode=lambda tile: torch.ones((1, 3, tile.shape[2] * 8)),
        ),
        handles_tiling=False,
        spacial_compression_decode=lambda: 8,
        upscale_ratio=8,
        upscale_index_formula=None,
        extra_1d_channel=32,
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        output_channels=3,
        memory_used_decode=lambda shape, dtype: 1,
        patcher=object(),
        disable_offload=True,
        device=torch.device("cpu"),
    )
    worker = types.SimpleNamespace(vae_model=vae)
    samples = {"samples": torch.zeros(1, 32, 64)}

    with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
        model_management, "cuda_device_context", lambda device: nullcontext()
    ):
        partials = ray_vae_decode_partial_impl(worker, samples, tile_size=512, overlap=8, job_rank=1, job_world_size=2)

    assert len(partials) == 1
    assert len(partials[0]["tiles"]) == 0, "Zero-work worker must return empty tiles"
    assert partials[0]["output_shape"][0] == 1, "Zero-work worker must still return deterministic output shape"


def test_combine_dist_vae_partials_rejects_empty():
    try:
        combine_dist_vae_partials([])
        assert False, "Expected ValueError for empty worker results"
    except ValueError as e:
        assert "no worker results" in str(e).lower()


def test_combine_dist_vae_partials_rejects_empty_worker_result():
    partial = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }

    try:
        combine_dist_vae_partials([None])
        assert False, "Expected ValueError for None worker result"
    except ValueError as e:
        assert "none" in str(e).lower()


def test_combine_dist_vae_partials_rejects_empty_worker_list():
    partial = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }

    try:
        combine_dist_vae_partials([[]])
        assert False, "Expected ValueError for empty worker result list"
    except ValueError as e:
        assert "empty result list" in str(e).lower()


def test_combine_dist_vae_partials_rejects_missing_2d_pass():
    partial_pass0 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_pass1 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 1,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }

    try:
        combine_dist_vae_partials([partial_pass0, partial_pass1])
        assert False, "Expected ValueError for missing pass 2"
    except ValueError as e:
        assert "missing" in str(e).lower()


def test_combine_dist_vae_partials_rejects_unexpected_pass():
    partial_pass0 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_pass1 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 1,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_pass2 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 2,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_pass3 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 3,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }

    try:
        combine_dist_vae_partials([[partial_pass0, partial_pass1, partial_pass2, partial_pass3]])
        assert False, "Expected ValueError for unexpected pass 3"
    except ValueError as e:
        assert "unexpected" in str(e).lower()


def test_combine_dist_vae_partials_rejects_duplicate_pass():
    partial_pass0_a = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_pass0_b = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_pass1 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 1,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_pass2 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 2,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }

    try:
        combine_dist_vae_partials([[partial_pass0_a, partial_pass0_b, partial_pass1, partial_pass2]])
        assert False, "Expected ValueError for duplicate pass 0"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_combine_dist_vae_partials_rejects_inconsistent_worker_spatial_dims():
    partial_2d_pass0 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_2d_pass1 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 1,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_2d_pass2 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 2,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_3d_pass0 = {
        "output_shape": (1, 3, 8, 64, 64),
        "spatial_dims": 3,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 8, 64, 64)))],
    }
    partial_3d_pass1 = {
        "output_shape": (1, 3, 8, 64, 64),
        "spatial_dims": 3,
        "pass_index": 1,
        "latent_spatial_shape": (8, 8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 8, 64, 64)))],
    }
    partial_3d_pass2 = {
        "output_shape": (1, 3, 8, 64, 64),
        "spatial_dims": 3,
        "pass_index": 2,
        "latent_spatial_shape": (8, 8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 8, 64, 64)))],
    }

    try:
        combine_dist_vae_partials(
            [[partial_2d_pass0, partial_2d_pass1, partial_2d_pass2],
             [partial_3d_pass0, partial_3d_pass1, partial_3d_pass2]]
        )
        assert False, "Expected ValueError for inconsistent spatial_dims"
    except ValueError as e:
        assert "spatial dimensions" in str(e).lower()


def test_combine_dist_vae_partials_rejects_inconsistent_overlap():
    partial_a_pass0 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_a_pass1 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 1,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_a_pass2 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 2,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_b_pass0 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 2,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_b_pass1 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 1,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 2,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }
    partial_b_pass2 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 2,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 2,
        "feather": (8, 8),
        "tiles": [(0, 0, 0, 0, torch.ones((1, 3, 64, 64)))],
    }

    try:
        combine_dist_vae_partials([[partial_a_pass0, partial_a_pass1, partial_a_pass2],
                                   [partial_b_pass0, partial_b_pass1, partial_b_pass2]])
        assert False, "Expected ValueError for inconsistent overlap_latent"
    except ValueError as e:
        assert "overlap" in str(e).lower()


def test_out_of_bounds_tile_rejected():
    partial = {
        "output_shape": (1, 3, 64),
        "spatial_dims": 1,
        "pass_index": 0,
        "latent_spatial_shape": (64,),
        "overlap_latent": 1,
        "feather": (8,),
        "tiles": [(0, 60, torch.ones((1, 3, 64)))],
    }

    try:
        combine_dist_vae_partials([partial])
        assert False, "Expected ValueError for out-of-bounds tile"
    except ValueError as e:
        assert "outside the output bounds" in str(e).lower()


def test_feather_gte_tile_extent_yields_all_ones():
    from directordeck_raylight.distributed_worker.ray_worker_vae import _linear_feather

    tile_size = 32
    feather = 32  # feather >= tile extent
    effective_feather = feather if 0 < feather < tile_size else 0
    weight = _linear_feather(tile_size, effective_feather)
    assert torch.all(weight == 1.0), "Feather >= tile extent should yield all-ones weight"


def test_index_formula_numeric_tuple():
    index_formula = (4, 8, 8)
    pos_0 = _get_pos_func(index_formula, 8, 0)
    pos_1 = _get_pos_func(index_formula, 8, 1)

    assert pos_0(4) == 16, "Numeric index formula should multiply position by formula value"
    assert pos_1(7) == 56, "Numeric index formula should multiply position by formula value"


def test_index_formula_scalar_numeric():
    index_formula = 3
    pos_0 = _get_pos_func(index_formula, 8, 0)
    pos_1 = _get_pos_func(index_formula, 8, 1)

    assert pos_0(4) == 12, "Scalar numeric index formula should multiply position by formula value"
    assert pos_1(5) == 15, "Scalar numeric index formula should be reused across dimensions"


def test_index_formula_callable_vs_numeric():
    index_formula = (lambda a: a * 3, 8)
    pos_0 = _get_pos_func(index_formula, 8, 0)
    pos_1 = _get_pos_func(index_formula, 8, 1)

    assert pos_0(4) == 12, "Callable index formula should be called"
    assert pos_1(7) == 56, "Numeric index formula should multiply position by formula value"


def test_index_formula_none_falls_back_to_upscale():
    index_formula = (None, 8)
    pos_0 = _get_pos_func(index_formula, 8, 0)
    pos_1 = _get_pos_func(index_formula, 8, 1)

    assert pos_0(4) == 32, "None index formula should fall back to upscale"
    assert pos_1(7) == 56, "Numeric index formula should multiply position by formula value"


def test_anisotropic_2d_feather_metadata():
    upscale = (lambda a: max(0, a * 4 - 3), 8)
    overlap_latent = 2
    feather = _compute_feather(upscale, overlap_latent, 2)
    assert len(feather) == 2
    assert feather[0] == round(max(0, overlap_latent * 4 - 3))
    assert feather[1] == round(overlap_latent * 8)


def test_1d_feather_metadata():
    upscale = 8
    overlap_latent = 2
    feather = _compute_feather(upscale, overlap_latent, 1)
    assert feather == (round(overlap_latent * 8),)


def test_3d_spatial_feather_metadata():
    upscale = (lambda a: max(0, a * 4 - 3), 8, 8)
    overlap_latent = 2
    feather = _compute_feather(upscale, overlap_latent, 3)
    assert feather == (round(overlap_latent * 8), round(overlap_latent * 8))


def test_normalize_worker_result_dict():
    result = {"pass_index": 0}
    normalized = _normalize_worker_result(result)
    assert normalized == [result]


def test_normalize_worker_result_list():
    result = [{"pass_index": 0}, {"pass_index": 1}]
    normalized = _normalize_worker_result(result)
    assert normalized == result


def test_normalize_worker_result_none():
    try:
        _normalize_worker_result(None)
        assert False, "Expected ValueError for None worker result"
    except ValueError as e:
        assert "none" in str(e).lower()


def test_normalize_worker_result_empty_list():
    try:
        _normalize_worker_result([])
        assert False, "Expected ValueError for empty worker result list"
    except ValueError as e:
        assert "empty result list" in str(e).lower()


def test_validate_worker_passes_missing():
    partial_pass0 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [],
    }

    try:
        _validate_worker_passes([partial_pass0], {0, 1, 2}, 2)
        assert False, "Expected ValueError for missing pass"
    except ValueError as e:
        assert "missing" in str(e).lower()


def test_validate_worker_passes_unexpected():
    partial_pass3 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 3,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [],
    }

    try:
        _validate_worker_passes([partial_pass3], {0, 1, 2}, 2)
        assert False, "Expected ValueError for unexpected pass"
    except ValueError as e:
        assert "unexpected" in str(e).lower()


def test_validate_worker_passes_duplicate():
    partial_pass0 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [],
    }
    partial_pass1 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 1,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [],
    }
    partial_pass2 = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 2,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [],
    }

    try:
        _validate_worker_passes([[partial_pass0, partial_pass0, partial_pass1, partial_pass2]], {0, 1, 2}, 2)
        assert False, "Expected ValueError for duplicate pass"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_validate_worker_passes_inconsistent_spatial_dims():
    partial_2d = {
        "output_shape": (1, 3, 64, 64),
        "spatial_dims": 2,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [],
    }
    partial_3d = {
        "output_shape": (1, 3, 8, 64, 64),
        "spatial_dims": 3,
        "pass_index": 0,
        "latent_spatial_shape": (8, 8, 8),
        "overlap_latent": 1,
        "feather": (8, 8),
        "tiles": [],
    }

    try:
        _validate_worker_passes([partial_2d, partial_3d], {0}, 2)
        assert False, "Expected ValueError for inconsistent spatial_dims"
    except ValueError as e:
        assert "spatial dimensions" in str(e).lower()
