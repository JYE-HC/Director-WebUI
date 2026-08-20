from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from directordeck.media import (
    MediaToolError,
    _scene_frames,
    assemble_video_bytes,
    create_24fps_proxy_file,
    create_24fps_proxy_bytes,
    detect_shots_path,
    probe_video_bytes,
)
from directordeck.schemas import VideoMetadata


def _sample_video(tmp_path: Path, *, fps: int = 30, audio: bool = True) -> bytes:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    path = tmp_path / f"sample-{fps}-{'audio' if audio else 'silent'}.mp4"
    command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x48:rate={fps}:duration=1",
    ]
    if audio:
        command.extend([
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-shortest",
        ])
    command.extend([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
    ])
    if audio:
        command.extend(["-c:a", "aac"])
    command.append(str(path))
    subprocess.run(
        command,
        check=True,
    )
    return path.read_bytes()


def test_probe_and_proxy_video_bytes(tmp_path: Path) -> None:
    source = _sample_video(tmp_path, fps=30)
    original = probe_video_bytes(source)
    assert original.width == 64
    assert original.height == 48
    assert original.native_fps == pytest.approx(30, rel=0.01)
    assert original.duration == pytest.approx(1, abs=0.05)
    assert original.has_audio is True

    proxy = create_24fps_proxy_bytes(source)
    assert proxy.filename_suffix == ".mp4"
    assert proxy.content
    assert proxy.metadata.native_fps == pytest.approx(24, rel=0.001)
    assert proxy.metadata.frame_count == pytest.approx(24, abs=1)
    assert proxy.metadata.probe_method == "backend_ffmpeg_proxy_24fps"
    assert proxy.metadata.has_audio is True

    silent = probe_video_bytes(_sample_video(tmp_path, fps=24, audio=False))
    assert silent.has_audio is False


def test_compliant_24fps_proxy_uses_remux_fast_path(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(_sample_video(tmp_path, fps=24))
    destination = tmp_path / "proxy.mp4"

    result = create_24fps_proxy_file(source, destination)

    assert result.strategy == "remux"
    assert destination.is_file()
    assert result.metadata.native_fps == pytest.approx(24, rel=0.001)
    assert result.metadata.probe_method == "backend_ffmpeg_proxy_24fps"


def test_noncompliant_proxy_uses_veryfast_transcode(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(_sample_video(tmp_path, fps=30))
    destination = tmp_path / "proxy.mp4"
    import directordeck.media as media_module
    commands: list[list[str]] = []
    original = media_module._run_command

    def record(args: list[str], *, timeout: float):
        commands.append(args)
        return original(args, timeout=timeout)

    monkeypatch.setattr(media_module, "_run_command", record)
    result = create_24fps_proxy_file(source, destination)

    assert result.strategy == "transcode"
    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert ffmpeg[ffmpeg.index("-preset") + 1] == "veryfast"


def test_historical_video_metadata_defaults_to_silent() -> None:
    historical = VideoMetadata.model_validate({
        "duration": 1,
        "native_fps": 24,
        "frame_count": 24,
        "width": 64,
        "height": 48,
        "probe_method": "legacy",
    })
    assert historical.has_audio is False


def test_assembly_normalizes_and_concatenates_segments(tmp_path: Path) -> None:
    source = _sample_video(tmp_path, fps=24)
    silent = _sample_video(tmp_path, fps=24, audio=False)
    assembled = assemble_video_bytes(
        [source, silent], fps=24, width=64, height=48
    )

    assert assembled.content
    assert assembled.filename_suffix == ".mp4"
    assert assembled.metadata.width == 64
    assert assembled.metadata.height == 48
    assert assembled.metadata.native_fps == pytest.approx(24, rel=0.001)
    assert assembled.metadata.duration == pytest.approx(2, abs=0.12)
    assert assembled.metadata.probe_method == "backend_ffmpeg_timeline_assembly"


def test_invalid_video_is_rejected() -> None:
    with pytest.raises(MediaToolError):
        probe_video_bytes(b"not a video")


def test_scene_timestamp_parser_uses_timeline_frame_rate() -> None:
    stderr = "n:1 pts_time:0.5 more\nn:2 pts_time:1.25 more\nn:3 pts_time:1.25"
    assert _scene_frames(stderr, frame_rate=24) == [12, 30]


def test_shot_detection_always_includes_bounds(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(_sample_video(tmp_path, fps=24))
    result = detect_shots_path(
        source,
        frame_rate=24,
        total_frames=24,
        sensitivity="medium",
        min_shot_frames=4,
    )
    assert result.cut_frames[0] == 0
    assert result.cut_frames[-1] == 24
    assert result.shot_count == len(result.cut_frames) - 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_rate": 0, "total_frames": 24, "sensitivity": "medium", "min_shot_frames": 4},
        {"frame_rate": 24, "total_frames": 0, "sensitivity": "medium", "min_shot_frames": 4},
        {"frame_rate": 24, "total_frames": 24, "sensitivity": "invalid", "min_shot_frames": 4},
    ],
)
def test_shot_detection_rejects_invalid_contract(tmp_path: Path, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        detect_shots_path(tmp_path / "unused.mp4", **kwargs)  # type: ignore[arg-type]
