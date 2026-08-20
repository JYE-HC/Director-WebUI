from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from directordeck.media import (
    MediaToolError,
    MediaToolTimeout,
    _scene_frames,
    assemble_video_bytes,
    create_24fps_proxy_file,
    create_24fps_proxy_bytes,
    detect_shots_path,
    probe_video_path,
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

    def record(
        args: list[str],
        *,
        timeout: float,
        operation: str | None = None,
    ):
        commands.append(args)
        return original(args, timeout=timeout, operation=operation)

    monkeypatch.setattr(media_module, "_run_command", record)
    result = create_24fps_proxy_file(source, destination)

    assert result.strategy == "transcode"
    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert ffmpeg[ffmpeg.index("-preset") + 1] == "veryfast"


def test_fps_sync_args_match_detected_ffmpeg_capability(monkeypatch) -> None:
    import directordeck.media as media_module

    try:
        monkeypatch.setattr(
            media_module,
            "_ffmpeg_full_help",
            lambda: "Per-stream options:\n-fps_mode ...\n-vsync ...\n",
        )
        media_module._fps_sync_args.cache_clear()
        assert media_module._fps_sync_args() == ("-fps_mode", "cfr")

        monkeypatch.setattr(
            media_module,
            "_ffmpeg_full_help",
            lambda: "AVOptions:\n-vsync ...\n",
        )
        media_module._fps_sync_args.cache_clear()
        assert media_module._fps_sync_args() == ("-vsync", "cfr")

        monkeypatch.setattr(media_module, "_ffmpeg_full_help", lambda: "")
        media_module._fps_sync_args.cache_clear()
        assert media_module._fps_sync_args() == ("-vsync", "cfr")
    finally:
        media_module._fps_sync_args.cache_clear()


def test_real_matroska_count_scan_is_separate_from_proxy_normalization(
    tmp_path: Path, monkeypatch
) -> None:
    source_mp4 = tmp_path / "source.mp4"
    source_mp4.write_bytes(_sample_video(tmp_path, fps=24, audio=False))
    source_mkv = tmp_path / "source-without-nb-frames.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source_mp4),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            str(source_mkv),
        ],
        check=True,
        timeout=15,
    )
    destination = tmp_path / "proxy.mp4"
    import directordeck.media as media_module
    commands: list[list[str]] = []
    original = media_module._run_command

    def record(
        args: list[str],
        *,
        timeout: float,
        operation: str | None = None,
    ):
        commands.append(args)
        return original(args, timeout=timeout, operation=operation)

    monkeypatch.setattr(media_module, "_run_command", record)
    source_metadata = probe_video_path(source_mkv)
    source_commands = list(commands)
    commands.clear()
    proxy = create_24fps_proxy_file(source_mkv, destination)

    assert source_metadata.frame_count == pytest.approx(24, abs=1)
    assert any("-count_frames" in command for command in source_commands), (
        "the real Matroska fixture must omit nb_frames and exercise the generic count scan"
    )
    assert not any(
        "-count_frames" in command and str(source_mkv) in command
        for command in commands
    )
    assert destination.is_file()
    assert proxy.metadata.frame_count == pytest.approx(24, abs=1)
    assert proxy.metadata.probe_method == "backend_ffmpeg_proxy_24fps"


def test_named_media_timeout_preserves_stage_and_budget(monkeypatch) -> None:
    import directordeck.media as media_module

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=7)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(MediaToolTimeout) as raised:
        media_module._run_command(
            ["ffprobe", "input.mp4"],
            timeout=7,
            operation="test frame scan",
        )

    assert raised.value.operation == "test frame scan"
    assert raised.value.timeout == 7
    assert str(raised.value) == "test frame scan exceeded its 7s timeout"


def test_frame_count_timeout_estimates_when_duration_and_fps_are_known(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import directordeck.media as media_module

    metadata_payload = json.dumps({
        "streams": [{
            "codec_type": "video",
            "width": 64,
            "height": 48,
            "avg_frame_rate": "24/1",
            "duration": "120.0",
        }],
        "format": {"duration": "120.0"},
    })
    calls: list[tuple[list[str], float, str | None]] = []

    def fake_run(
        args: list[str],
        *,
        timeout: float,
        operation: str | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, timeout, operation))
        if operation == "video metadata probe":
            return subprocess.CompletedProcess(args, 0, metadata_payload.encode(), b"")
        assert operation == "video frame-count scan"
        assert "-count_frames" in args
        raise MediaToolTimeout(operation or args[0], timeout)

    monkeypatch.setattr(media_module, "_run_command", fake_run)
    metadata = probe_video_path(
        tmp_path / "long-no-nb-frames.mp4",
        probe_method="canonical_cfr_proxy",
        allow_frame_count_estimate_on_timeout=True,
    )

    assert metadata.frame_count == 2880
    assert metadata.probe_method == "canonical_cfr_proxy_estimated_frames"
    assert len(calls) == 2
    assert calls[1][1:] == (60, "video frame-count scan")
    assert "estimating frame count from duration and frame rate" in caplog.text


def test_generic_probe_does_not_estimate_a_frame_count_after_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    import directordeck.media as media_module

    metadata_payload = json.dumps({
        "streams": [{
            "codec_type": "video",
            "width": 64,
            "height": 48,
            "avg_frame_rate": "24/1",
            "duration": "120.0",
        }],
        "format": {"duration": "120.0"},
    })

    def fake_run(
        args: list[str],
        *,
        timeout: float,
        operation: str | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        if operation == "video metadata probe":
            return subprocess.CompletedProcess(args, 0, metadata_payload.encode(), b"")
        assert operation == "video frame-count scan"
        assert "-count_frames" in args
        raise MediaToolTimeout(operation, timeout)

    monkeypatch.setattr(media_module, "_run_command", fake_run)
    with pytest.raises(MediaToolTimeout, match="video frame-count scan"):
        probe_video_path(tmp_path / "generic-vfr-input.mp4")


def test_frame_count_non_timeout_error_is_not_estimated(
    tmp_path: Path, monkeypatch
) -> None:
    import directordeck.media as media_module

    metadata_payload = json.dumps({
        "streams": [{
            "codec_type": "video",
            "width": 64,
            "height": 48,
            "avg_frame_rate": "24/1",
            "duration": "120.0",
        }],
        "format": {"duration": "120.0"},
    })

    def fake_run(
        args: list[str],
        *,
        timeout: float,
        operation: str | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        if operation == "video metadata probe":
            return subprocess.CompletedProcess(args, 0, metadata_payload.encode(), b"")
        assert operation == "video frame-count scan"
        raise MediaToolError("frame count command failed")

    monkeypatch.setattr(media_module, "_run_command", fake_run)
    with pytest.raises(MediaToolError, match="frame count command failed"):
        probe_video_path(
            tmp_path / "invalid-count-result.mp4",
            allow_frame_count_estimate_on_timeout=True,
        )


def test_frame_count_timeout_without_estimation_inputs_remains_an_error(
    tmp_path: Path, monkeypatch
) -> None:
    import directordeck.media as media_module

    metadata_payload = json.dumps({
        "streams": [{
            "codec_type": "video",
            "width": 64,
            "height": 48,
            "avg_frame_rate": "24/1",
        }],
        "format": {},
    })
    calls: list[str | None] = []

    def fake_run(
        args: list[str],
        *,
        timeout: float,
        operation: str | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(operation)
        if operation == "video metadata probe":
            return subprocess.CompletedProcess(args, 0, metadata_payload.encode(), b"")
        assert operation == "video frame-count scan"
        assert "-count_frames" in args
        raise MediaToolTimeout(operation or args[0], timeout)

    monkeypatch.setattr(media_module, "_run_command", fake_run)
    with pytest.raises(MediaToolTimeout, match="video frame-count scan"):
        probe_video_path(
            tmp_path / "unknown-duration.mp4",
            allow_frame_count_estimate_on_timeout=True,
        )
    assert calls == ["video metadata probe", "video frame-count scan"]


def test_packet_scan_timeout_falls_back_from_remux_eligibility(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import directordeck.media as media_module
    source = tmp_path / "packet-timeout-source.mp4"
    source.write_bytes(_sample_video(tmp_path, fps=24))
    destination = tmp_path / "packet-timeout-proxy.mp4"
    calls: list[tuple[list[str], float, str | None]] = []
    original = media_module._run_command

    def fake_run(
        args: list[str],
        *,
        timeout: float,
        operation: str | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, timeout, operation))
        if operation == "proxy packet-duration scan":
            assert "packet=duration_time" in args
            raise MediaToolTimeout(operation, timeout)
        return original(args, timeout=timeout, operation=operation)

    monkeypatch.setattr(media_module, "_run_command", fake_run)
    proxy = create_24fps_proxy_file(source, destination)

    packet_call = next(call for call in calls if call[2] == "proxy packet-duration scan")
    assert packet_call[1] == 60
    assert proxy.strategy == "transcode"
    assert destination.is_file()
    assert proxy.metadata.native_fps == pytest.approx(24, rel=0.001)
    assert proxy.metadata.frame_count == pytest.approx(24, abs=1)
    assert "falling back to canonical transcode" in caplog.text


def test_packet_scan_non_timeout_error_is_not_downgraded(
    tmp_path: Path, monkeypatch
) -> None:
    import directordeck.media as media_module
    source = tmp_path / "packet-error-source.mp4"
    source.write_bytes(_sample_video(tmp_path, fps=24))
    destination = tmp_path / "packet-error-proxy.mp4"
    original = media_module._run_command

    def fake_run(
        args: list[str],
        *,
        timeout: float,
        operation: str | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        if operation == "proxy packet-duration scan":
            raise MediaToolError("packet scan command failed")
        return original(args, timeout=timeout, operation=operation)

    monkeypatch.setattr(media_module, "_run_command", fake_run)
    with pytest.raises(MediaToolError, match="packet scan command failed"):
        create_24fps_proxy_file(source, destination)
    assert not destination.exists()


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
