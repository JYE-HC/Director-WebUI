from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from .schemas import DetectShotsResponse, VideoMetadata


class MediaToolError(RuntimeError):
    """Raised when a bounded ffmpeg/ffprobe operation cannot be completed."""


class MediaToolTimeout(MediaToolError):
    """Raised when a named media stage exceeds its bounded subprocess budget."""

    def __init__(self, operation: str, timeout: float) -> None:
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"{operation} exceeded its {timeout:g}s timeout")


@dataclass(frozen=True, slots=True)
class VideoProxy:
    content: bytes
    filename_suffix: str
    metadata: VideoMetadata


@dataclass(frozen=True, slots=True)
class VideoProxyResult:
    """Metadata for a file-backed proxy and the work used to create it."""

    metadata: VideoMetadata
    strategy: str


_SCENE_THRESHOLDS = {"low": 0.55, "medium": 0.35, "high": 0.18}
_PTS_TIME = re.compile(r"\bpts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
_SAFE_SUFFIX = re.compile(r"^\.[a-zA-Z0-9]{1,10}$")
_LOGGER = logging.getLogger(__name__)
_METADATA_PROBE_TIMEOUT_SECONDS = 60
_REMUX_ELIGIBILITY_SCAN_TIMEOUT_SECONDS = 60
_FRAME_COUNT_SCAN_TIMEOUT_SECONDS = 60
_MEDIA_PROCESS_TIMEOUT_SECONDS = 1800


def _suffix(value: str) -> str:
    candidate = value if value.startswith(".") else f".{value}"
    return candidate.lower() if _SAFE_SUFFIX.fullmatch(candidate) else ".bin"


def _run_command(
    args: list[str],
    *,
    timeout: float,
    operation: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    label = operation or args[0]
    try:
        result = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaToolError(f"required media tool is unavailable: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaToolTimeout(label, timeout) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-4000:].strip()
        raise MediaToolError(f"{label} failed with exit code {result.returncode}: {detail}")
    return result


def _ffmpeg_full_help() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "full"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.decode("utf-8", errors="replace")


@lru_cache(maxsize=1)
def _fps_sync_args() -> tuple[str, str]:
    """Return the CFR frame-sync option spelling this ffmpeg build accepts.

    ``-fps_mode`` only exists since ffmpeg 5.1 while ``-vsync`` is deprecated
    and marked for future removal, so no single spelling runs everywhere.
    Probe the binary once and cache the answer; an unprobed or unreadable
    binary keeps the historical ``-vsync cfr``.
    """
    if "-fps_mode" in _ffmpeg_full_help():
        return ("-fps_mode", "cfr")
    return ("-vsync", "cfr")


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _frame_rate(value: object) -> float | None:
    try:
        parsed = float(Fraction(str(value)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def probe_video_path(
    path: str | Path,
    *,
    probe_method: str = "backend_ffprobe",
    allow_frame_count_estimate_on_timeout: bool = False,
) -> VideoMetadata:
    source = Path(path)
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(source),
        ],
        timeout=_METADATA_PROBE_TIMEOUT_SECONDS,
        operation="video metadata probe",
    )
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        stream = next(
            item for item in streams
            if isinstance(item, dict) and item.get("codec_type") == "video"
        )
    except (json.JSONDecodeError, KeyError, IndexError, StopIteration, TypeError) as exc:
        raise MediaToolError("ffprobe returned no readable video stream") from exc
    if not isinstance(stream, dict):
        raise MediaToolError("ffprobe returned an invalid video stream")

    width = _positive_int(stream.get("width"))
    height = _positive_int(stream.get("height"))
    fps = _frame_rate(stream.get("avg_frame_rate")) or _frame_rate(
        stream.get("r_frame_rate")
    )
    format_value = payload.get("format") if isinstance(payload, dict) else None
    format_duration = format_value.get("duration") if isinstance(format_value, dict) else None
    duration = _positive_float(stream.get("duration")) or _positive_float(format_duration)
    frame_count = _positive_int(stream.get("nb_frames"))
    frame_count_estimated = False
    if frame_count is None:
        try:
            counted = _run_command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "json",
                    str(source),
                ],
                timeout=_FRAME_COUNT_SCAN_TIMEOUT_SECONDS,
                operation="video frame-count scan",
            )
        except MediaToolTimeout as exc:
            if (
                not allow_frame_count_estimate_on_timeout
                or duration is None
                or fps is None
            ):
                raise
            # Frame counting scans the complete stream. If the container already
            # supplies duration and frame rate, keep the scan bounded and use
            # the same estimate accepted when ffprobe returns no readable count.
            _LOGGER.warning(
                "%s; estimating frame count from duration and frame rate",
                exc,
            )
        else:
            try:
                counted_stream = json.loads(counted.stdout)["streams"][0]
                frame_count = _positive_int(counted_stream.get("nb_read_frames"))
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
                frame_count = None
    if duration is None and frame_count is not None and fps is not None:
        duration = frame_count / fps
    if frame_count is None and duration is not None and fps is not None:
        frame_count = max(1, int(round(duration * fps)))
        frame_count_estimated = True
    if None in (width, height, fps, duration, frame_count):
        raise MediaToolError("ffprobe returned incomplete video metadata")
    resolved_probe_method = probe_method
    if frame_count_estimated:
        candidate = f"{probe_method}_estimated_frames"
        resolved_probe_method = candidate if len(candidate) <= 128 else "estimated_frames"
    return VideoMetadata(
        duration=duration,
        native_fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        probe_method=resolved_probe_method,
        has_audio=any(
            isinstance(item, dict) and item.get("codec_type") == "audio"
            for item in streams
        ),
    )


def probe_video_bytes(content: bytes, suffix: str = ".mp4") -> VideoMetadata:
    with tempfile.TemporaryDirectory(prefix="director-media-") as directory:
        path = Path(directory) / f"source{_suffix(suffix)}"
        path.write_bytes(content)
        return probe_video_path(path)


def create_24fps_proxy_path(source: str | Path, destination: str | Path) -> VideoMetadata:
    """Create the immutable CFR proxy required by MiniMax H3 ref-video inputs."""

    return create_24fps_proxy_file(source, destination).metadata


def _can_remux_24fps_proxy(source: str | Path) -> bool:
    """Return true only for streams already satisfying the immutable proxy contract."""

    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(source),
        ],
        timeout=_METADATA_PROBE_TIMEOUT_SECONDS,
        operation="proxy compatibility probe",
    )
    try:
        streams = json.loads(result.stdout)["streams"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MediaToolError("ffprobe returned no readable streams") from exc
    if not isinstance(streams, list):
        return False
    videos = [
        item
        for item in streams
        if isinstance(item, dict) and item.get("codec_type") == "video"
    ]
    audios = [
        item
        for item in streams
        if isinstance(item, dict) and item.get("codec_type") == "audio"
    ]
    if len(videos) != 1 or len(audios) > 1:
        return False
    video = videos[0]
    width = _positive_int(video.get("width"))
    height = _positive_int(video.get("height"))
    avg_fps = _frame_rate(video.get("avg_frame_rate"))
    real_fps = _frame_rate(video.get("r_frame_rate"))
    stream_compatible = bool(
        video.get("codec_name") == "h264"
        and video.get("pix_fmt") == "yuv420p"
        and width is not None
        and height is not None
        and width % 2 == 0
        and height % 2 == 0
        and avg_fps is not None
        and real_fps is not None
        and math.isclose(avg_fps, 24.0, abs_tol=1e-6)
        and math.isclose(real_fps, 24.0, abs_tol=1e-6)
        and (not audios or audios[0].get("codec_name") == "aac")
    )
    if not stream_compatible:
        return False
    # Container rate fields can claim 24 fps for VFR media. Packet durations
    # are cheap to scan compared with decoding/re-encoding and close that gap.
    try:
        packet_result = _run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=duration_time",
                "-of",
                "csv=p=0",
                str(source),
            ],
            timeout=_REMUX_ELIGIBILITY_SCAN_TIMEOUT_SECONDS,
            operation="proxy packet-duration scan",
        )
    except MediaToolTimeout as exc:
        # This scan only proves eligibility for the remux optimization. A
        # timeout must conservatively choose canonical transcoding, not reject
        # an otherwise valid upload.
        _LOGGER.warning("%s; falling back to canonical transcode", exc)
        return False
    durations = [
        _positive_float(line.strip())
        for line in packet_result.stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return bool(
        durations
        and all(
            duration is not None and math.isclose(duration, 1 / 24, abs_tol=1e-6)
            for duration in durations
        )
    )


def create_24fps_proxy_file(source: str | Path, destination: str | Path) -> VideoProxyResult:
    """Create a file-backed proxy, remuxing compliant media without re-encoding."""

    if _can_remux_24fps_proxy(source):
        _run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                str(destination),
            ],
            timeout=_MEDIA_PROCESS_TIMEOUT_SECONDS,
            operation="video proxy remux",
        )
        return VideoProxyResult(
            metadata=probe_video_path(
                destination,
                probe_method="backend_ffmpeg_proxy_24fps",
                allow_frame_count_estimate_on_timeout=True,
            ),
            strategy="remux",
        )

    _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "fps=24,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            # A single video output stream keeps both frame-sync spellings
            # unambiguous here.
            *_fps_sync_args(),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            str(destination),
        ],
        timeout=_MEDIA_PROCESS_TIMEOUT_SECONDS,
        operation="video proxy transcode",
    )
    return VideoProxyResult(
        metadata=probe_video_path(
            destination,
            probe_method="backend_ffmpeg_proxy_24fps",
            allow_frame_count_estimate_on_timeout=True,
        ),
        strategy="transcode",
    )


def create_24fps_proxy_bytes(content: bytes, suffix: str = ".mp4") -> VideoProxy:
    with tempfile.TemporaryDirectory(prefix="director-media-") as directory:
        source = Path(directory) / f"source{_suffix(suffix)}"
        destination = Path(directory) / "proxy.mp4"
        source.write_bytes(content)
        metadata = create_24fps_proxy_path(source, destination)
        return VideoProxy(
            content=destination.read_bytes(),
            filename_suffix=".mp4",
            metadata=metadata,
        )


def _has_audio_stream(path: str | Path) -> bool:
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        timeout=60,
    )
    return bool(result.stdout.strip())


def assemble_video_paths(
    sources: list[str | Path],
    destination: str | Path,
    *,
    fps: float,
    width: int,
    height: int,
) -> VideoMetadata:
    """Normalize and concatenate segment files into one deterministic MP4.

    Every intermediate receives H.264 video and a stereo 48 kHz AAC track.
    Supplying silence for a segment without audio keeps ffmpeg's concat
    timeline aligned instead of dropping or shifting later soundtracks.
    """

    if not sources:
        raise ValueError("at least one segment video is required")
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        raise ValueError("fps must be finite and between 0 and 240")
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("assembly dimensions must be positive even integers")

    destination_path = Path(destination)
    with tempfile.TemporaryDirectory(prefix="director-assembly-") as directory:
        workspace = Path(directory)
        normalized: list[Path] = []
        normalized_durations: list[float] = []
        for index, source in enumerate(sources):
            source_path = Path(source)
            normalized_path = workspace / f"segment-{index:04d}.mp4"
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source_path),
            ]
            has_audio = _has_audio_stream(source_path)
            if not has_audio:
                command.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=channel_layout=stereo:sample_rate=48000",
                    ]
                )
            command.extend(
                [
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0" if has_audio else "1:a:0",
                    "-vf",
                    f"fps={fps:g},scale={width}:{height}:flags=lanczos,setsar=1",
                    "-af",
                    "aresample=48000:async=1:first_pts=0,apad",
                    *_fps_sync_args(),
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-b:a",
                    "192k",
                    "-map_metadata",
                    "-1",
                    "-map_chapters",
                    "-1",
                    str(normalized_path),
                ]
            )
            _run_command(command, timeout=1800)
            normalized.append(normalized_path)
            normalized_durations.append(
                probe_video_path(
                    normalized_path,
                    allow_frame_count_estimate_on_timeout=True,
                ).duration
            )

        concat_file = workspace / "segments.txt"
        concat_file.write_text(
            "".join(
                f"file '{path.as_posix()}'\nduration {duration:.9f}\n"
                for path, duration in zip(normalized, normalized_durations, strict=True)
            ),
            encoding="utf-8",
        )
        _run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(destination_path),
            ],
            timeout=1800,
        )
    return probe_video_path(
        destination_path,
        probe_method="backend_ffmpeg_timeline_assembly",
        allow_frame_count_estimate_on_timeout=True,
    )


def assemble_video_bytes(
    segments: list[bytes],
    *,
    fps: float,
    width: int,
    height: int,
) -> VideoProxy:
    with tempfile.TemporaryDirectory(prefix="director-assembly-input-") as directory:
        workspace = Path(directory)
        sources: list[Path] = []
        for index, content in enumerate(segments):
            if not content:
                raise ValueError(f"segment {index} is empty")
            source = workspace / f"input-{index:04d}.mp4"
            source.write_bytes(content)
            sources.append(source)
        destination = workspace / "timeline.mp4"
        metadata = assemble_video_paths(
            sources,
            destination,
            fps=fps,
            width=width,
            height=height,
        )
        return VideoProxy(
            content=destination.read_bytes(),
            filename_suffix=".mp4",
            metadata=metadata,
        )


def _scene_frames(stderr: str, *, frame_rate: float) -> list[int]:
    frames: list[int] = []
    for match in _PTS_TIME.finditer(stderr):
        seconds = float(match.group(1))
        if math.isfinite(seconds) and seconds >= 0:
            frames.append(max(0, int(round(seconds * frame_rate))))
    return sorted(set(frames))


def detect_shots_path(
    path: str | Path,
    *,
    frame_rate: float,
    total_frames: int,
    sensitivity: str,
    min_shot_frames: int,
) -> DetectShotsResponse:
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("frame_rate must be positive and finite")
    if total_frames < 1:
        raise ValueError("total_frames must be positive")
    if min_shot_frames < 1:
        raise ValueError("min_shot_frames must be positive")
    try:
        threshold = _SCENE_THRESHOLDS[sensitivity]
    except KeyError as exc:
        raise ValueError("sensitivity must be low, medium or high") from exc
    result = _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-vf",
            f"select=gt(scene\\,{threshold}),showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        timeout=1800,
    )
    candidates = [
        frame
        for frame in _scene_frames(result.stderr.decode("utf-8", errors="replace"), frame_rate=frame_rate)
        if 0 < frame < total_frames
    ]
    cuts = [0]
    removed = 0
    for frame in candidates:
        if frame - cuts[-1] < min_shot_frames or total_frames - frame < min_shot_frames:
            removed += 1
            continue
        cuts.append(frame)
    cuts.append(total_frames)
    warnings = (
        [f"{removed} 个镜头切点因小于最小镜头长度而被忽略"] if removed else []
    )
    return DetectShotsResponse(
        cut_frames=cuts,
        shot_count=max(0, len(cuts) - 1),
        warnings=warnings,
    )


def detect_shots_bytes(
    content: bytes,
    *,
    suffix: str = ".mp4",
    frame_rate: float,
    total_frames: int,
    sensitivity: str,
    min_shot_frames: int,
) -> DetectShotsResponse:
    with tempfile.TemporaryDirectory(prefix="director-media-") as directory:
        source = Path(directory) / f"source{_suffix(suffix)}"
        source.write_bytes(content)
        return detect_shots_path(
            source,
            frame_rate=frame_rate,
            total_frames=total_frames,
            sensitivity=sensitivity,
            min_shot_frames=min_shot_frames,
        )
