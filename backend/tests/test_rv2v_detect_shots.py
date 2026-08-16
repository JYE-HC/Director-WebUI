from __future__ import annotations

import pytest

from director.media import MediaToolError
from director.schemas import DetectShotsResponse


@pytest.fixture(autouse=True)
def synchronous_media_worker(monkeypatch):
    # The managed test sandbox blocks asyncio's default executor wakeup. Keep
    # endpoint tests deterministic while media.py itself has real ffmpeg tests.
    async def run_sync(function):
        return function()

    monkeypatch.setattr("director.app.anyio.to_thread.run_sync", run_sync)


async def test_rv2v_shot_detection_uses_registered_asset_and_local_media_tool(
    client, fake_comfy, monkeypatch
) -> None:
    captured: dict = {}

    def detect(content: bytes, **kwargs):
        captured.update(content=content, **kwargs)
        return DetectShotsResponse(
            cut_frames=[0, 120, 288],
            shot_count=2,
            warnings=["low contrast"],
        )

    monkeypatch.setattr("director.app.detect_shots_bytes", detect)
    response = await client.post(
        "/api/rv2v/detect-shots",
        json={
            "asset_id": "fixture-video-source.mp4",
            "frame_rate": 24.0,
            "sensitivity": "medium",
            "min_shot_frames": 12,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "cut_frames": [0, 120, 288],
        "shot_count": 2,
        "warnings": ["low contrast"],
    }
    assert captured["content"] == fake_comfy.view_content
    assert captured["frame_rate"] == 24.0
    assert captured["total_frames"] == 288


async def test_rv2v_shot_detection_surfaces_local_media_error(
    client, monkeypatch
) -> None:
    def fail(*_args, **_kwargs):
        raise MediaToolError("PySceneDetect is unavailable")

    monkeypatch.setattr("director.app.detect_shots_bytes", fail)
    response = await client.post(
        "/api/rv2v/detect-shots",
        json={
            "asset_id": "fixture-video-source.mp4",
            "frame_rate": 24.0,
            "sensitivity": "medium",
            "min_shot_frames": 12,
        },
    )
    assert response.status_code == 422
    assert "PySceneDetect" in response.text


async def test_rv2v_shot_detection_rejects_client_paths(client) -> None:
    response = await client.post(
        "/api/rv2v/detect-shots",
        json={
            "asset_id": "fixture-video-source.mp4",
            "video_file": "/etc/passwd",
            "frame_rate": 24.0,
            "sensitivity": "medium",
            "min_shot_frames": 12,
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("asset_id", "status", "message"),
    [
        ("fixture-image-first.png", 422, "video asset"),
        ("missing", 404, "asset not found"),
    ],
)
async def test_rv2v_shot_detection_requires_registered_video(
    client, asset_id: str, status: int, message: str
) -> None:
    response = await client.post(
        "/api/rv2v/detect-shots",
        json={
            "asset_id": asset_id,
            "frame_rate": 24.0,
            "sensitivity": "medium",
            "min_shot_frames": 12,
        },
    )
    assert response.status_code == status
    assert message in response.text


async def test_rv2v_shot_detection_blocks_inactive_endpoint(client) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["comfy_url"] = "http://other-comfy.test:8188"
    assert (await client.put("/api/settings", json=settings)).status_code == 200
    response = await client.post(
        "/api/rv2v/detect-shots",
        json={
            "asset_id": "fixture-video-source.mp4",
            "frame_rate": 24.0,
            "sensitivity": "low",
            "min_shot_frames": 12,
        },
    )
    assert response.status_code == 409
    assert "not the active endpoint" in response.text


@pytest.mark.parametrize(
    ("field", "value"),
    [("frame_rate", 0), ("sensitivity", "extreme"), ("min_shot_frames", 3)],
)
async def test_rv2v_shot_detection_validates_options(
    client, field: str, value: object
) -> None:
    body = {
        "asset_id": "fixture-video-source.mp4",
        "frame_rate": 24.0,
        "sensitivity": "medium",
        "min_shot_frames": 12,
    }
    body[field] = value
    assert (await client.post("/api/rv2v/detect-shots", json=body)).status_code == 422
