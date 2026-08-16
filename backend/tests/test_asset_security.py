from __future__ import annotations

import json
import sqlite3

import pytest

from director.app import _UPLOAD_LIMITS, _media_response, _read_upload_limited
from director.database import Database
from director.schemas import default_settings

from .conftest import runnable_draft, wait_for_submission_tasks


async def test_default_empty_draft_can_still_be_saved(client) -> None:
    draft = (await client.get("/api/drafts/i2v")).json()
    assert draft["shots"][0]["first_image"] is None

    response = await client.put("/api/drafts/i2v", json=draft)

    assert response.status_code == 200, response.text


async def test_draft_rejects_asset_without_upload_id(client) -> None:
    draft = runnable_draft("i2v")
    draft["shots"][0]["first_image"].pop("id")

    response = await client.put("/api/drafts/i2v", json=draft)

    assert response.status_code == 422
    assert '"id"' in response.text
    assert "Field required" in response.text


async def test_draft_rejects_unknown_asset_id_even_on_a_disabled_shot(client) -> None:
    draft = runnable_draft("i2v")
    draft["shots"][0]["enabled"] = False
    draft["shots"][0]["first_image"]["id"] = "not-uploaded"

    response = await client.put("/api/drafts/i2v", json=draft)

    assert response.status_code == 422
    assert "is not registered" in response.text


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("name", "forged.png", "name"),
        ("subfolder", "elsewhere", "subfolder"),
        ("path", "elsewhere/first.png", "path"),
    ],
)
async def test_draft_rejects_forged_asset_identity_or_path(
    client, field: str, value: str, expected: str
) -> None:
    draft = runnable_draft("i2v")
    draft["shots"][0]["first_image"][field] = value

    response = await client.put("/api/drafts/i2v", json=draft)

    assert response.status_code == 422
    assert expected in response.text


async def test_draft_rejects_claimed_kind_that_differs_from_registered_asset(client) -> None:
    draft = runnable_draft("i2v")
    reference = draft["shots"][0]["first_image"]
    reference.update(name="voice.wav", id="fixture-audio-voice.wav", kind="image")

    response = await client.put("/api/drafts/i2v", json=draft)

    assert response.status_code == 422
    assert "kind" in response.text


@pytest.mark.parametrize("bad_id", [None, "not-uploaded"])
async def test_job_create_revalidates_asset_upload_identity(client, fake_comfy, bad_id) -> None:
    draft = runnable_draft("v2v")
    if bad_id is None:
        draft["shots"][0]["source_video"].pop("id")
    else:
        draft["shots"][0]["source_video"]["id"] = bad_id

    response = await client.post("/api/jobs", json={"mode": "v2v", "config": draft})

    assert response.status_code == 422
    assert fake_comfy.prompts == []


async def test_job_create_accepts_an_exact_registered_asset(client, fake_comfy) -> None:
    response = await client.post(
        "/api/jobs", json={"mode": "i2v", "config": runnable_draft("i2v")}
    )

    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    assert len(fake_comfy.prompts) == 1


async def test_draft_rejects_forged_video_metadata(client) -> None:
    draft = runnable_draft("v2v")
    draft["shots"][0]["source_video"]["metadata"]["duration"] = 120.0

    response = await client.put("/api/drafts/v2v", json=draft)

    assert response.status_code == 422
    assert "metadata" in response.text


@pytest.mark.parametrize("mode", ["v2v", "r2v"])
def test_every_video_asset_requires_metadata(mode: str) -> None:
    draft = runnable_draft(mode)
    if mode == "v2v":
        draft["shots"][0]["source_video"].pop("metadata")
    else:
        draft["shots"][0]["reference_videos"][0].pop("metadata")

    from director.schemas import validate_mode_draft

    with pytest.raises(ValueError, match="server-probed metadata"):
        validate_mode_draft(mode, draft)


def test_upload_limits_match_the_public_policy() -> None:
    assert _UPLOAD_LIMITS == {
        "image": 32 * 1024 * 1024,
        "audio": 128 * 1024 * 1024,
        "video": 512 * 1024 * 1024,
    }


async def test_upload_reader_always_requests_bounded_chunks(monkeypatch) -> None:
    import director.app as app_module

    class Reader:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.offset = 0
            self.requested_sizes: list[int] = []

        async def read(self, size: int = -1) -> bytes:
            self.requested_sizes.append(size)
            if size < 0:
                raise AssertionError("unbounded read is forbidden")
            chunk = self.content[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    monkeypatch.setattr(app_module, "_UPLOAD_READ_CHUNK", 2)
    reader = Reader(b"12345")

    content = await _read_upload_limited(reader, 5)  # type: ignore[arg-type]

    assert content == b"12345"
    assert reader.requested_sizes == [2, 2, 2, 1]


async def test_upload_rejects_disguised_media_bytes(client, fake_comfy) -> None:
    response = await client.post(
        "/api/assets",
        data={"kind": "image"},
        files={"file": ("payload.png", b"<html><script>alert(1)</script>", "image/png")},
    )

    assert response.status_code == 422
    assert fake_comfy.uploads == []


async def test_asset_origin_blocks_execution_after_endpoint_switch(client, fake_comfy) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["comfy_url"] = "http://other-comfy.test:8188"
    assert (await client.put("/api/settings", json=settings)).status_code == 200

    draft = runnable_draft("i2v")
    saved = await client.put("/api/drafts/i2v", json=draft)
    submitted = await client.post("/api/jobs", json={"mode": "i2v", "config": draft})

    assert saved.status_code == 422
    assert submitted.status_code == 422
    assert "belongs to ComfyUI" in submitted.text
    assert fake_comfy.prompts == []


async def test_preview_uses_the_asset_origin_not_current_settings(client) -> None:
    seen_urls: list[str] = []
    app = client.director_app
    original_factory = app.state.comfy_factory

    def recording_factory(settings):
        seen_urls.append(str(settings.comfy_url).rstrip("/"))
        return original_factory(settings)

    app.state.comfy_factory = recording_factory
    settings = (await client.get("/api/settings")).json()
    settings["comfy_url"] = "http://other-comfy.test:8188"
    await client.put("/api/settings", json=settings)

    preview = await client.get("/api/assets/fixture-image-first.png/preview")

    assert preview.status_code == 200
    assert seen_urls[-1] == "http://comfy.test:8188"


def test_asset_origin_migration_backfills_from_persisted_settings(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    settings = default_settings("http://legacy-comfy.test:8188")
    document = {
        "name": "legacy.png",
        "subfolder": "director-web",
        "type": "input",
        "kind": "image",
        "id": "legacy-asset",
    }
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE settings(singleton INTEGER PRIMARY KEY, document TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE assets(id TEXT PRIMARY KEY, document TEXT NOT NULL, created_at TEXT NOT NULL);
            """
        )
        db.execute(
            "INSERT INTO settings VALUES(1, ?, 'then')", (settings.model_dump_json(),)
        )
        db.execute(
            "INSERT INTO assets VALUES('legacy-asset', ?, 'then')",
            (json.dumps(document),),
        )

    database = Database(path)
    database.initialize()

    assert database.get_asset_record("legacy-asset") == (
        document,
        "http://legacy-comfy.test:8188",
    )


def test_media_response_is_nosniff_and_unicode_header_is_latin1_safe() -> None:
    import httpx

    upstream = httpx.Response(200, content=b"media", headers={"content-type": "image/png"})
    response = _media_response(upstream, filename='测试\r\n".png')

    assert response.headers["x-content-type-options"] == "nosniff"
    disposition = response.headers["content-disposition"]
    disposition.encode("latin-1")
    assert "filename*=UTF-8''" in disposition
    assert "%E6%B5%8B%E8%AF%95" in disposition
    assert "\r" not in disposition and "\n" not in disposition


def test_media_response_forces_attachment_on_type_mismatch() -> None:
    import httpx

    upstream = httpx.Response(200, content=b"html", headers={"content-type": "text/html"})
    response = _media_response(upstream, filename="preview.png")

    assert response.media_type == "application/octet-stream"
    assert response.headers["content-disposition"].startswith("attachment;")
