from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from directordeck.app import create_app
from directordeck.config_manager import DirectorDeckConfig, DirectorDeckConfigManager


def _document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "lora": {
            "loaders": [
                {
                    "id": "model_only",
                    "display_name": "LoRA加载器（仅模型）",
                    "class_type": "LoraLoaderModelOnly",
                    "input_contract": "model_only",
                    "supported_families": ["fl2va", "ref2va"],
                    "options": [],
                },
                {
                    "id": "minimax_h3_turbo",
                    "display_name": "MiniMax-H3 Turbo LoRA",
                    "class_type": "MiniMaxH3TurboLoRA",
                    "input_contract": "dedicated_model",
                    "supported_families": ["fl2va", "ref2va"],
                    "options": [{
                        "id": "low_vram",
                        "type": "boolean",
                        "label": "low_vram",
                        "description": "低显存模式",
                        "default": False,
                    }],
                },
            ],
            "fallback_policy": {
                "loader_ids": ["model_only"],
                "default_loader_id": "model_only",
            },
            "loader_policies": [{
                "lora_filename": r"minimax_h3_turbo_.*\.safetensors$",
                "loader_ids": ["minimax_h3_turbo"],
                "default_loader_id": "minimax_h3_turbo",
            }],
        },
    }


def test_config_manager_reads_and_validates_only_once(tmp_path) -> None:
    source = tmp_path / "directordeck.json"
    source.write_text(json.dumps(_document()), encoding="utf-8")
    manager = DirectorDeckConfigManager()

    first = manager.initialize(source)
    source.write_text("not json", encoding="utf-8")
    second = manager.initialize(source)

    assert second is first
    assert first.default_loader_id == "model_only"
    assert [loader.id for loader in first.loaders] == [
        "model_only",
        "minimax_h3_turbo",
    ]
    assert first.normalize_lora_loader_options(
        "minimax_h3_turbo",
        {},
    ) == {"low_vram": False}
    assert manager.lora_loader_policy(
        "nested/minimax_h3_turbo_v4_step600.safetensors"
    ).default_loader_id == "minimax_h3_turbo"
    fallback = manager.lora_loader_policy("nested/style.safetensors")
    assert fallback.loader_ids == ("model_only",)
    assert fallback.default_loader_id == "model_only"


def test_config_manager_rejects_a_second_source(tmp_path) -> None:
    first_source = tmp_path / "first.json"
    second_source = tmp_path / "second.json"
    payload = json.dumps(_document())
    first_source.write_text(payload, encoding="utf-8")
    second_source.write_text(payload, encoding="utf-8")
    manager = DirectorDeckConfigManager()
    manager.initialize(first_source)

    with pytest.raises(RuntimeError, match="already initialized"):
        manager.initialize(second_source)


def test_config_manager_isolates_an_invalid_policy_regex(tmp_path) -> None:
    document = _document()
    lora = document["lora"]
    assert isinstance(lora, dict)
    policies = lora["loader_policies"]
    assert isinstance(policies, list)
    policies[0]["lora_filename"] = "["
    source = tmp_path / "invalid-regex.json"
    source.write_text(json.dumps(document), encoding="utf-8")

    manager = DirectorDeckConfigManager()
    snapshot = manager.initialize(source)

    assert [loader.id for loader in snapshot.loaders] == [
        "model_only",
        "minimax_h3_turbo",
    ]
    assert snapshot.lora.loader_policies == ()
    assert [item.code for item in manager.diagnostics()] == [
        "lora_loader_policy_invalid"
    ]
    with pytest.raises(ValueError, match="regular expression"):
        DirectorDeckConfig.model_validate(document)


def test_config_manager_isolates_an_invalid_unused_loader(tmp_path) -> None:
    document = _document()
    lora = document["lora"]
    assert isinstance(lora, dict)
    loaders = lora["loaders"]
    assert isinstance(loaders, list)
    loaders.append({"id": "broken loader"})
    source = tmp_path / "invalid-unused-loader.json"
    source.write_text(json.dumps(document), encoding="utf-8")

    manager = DirectorDeckConfigManager()
    snapshot = manager.initialize(source)

    assert [loader.id for loader in snapshot.loaders] == [
        "model_only",
        "minimax_h3_turbo",
    ]
    assert [item.code for item in manager.diagnostics()] == [
        "lora_loader_entry_invalid"
    ]


def test_config_manager_rejects_an_unknown_fallback_loader(tmp_path) -> None:
    document = _document()
    lora = document["lora"]
    assert isinstance(lora, dict)
    lora["fallback_policy"] = {
        "loader_ids": ["not_installed"],
        "default_loader_id": "not_installed",
    }
    source = tmp_path / "invalid-fallback.json"
    source.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unavailable loader"):
        DirectorDeckConfigManager().initialize(source)


def test_first_matching_policy_wins(tmp_path) -> None:
    document = _document()
    lora = document["lora"]
    assert isinstance(lora, dict)
    policies = lora["loader_policies"]
    assert isinstance(policies, list)
    policies.insert(0, {
        "lora_filename": "minimax_h3",
        "loader_ids": ["model_only"],
        "default_loader_id": "model_only",
    })
    source = tmp_path / "ordered-policies.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    manager = DirectorDeckConfigManager()
    manager.initialize(source)

    policy = manager.lora_loader_policy("minimax_h3_turbo_v4.safetensors")
    assert policy is not None
    assert policy.default_loader_id == "model_only"


def test_configured_default_loader_may_have_defaulted_options(tmp_path) -> None:
    document = _document()
    lora = document["lora"]
    assert isinstance(lora, dict)
    lora["fallback_policy"] = {
        "loader_ids": ["model_only", "minimax_h3_turbo"],
        "default_loader_id": "minimax_h3_turbo",
    }
    source = tmp_path / "configured-default.json"
    source.write_text(json.dumps(document), encoding="utf-8")

    config = DirectorDeckConfigManager().initialize(source)

    assert config.default_loader_id == "minimax_h3_turbo"
    assert config.normalize_lora_loader_options(
        config.default_loader_id,
        {},
    ) == {"low_vram": False}


async def test_product_config_api_exposes_loaders_without_claiming_node_ownership(
    client,
) -> None:
    response = await client.get("/api/config")

    assert response.status_code == 200, response.text
    document = response.json()
    assert document["diagnostics"] == []
    assert document["lora"]["fallback_policy"] == {
        "loader_ids": ["model_only"],
        "default_loader_id": "model_only",
    }
    assert [loader["id"] for loader in document["lora"]["loaders"]] == [
        "model_only",
        "minimax_h3_turbo",
    ]
    assert document["lora"]["loader_policies"] == [{
        "lora_filename": r"minimax_h3_turbo_.*\.safetensors$",
        "loader_ids": ["minimax_h3_turbo"],
        "default_loader_id": "minimax_h3_turbo",
    }]
    assert all(
        "node_contract_id" not in loader
        for loader in document["lora"]["loaders"]
    )
    assert all(
        "semantic_version" not in loader
        for loader in document["lora"]["loaders"]
    )


async def test_invalid_product_config_does_not_block_database_startup(
    tmp_path,
    fake_comfy,
    monkeypatch,
) -> None:
    database_path = tmp_path / "config-failure.sqlite3"

    def fail_initialization():
        raise RuntimeError("sensitive config parser detail")

    monkeypatch.setattr(
        "directordeck.app.initialize_directordeck_config",
        fail_initialization,
    )
    app = create_app(
        database_path=database_path,
        comfy_url="http://comfy.test:8188",
        comfy_factory=lambda _url: fake_comfy,
    )
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=app,
                raise_app_exceptions=True,
            ),
            base_url="http://testserver",
        ) as test_client:
            settings = await test_client.get("/api/settings")
            product_config = await test_client.get("/api/config")

    assert database_path.exists()
    assert settings.status_code == 200, settings.text
    assert product_config.status_code == 200, product_config.text
    assert product_config.json() == {
        "schema_version": 1,
        "lora": {
            "loaders": [],
            "fallback_policy": None,
            "loader_policies": [],
        },
        "diagnostics": [{
            "code": "lora_product_config_unavailable",
            "message": "DirectorDeck's LoRA loader configuration is unavailable.",
        }],
    }
    assert "sensitive config parser detail" not in product_config.text
