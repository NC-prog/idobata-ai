"""config.py のプロファイル解決のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    REPO_ROOT,
    ConfigError,
    Profile,
    build_config,
    load_env_file,
)


class TestBuildConfig:
    def test_default_is_standalone_localhost(self) -> None:
        cfg = build_config({})
        assert cfg.profile is Profile.STANDALONE
        assert cfg.llm.gateway_url == "http://127.0.0.1:8001/v1"
        assert cfg.asr.base_url == "http://127.0.0.1:8005/v1"
        assert cfg.voicevox.base_url == "http://127.0.0.1:50021"
        assert cfg.llm.chat_model == "/models/llm"
        assert cfg.asr.model == "/models/asr"
        assert cfg.llm.api_key == "dummy"

    def test_site_uses_backend_url_for_llm_only(self) -> None:
        cfg = build_config({
            "IDOBATA_PROFILE": "site",
            "IDOBATA_BACKEND_URL": "http://t2.tailnet.ts.net:8001/v1/",
        })
        assert cfg.profile is Profile.SITE
        assert cfg.llm.gateway_url == "http://t2.tailnet.ts.net:8001/v1"
        assert cfg.asr.base_url.startswith("http://127.0.0.1:")
        assert cfg.voicevox.base_url.startswith("http://127.0.0.1:")

    def test_site_without_backend_url_fails(self) -> None:
        with pytest.raises(ConfigError):
            build_config({"IDOBATA_PROFILE": "site"})

    def test_unknown_profile_fails(self) -> None:
        with pytest.raises(ConfigError):
            build_config({"IDOBATA_PROFILE": "cloud"})

    def test_site_selection(self) -> None:
        assert build_config({"IDOBATA_SITE": "akiota"}).site == "akiota"


class TestLoadEnvFile:
    def test_does_not_override_existing(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("# c\nA=1\nB='two'\n\nC=x\n", encoding="utf-8")
        env = {"C": "keep"}
        load_env_file(env_file, env)
        assert env == {"A": "1", "B": "two", "C": "keep"}

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        env: dict[str, str] = {}
        load_env_file(tmp_path / "none", env)
        assert env == {}


def test_no_townhall_internal_address_in_repo() -> None:
    """townhall の社内サーバ固定アドレスが残っていないこと。"""
    banned = "192.168." + "11.3"
    hits = [
        p for p in (REPO_ROOT / "server").rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml", ".toml", ".sh", ".md"}
        and ".venv" not in p.parts and banned in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert hits == []
