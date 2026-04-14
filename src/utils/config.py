from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_yaml_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_environment(config: dict[str, Any]) -> dict[str, Any]:
    env_cfg = config.get("environment", {})
    dotenv_path = env_cfg.get("dotenv_path")
    if dotenv_path and Path(dotenv_path).exists():
        load_dotenv(dotenv_path)

    model_cache_dir = os.environ.get("MODEL_CACHE_DIR") or env_cfg.get("model_cache_dir")
    if model_cache_dir:
        os.environ.setdefault("TRANSFORMERS_CACHE", model_cache_dir)

    llvm_tmp_dir = os.environ.get("LLVM_TMP_DIR") or env_cfg.get("llvm_tmp_dir")
    if llvm_tmp_dir:
        os.environ.setdefault("TMPDIR", llvm_tmp_dir)

    return config


def resolve_llm_settings(config: dict[str, Any], section_name: str = "llm") -> dict[str, Any]:
    env_cfg = config.get("environment", {})
    section = config.get(section_name, {})
    api_key_env_name = env_cfg.get("deepseek_api_key_env", "DEEPSEEK_API_KEY")
    base_url_env_name = env_cfg.get("deepseek_base_url_env", "DEEPSEEK_BASE_URL")

    return {
        "provider": section.get("provider", "deepseek"),
        "api_key": section.get("api_key") or os.environ.get(api_key_env_name),
        "base_url": section.get("base_url") or os.environ.get(base_url_env_name) or "https://api.deepseek.com",
        "model_name": section.get("model_name", "deepseek-chat"),
        "timeout": int(section.get("timeout", 300)),
        "system_prompt": section.get("system_prompt", "You are an expert reverse engineer and compiler architect."),
    }
