#!/usr/bin/env python3
"""Run binary lifting and iterative refinement from a YAML configuration."""
import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.binlift.rag_lifter import read_and_compile_json
from src.refinement.refinement_pipeline import run_refinement


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the lift + refinement pipeline.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to the YAML config file")
    parser.add_argument("--json-path", help="Override the input JSON path")
    parser.add_argument("--stage", choices=["lift", "refine", "all"], help="Override the configured pipeline stage")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

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

    pipeline_cfg = config.get("pipeline", {})
    refinement_cfg = config.get("refinement", {})
    input_json = args.json_path or pipeline_cfg.get("input_json") or os.environ.get("RAG_INPUT_JSON")
    if not input_json:
        raise ValueError("Missing input JSON. Set pipeline.input_json in the config or pass --json-path.")

    stage = args.stage or pipeline_cfg.get("stage", "all")

    api_key_env_name = env_cfg.get("deepseek_api_key_env", "DEEPSEEK_API_KEY")
    base_url_env_name = env_cfg.get("deepseek_base_url_env", "DEEPSEEK_BASE_URL")
    api_key = refinement_cfg.get("api_key") or os.environ.get(api_key_env_name)
    base_url = refinement_cfg.get("base_url") or os.environ.get(base_url_env_name) or "https://api.deepseek.com"

    if stage in {"lift", "all"} and config.get("lift", {}).get("enabled", True):
        read_and_compile_json(input_json)
    if stage in {"refine", "all"} and refinement_cfg.get("enabled", True):
        run_refinement(input_json, api_key=api_key, base_url=base_url)


if __name__ == "__main__":
    main()
