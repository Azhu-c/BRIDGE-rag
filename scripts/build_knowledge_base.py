#!/usr/bin/env python3
"""Build the external knowledge base from a YAML configuration."""
import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.knowledge_base.preprocess import run_preprocess
from src.knowledge_base.build_cfg_dataset import build_cfg_dataset
from src.knowledge_base.deduplicate_dataset import deduplicate_dataset
from src.knowledge_base.build_vector_index import build_vector_index


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Build the external knowledge base for retrieval-augmented lifting.")
    parser.add_argument("--config", default="configs/knowledge_base.yaml", help="Path to the YAML config file")
    parser.add_argument("--source-dir", help="Override the source directory containing raw C files or generated artifacts")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing even if enabled in the config")
    return parser.parse_args()


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


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

    source_dir = args.source_dir or config.get("input", {}).get("source_dir")
    if not source_dir:
        raise ValueError("Missing source_dir. Set input.source_dir in the config or pass --source-dir.")

    preprocess_cfg = config.get("preprocess", {})
    dataset_cfg = config.get("cfg_dataset", {})
    dedup_cfg = config.get("deduplication", {})
    index_cfg = config.get("vector_index", {})

    aligned_json = dataset_cfg.get("output_json", "./artifacts/knowledge_base/aligned_blocks.json")
    dedup_json = dedup_cfg.get("output_json", "./artifacts/knowledge_base/aligned_blocks_dedup.json")
    sqlite_db = dedup_cfg.get("sqlite_db", "./artifacts/knowledge_base/dedup_cache.db")
    index_dir = index_cfg.get("output_dir", "./artifacts/knowledge_base/index")

    ensure_parent(aligned_json)
    ensure_parent(dedup_json)
    ensure_parent(sqlite_db)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    if preprocess_cfg.get("enabled", True) and not args.skip_preprocess:
        run_preprocess(source_dir, jobs=int(preprocess_cfg.get("jobs", 8)))

    build_cfg_dataset(
        source_dir,
        aligned_json,
        optimization_levels=dataset_cfg.get("optimization_levels", [3]),
        ir_suffix=dataset_cfg.get("ir_suffix", "_0.ll"),
    )
    deduplicate_dataset(
        aligned_json,
        dedup_json,
        sqlite_db,
        ir_bb_max_len=int(dedup_cfg.get("ir_bb_max_len", 10000)),
        similarity_threshold=float(dedup_cfg.get("similarity_threshold", 0.95)),
        num_perm=int(dedup_cfg.get("num_perm", 128)),
    )
    build_vector_index(
        dedup_json,
        index_dir,
        model_name=index_cfg.get("model_name", "/path/to/Nova-1.3b-new-arm"),
        base_tokenizer_name=index_cfg.get("base_tokenizer_name", "deepseek-ai/deepseek-coder-1.3b-base"),
    )


if __name__ == "__main__":
    main()
