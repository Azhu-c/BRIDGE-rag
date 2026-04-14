#!/usr/bin/env python3
"""Build the external knowledge base from a YAML configuration."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.knowledge_base.preprocess import run_preprocess
from src.knowledge_base.build_cfg_dataset import build_cfg_dataset
from src.knowledge_base.deduplicate_dataset import deduplicate_dataset
from src.knowledge_base.build_vector_index import build_vector_index
from src.utils.config import apply_environment, load_yaml_config


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
    config = apply_environment(load_yaml_config(args.config))

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
        input_dir=source_dir,
        output_file=aligned_json,
        optimization_levels=dataset_cfg.get("optimization_levels", [3]),
        ir_suffix=dataset_cfg.get("ir_suffix", "_0.ll"),
    )
    deduplicate_dataset(
        input_file=aligned_json,
        output_file=dedup_json,
        sqlite_db=sqlite_db,
        ir_bb_max_len=int(dedup_cfg.get("ir_bb_max_len", 10000)),
        similarity_threshold=float(dedup_cfg.get("similarity_threshold", 0.95)),
        num_perm=int(dedup_cfg.get("num_perm", 128)),
    )
    build_vector_index(
        input_json=dedup_json,
        output_dir=index_dir,
        model_name=index_cfg.get("model_name", "/path/to/Nova-1.3b-new-arm"),
        base_tokenizer_name=index_cfg.get("base_tokenizer_name", "deepseek-ai/deepseek-coder-1.3b-base"),
        nova_module_dir=index_cfg.get("nova_module_dir"),
        index_filename=index_cfg.get("index_filename", "index_file.index"),
        ir_mapping_filename=index_cfg.get("ir_mapping_filename", "ir_files.pkl"),
        asm_mapping_filename=index_cfg.get("asm_mapping_filename", "asm_files.pkl"),
        max_len=int(index_cfg.get("max_len", 1024)),
        device=index_cfg.get("device"),
    )


if __name__ == "__main__":
    main()
