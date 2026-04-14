#!/usr/bin/env python3
"""Run binary lifting and iterative refinement from a YAML configuration."""
from __future__ import annotations

import argparse

from src.binlift.rag_lifter import read_and_compile_json
from src.refinement.refinement_pipeline import run_refinement
from src.utils.config import apply_environment, load_yaml_config, resolve_llm_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Run the lift + refinement pipeline.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to the YAML config file")
    parser.add_argument("--json-path", help="Override the input JSON path")
    parser.add_argument("--stage", choices=["lift", "refine", "all"], help="Override the configured pipeline stage")
    return parser.parse_args()


def main():
    args = parse_args()
    config = apply_environment(load_yaml_config(args.config))

    pipeline_cfg = config.get("pipeline", {})
    lift_cfg = config.get("lift", {})
    refinement_cfg = config.get("refinement", {})
    llm_cfg = resolve_llm_settings(config, section_name="llm")
    input_json = args.json_path or pipeline_cfg.get("input_json")
    if not input_json:
        raise ValueError("Missing input JSON. Set pipeline.input_json in the config or pass --json-path.")

    stage = args.stage or pipeline_cfg.get("stage", "all")

    if stage in {"lift", "all"} and lift_cfg.get("enabled", True):
        read_and_compile_json(
            json_file=input_json,
            output_dir=lift_cfg.get("output_dir", "./artifacts/pipeline/lift_outputs"),
            llm_model_path=lift_cfg.get("llm_model_path", "/path/to/Nova-1.3b-new-arm"),
            llm_tokenizer_name=lift_cfg.get("llm_tokenizer_name", "deepseek-ai/deepseek-coder-1.3b-base"),
            nova_module_dir=lift_cfg.get("nova_module_dir"),
            model_device=lift_cfg.get("device"),
            index_file=lift_cfg.get("index_file", "/path/to/index_file.index"),
            mapping_file=lift_cfg.get("mapping_file", "/path/to/ir_files.pkl"),
            mapping_file_asm=lift_cfg.get("mapping_file_asm", "/path/to/asm_files.pkl"),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
            llm_model_name=llm_cfg.get("model_name"),
            llm_timeout=int(llm_cfg.get("timeout", 300)),
            llm_system_prompt=llm_cfg.get("system_prompt"),
            command_timeout=int(lift_cfg.get("command_timeout", 200)),
            temp_dir=lift_cfg.get("temp_dir", "/tmp"),
            clang_target=lift_cfg.get("clang_target", "aarch64-linux-gnu"),
            qemu_binary=lift_cfg.get("qemu_binary", "qemu-aarch64"),
            qemu_library_path=lift_cfg.get("qemu_library_path", "/usr/aarch64-linux-gnu/"),
        )
    if stage in {"refine", "all"} and refinement_cfg.get("enabled", True):
        run_refinement(
            json_path=input_json,
            input_ll_dir=refinement_cfg.get("input_ll_dir", "./artifacts/pipeline/lift_outputs"),
            output_ll_dir=refinement_cfg.get("output_ll_dir", "./artifacts/pipeline/refined_outputs"),
            file_pattern=refinement_cfg.get("file_pattern", "onlyfunc_{task_id}_{opt}.o.ll"),
            retry_limit=int(refinement_cfg.get("retry_limit", 5)),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
            llm_model_name=llm_cfg.get("model_name"),
            llm_timeout=int(llm_cfg.get("timeout", 300)),
            temp_dir=refinement_cfg.get("temp_dir", "/tmp"),
        )


if __name__ == "__main__":
    main()
