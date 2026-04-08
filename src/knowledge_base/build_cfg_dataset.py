#!/usr/bin/env python3
import json
import os
from typing import Iterable

from src.knowledge_base.process_ir import process_llvm_ir_cfg
from src.knowledge_base.process_asm import probe_asm
from src.knowledge_base.cfg_loader import re_probes


def build_cfg_dataset(input_dir: str, output_file: str, optimization_levels: Iterable[int] = (3,), ir_suffix: str = "_0.ll") -> list[dict]:
    """Build an ASM/IR block-aligned dataset from a directory of generated artifacts."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_list: list[dict] = []

    for root, _, files in os.walk(input_dir):
        c_files = [f for f in files if f.endswith('.c')]
        for c_file in c_files:
            base_name = os.path.splitext(c_file)[0]
            group_str = base_name.split("_")[1] if "_" in base_name else base_name
            ir_file = os.path.join(root, f"{base_name}{ir_suffix}")
            if not os.path.exists(ir_file):
                continue

            ir_result = process_llvm_ir_cfg(ir_file)
            for opt_level in optimization_levels:
                asm_file = os.path.join(root, f"{base_name}_{opt_level}.asm")
                txt_file = os.path.join(root, f"{base_name}_{opt_level}.txt")
                if not (os.path.exists(asm_file) and os.path.exists(txt_file)):
                    continue

                asm_blocks = probe_asm(txt_file, asm_file)
                for ir_func in ir_result:
                    func_name = ir_func["name"]
                    try:
                        re_index = re_probes(ir_func['blocks'])
                    except Exception as exc:
                        print(f"[WARN] Failed to rebuild probes for {func_name}: {exc}")
                        continue

                    if func_name not in asm_blocks:
                        print(f"[WARN] Function {func_name} in group {group_str} could not be matched.")
                        continue

                    for indexes in re_index:
                        target_set = set(indexes)
                        asm_struc = [node['content'] for node in asm_blocks[func_name] if target_set.intersection(node.get('index', []))]
                        ir_struc = [node['content'] for node in ir_func['blocks'] if target_set.intersection(node.get('probe_indexes', []))]
                        output_list.append({
                            "group": group_str,
                            "function": func_name,
                            "type": opt_level,
                            "ir_bb": "\n".join(ir_struc),
                            "asm_bb": "\n".join(asm_struc),
                        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(output_list)} aligned samples to {output_file}")
    return output_list
