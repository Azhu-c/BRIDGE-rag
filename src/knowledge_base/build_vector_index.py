from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer


def _import_nova(nova_module_dir: Optional[str] = None):
    if nova_module_dir and nova_module_dir not in sys.path:
        sys.path.append(nova_module_dir)
    from modeling_nova import NovaTokenizer, NovaForCausalLM  # type: ignore
    return NovaTokenizer, NovaForCausalLM


def clean_asm(asm: str) -> str:
    asm_clean = ""
    for line in asm.split("\n"):
        asm_clean += re.sub(r"^\s*[\da-fA-F]+:\s*", "", line, flags=re.MULTILINE) + "\n"
    return asm_clean


def load_nova_model_and_tokenizer(
    model_name: str,
    base_tokenizer_name: str,
    nova_module_dir: Optional[str] = None,
):
    NovaTokenizer, NovaForCausalLM = _import_nova(nova_module_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, trust_remote_code=True)
    tokenizer.add_tokens(["<unk>", "<cls>"] + [f"<label-{i}>" for i in range(1, 257)], special_tokens=True)
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print("Vocabulary:", len(tokenizer.get_vocab()))
    nova_tokenizer = NovaTokenizer(tokenizer)
    model = NovaForCausalLM.from_pretrained(model_name, device_map="auto").eval()
    token_id = tokenizer.encode("<label-1>")[1]
    return tokenizer, nova_tokenizer, model, token_id


def asm2vector_nova(
    asm: str,
    nova_tokenizer,
    model,
    token_id: int,
    max_len: int = 1024,
    device: Optional[str] = None,
):
    asm = asm.strip()
    char_types = ("0" * len("<func0>:") + "1" * max(0, len(asm) - len("<func0>:")))
    temp = nova_tokenizer.encode("", asm, char_types)
    input_ids = temp["input_ids"][:max_len].tolist()
    nova_attention_mask = temp["nova_attention_mask"][:max_len, :max_len]
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).unsqueeze(0).to(target_device)
    nova_attention_mask = torch.as_tensor(nova_attention_mask, dtype=torch.bool).unsqueeze(0).to(target_device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            nova_attention_mask=nova_attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        embedding = hidden[0][input_ids[0] >= token_id].mean(dim=0)
    return embedding.detach().cpu().numpy().astype(np.float32)


def build_vector_index(
    input_json: str,
    output_dir: str,
    model_name: str,
    base_tokenizer_name: str,
    nova_module_dir: Optional[str] = None,
    index_filename: str = "index_file.index",
    ir_mapping_filename: str = "ir_files.pkl",
    asm_mapping_filename: str = "asm_files.pkl",
    max_len: int = 1024,
    device: Optional[str] = None,
):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    asm_bb_list, ir_bb_list = [], []
    for item in data:
        asm = item.get("asm_bb", "")
        ir_bb = item.get("ir_bb", "")
        if isinstance(asm, str) and asm.strip() and isinstance(ir_bb, str) and ir_bb.strip():
            asm_bb_list.append(asm)
            ir_bb_list.append(ir_bb)

    print(f"Encoding {len(asm_bb_list)} ASM blocks")
    _, nova_tokenizer, model, token_id = load_nova_model_and_tokenizer(
        model_name=model_name,
        base_tokenizer_name=base_tokenizer_name,
        nova_module_dir=nova_module_dir,
    )
    asm_bb_embeddings = [
        asm2vector_nova(asm_text, nova_tokenizer, model, token_id, max_len=max_len, device=device)
        for asm_text in asm_bb_list
    ]
    asm_bb_embeddings = np.vstack(asm_bb_embeddings).astype(np.float32)

    dim = asm_bb_embeddings.shape[1]
    base_index = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap(base_index)
    ids = np.arange(len(asm_bb_embeddings), dtype=np.int64)
    index.add_with_ids(asm_bb_embeddings, ids)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / index_filename
    ir_path = out / ir_mapping_filename
    asm_path = out / asm_mapping_filename
    faiss.write_index(index, str(index_path))
    with open(ir_path, "wb") as f:
        pickle.dump(ir_bb_list, f)
    with open(asm_path, "wb") as f:
        pickle.dump(asm_bb_list, f)
    print(f"Saved index to {index_path}")
    return str(index_path), str(ir_path), str(asm_path)
