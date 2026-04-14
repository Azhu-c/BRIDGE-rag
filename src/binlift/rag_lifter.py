import os
import sys
import json
import subprocess
import tempfile
import re
import editdistance
import math
from collections import defaultdict
from src.llm.llm_api import call_llm_disassembler, create_openai_client
import concurrent.futures
from transformers import AutoTokenizer, AutoModel
import faiss
import pickle
import torch
import numpy as np
from modeling_nova import NovaTokenizer, NovaForCausalLM

import traceback

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def fix_llvm_ir_errors(ir_content):
    print("1111")

    ir_content = fix_metadata_errors(ir_content)
    
    
    return ir_content


def fix_metadata_errors(ir_content):
    """metadata"""
    original_content = ir_content

    ir_content = re.sub(r',\s*!llvm\.loop\s+![0-9]+', '', ir_content)
    
    ir_content = re.sub(r',\s*!dbg\s+![0-9]+', '', ir_content)

    ir_content = re.sub(r',\s*![0-9]+\s*$', '', ir_content, flags=re.MULTILINE)

    ir_content = re.sub(r'\s+![0-9]+\s*$', '', ir_content, flags=re.MULTILINE)
    
    ir_content = re.sub(r',\s*,', ',', ir_content)  # comment removed
    ir_content = re.sub(r',\s*\n', '\n', ir_content)  # comment removed
    
    if ir_content != original_content:
        print("metadata")
    
    return ir_content


# def clean_asm(asm):
#     asm_clean = " "
#     for tmp in asm.split("\n"):   
#         tmp_asm = re.sub(r'^\s*[\da-fA-F]+:\s*', '', tmp, flags=re.MULTILINE)
#         asm_clean += tmp_asm + "\n"
#     asm = asm_clean
#     return asm


def hex_to_decimal(matched):
    return str(int(matched.group(), 16))
def normalize_asm(asm):
    asm = asm.strip().split('\n')[: 257]

    asm_lst = []
    addr2label = {}
    func_cnt, label_cnt = 0, 0

    for i, line in enumerate(asm):
        if line.strip() == '' or 'file format elf64-x86-64' in line:
            continue

        if len(line.split('\t')) == 1 and line.endswith(':'):
            func = line[line.index('<') + 1 : line.index('>')]
            asm_lst.append([f'<func{func_cnt}>:'])
            func_cnt += 1
        else:
            addr = None
            content = None

            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    addr, content = parts[0], parts[1]

            if (addr is None or content is None) and (':' in line):
                left, right = line.split(':', 1)
                if right.strip() != '':
                    addr = left.strip() + ':'  # comment removed
                    content = right.strip()

            if addr is None or content is None:
                print(line)
                continue

            label_cnt += 1

            addr = addr[:-1]
            addr2label[addr] = f'<label-{label_cnt}>'
            asm_lst.append([content.strip(), f'<label-{label_cnt}>'])

    new_asm = ''
    for i, item in enumerate(asm_lst):
        if len(item) == 1:
            new_asm += '\n' + item[0]
            continue

        content, label = item

        if '<' in content and '>' in content:
            content = content[: content.index('<')].strip()

        if content.startswith('j') or content.startswith('loop') or content.startswith('call'):
            if len(content.split()) == 2:
                inst, addr = content.split()
                if addr.startswith('0x'):
                    addr = addr[2:]
                if addr not in addr2label:
                    content = inst + '\t' + '<unk>'
                else:
                    content = inst + '\t' + addr2label[addr]

        content = re.sub(r"0x([0-9A-Fa-f]+)", hex_to_decimal, content)
        content = content.replace('%', '')
        content = re.sub(r"([,(])|([),])", r' \1\2 ', content)
        content = re.sub(r' +', ' ', content).strip()

        new_asm += '\n' + content + '\t' + label

    return new_asm




#RAG
def load_llm_model(model_name: str, base_tokenizer_name: str, device: str = DEFAULT_DEVICE, nova_module_dir: str | None = None):
    if nova_module_dir and nova_module_dir not in sys.path:
        sys.path.append(nova_module_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        base_tokenizer_name,
        trust_remote_code=True
    )

    tokenizer.add_tokens(
        ['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)],
        special_tokens=True
    )

    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print('Vocabulary:', len(tokenizer.get_vocab()))

    nova_tokenizer = NovaTokenizer(tokenizer)

    model = NovaForCausalLM.from_pretrained(
        model_name,
        device_map='auto'
    ).eval()

    ID = tokenizer.encode('<label-1>')[1]

    return nova_tokenizer, model, ID
def asm_to_bert_vector(asm: str, nova_tokenizer, model, ID, device: str = DEFAULT_DEVICE, max_len: int = 1024):
    asm = asm.strip()

    char_types = ('0' * len('<func0>:') +'1' * (len(asm) - len('<func0>:')))

    temp = nova_tokenizer.encode('', asm, char_types)

    input_ids = temp['input_ids'][:max_len].tolist()
    nova_attention_mask = temp['nova_attention_mask'][:max_len, :max_len]

    # input_ids = torch.LongTensor([input_ids]).to(device)
    # nova_attention_mask = torch.tensor([nova_attention_mask],dtype=torch.bool,device=device)
    input_ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).unsqueeze(0).to(device)
    nova_attention_mask = torch.as_tensor(nova_attention_mask, dtype=torch.bool).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            nova_attention_mask=nova_attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        h = outputs.hidden_states[-1]          # [1, T, H]
        e = h[0][input_ids[0] >= ID].mean(dim=0)

    return e.cpu().numpy().astype(np.float32)

def load_index_and_mapping(index_file, mapping_file, mapping_file_asm):
    if not os.path.exists(index_file):
        raise FileNotFoundError(f" {index_file} ")
    index = faiss.read_index(index_file)

    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f" {mapping_file} ")
    with open(mapping_file, "rb") as f:
        ir_bb_list = pickle.load(f)
    if not os.path.exists(mapping_file_asm):
        raise FileNotFoundError(f" {mapping_file_asm} ")
    with open(mapping_file_asm, "rb") as f:
        asm_bb_list = pickle.load(f)

    return index, ir_bb_list, asm_bb_list

# Function to calculate edit similarity
def normalize_spaces(s):
    if not s:
        s = ''
    s = re.sub(r'([.,!?();+\-/*{}^&=!])', r' \1 ', s)  # comment removed
    s = ' '.join(s.split())
    return s

def normalize_variables(s):
    s = re.sub(r'%[\w\.\-]+', '%VAR', s)
    return s

def preprocess(s):
    s = normalize_spaces(s)
    s = normalize_variables(s)
    return s

def get_normalized_edit_similarity(ground_truth, pred):
    if not pred:
        return 0.0, False

    gt = preprocess(ground_truth)
    out = preprocess(pred)

    exact_match = (gt == out)
    ed = editdistance.distance(gt, out)
    normalized = ed / max(len(gt), len(out))
    return 1 - normalized
##——————————————————————————————————————————————————————————————##
def sanitize_prompt_ir(ir_content):
    """ RAG  IR， metadata/"""
    ir_content = fix_metadata_errors(ir_content)
    ir_content = re.sub(r'^\s*!\d+\s*=\s*.*$', '', ir_content, flags=re.MULTILINE)
    ir_content = re.sub(r'^\s*!llvm\..*$', '', ir_content, flags=re.MULTILINE)
    ir_content = re.sub(r'^\s*declare\s+[^{\n]+$', '', ir_content, flags=re.MULTILINE)
    ir_content = re.sub(r'^\s*@\.[^\n]*$', '', ir_content, flags=re.MULTILINE)
    ir_content = re.sub(r'\n{3,}', '\n\n', ir_content)
    return ir_content.strip()
##————————————————————————————————————————————————————————————————##

def read_and_compile_json(
    json_file: str,
    output_dir: str,
    llm_model_path: str,
    llm_tokenizer_name: str,
    index_file: str = "/path/to/index_file.index",
    mapping_file: str = "/path/to/ir_files.pkl",
    mapping_file_asm: str = "/path/to/asm_files.pkl",
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
    llm_model_name: str = "deepseek-chat",
    llm_timeout: int = 300,
    llm_system_prompt: str | None = None,
    command_timeout: int = 200,
    temp_dir: str = "/tmp",
    clang_target: str = "aarch64-linux-gnu",
    qemu_binary: str = "qemu-aarch64",
    qemu_library_path: str = "/usr/aarch64-linux-gnu/",
    model_device: str = DEFAULT_DEVICE,
    nova_module_dir: str | None = None,
):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Counters for successful compilations and executions
    successful_disassembly_compilations = 0
    successful_executions = 0
    total_entries = len(data)

    edit_similarities = [] # List to store edit similarities
    
    # Define a default timeout in seconds
    COMMAND_TIMEOUT = command_timeout

    stats_by_type = defaultdict(lambda: {
        'count': 0,
        'successful_disassembly_compilations': 0,
        'successful_executions': 0,
        'edit_similarities': []
    })

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    asm_tokenizer, asm_model, ID = load_llm_model(
        model_name=llm_model_path,
        base_tokenizer_name=llm_tokenizer_name,
        device=model_device,
        nova_module_dir=nova_module_dir,
    )
    index, ir_bb_list, asm_bb_list = load_index_and_mapping(index_file, mapping_file, mapping_file_asm)
    client = create_openai_client(api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""), base_url=base_url)

    # Iterate through each entry in the JSON data
    for i, entry in enumerate(data):
        task_id = entry.get("task_id", f"unknown_task_{i}") # Use task_id from JSON or create a unique one
        test_type = entry.get("type", "unknown_type") # Use type from JSON or default

        ir_func = entry.get("ir_func", "")
        ir_test = entry.get("ir_test", "")
        asm_input = entry.get("input_asm_prompt", "")

        if not ir_func or not ir_test:
            print(f"Entry {task_id} is missing 'ir_func' or 'ir_test' key! Skipping this entry.")
            continue

        stats_by_type[test_type]['count'] += 1
        # Define paths for source C file and the executable - ARM version
        source_c_dir = output_dir
        os.makedirs(source_c_dir, exist_ok=True)
        
        func_executable_name = f"onlyfunc_{task_id}_{test_type}_exe"
        func_executable_path = os.path.join(source_c_dir, func_executable_name)


        # Run llvm-mctoll
        disassembled_ll_file_path = f"{func_executable_path}-deepseek-rag.ll"
        try:

            prompt = (
                "#Task: Translate the given ARM64 (aarch64-linux-gnu, little-endian) assembly into **one** LLVM IR function compiled at  O0-O3.\n"
                "#Requirements:\n"
                "1. Output LLVM IR only, no explanations or comments.\n"
                "2. Target architecture must remain ARM64; prefer i64/ptr types and alignments that match 64-bit ABI (e.g., align 8). Avoid emitting x86-specific constructs.\n"
                "3. Do NOT introduce helpers, globals, or library calls that are absent from the assembly (e.g., @malloc, @strlen, @helper_func). Inline all logic inside the single function.\n"
                "4. Remove/avoid metadata such as !llvm.* or !dbg and do not invent custom intrinsics (e.g., llvm.return.*).\n"
                "5. Keep SSA names simple and ordered (e.g., %val0, %val1). Maintain strict type consistency—never treat an i64 as a ptr or vice versa.\n"
                "6. If the assembly relies on SIMD instructions, rewrite the behavior as equivalent scalar loops.\n"
                "7. For any contiguous stack allocation (e.g., stack frame size like 0xFA0 bytes), you must represent it using a single aggregate allocation, such as alloca [1000 x i32] or alloca i8, i64 4000 followed by appropriate bitcast / getelementptr. Do NOT expand such memory into many independent alloca instructions (e.g., val0, val1, ..., valN).\n"
            )
            asm_em_input = normalize_asm(asm_input)
            new_embedding = asm_to_bert_vector(asm_em_input, asm_tokenizer, asm_model, ID, device, 1024)
            #indices = index.search(new_embedding, 1)
            #print(type(new_embedding))
            new_embedding = np.asarray(new_embedding, dtype=np.float32)
            if new_embedding.ndim == 1:
                new_embedding = new_embedding.reshape(1, -1)
            print(new_embedding.shape)
            distances, indices = index.search(new_embedding, 1)
            
            prompt += f"#asm-IR pair Example:\n "

            for i, idx in enumerate(indices[0]):
                example_ir = sanitize_prompt_ir(ir_bb_list[idx])
                prompt += f"ir:'''llvm\n{example_ir}\n'''\nasm:'''asm\n{asm_bb_list[idx]}\n'''"
            prompt += (
                "#Translate the source ARM assembly into LLVM IR (follow all requirements above) and output IR only:\n"
                "'''asm\n"
                f"{asm_input}\n"
                "'''\n"
            )
            print(prompt)

            disassembled_ir_func = call_llm_disassembler(client, prompt)

            if not disassembled_ir_func.strip():
                print(f"LLM returned empty IR for {func_executable_path}")
                continue

            print("...")
            disassembled_ir_func = fix_llvm_ir_errors(disassembled_ir_func)

            with open(disassembled_ll_file_path, 'w') as f:
                f.write(disassembled_ir_func)

            if isinstance(ir_func, str) and isinstance(disassembled_ir_func, str):
                similarity = get_normalized_edit_similarity(ir_func, disassembled_ir_func)
                edit_similarities.append(similarity)
                stats_by_type[test_type]['edit_similarities'].append(similarity)
                print(f"Edit similarity between original ir_func and LLM IR: {similarity:.4f}")
            else:
                print(f"Warning: ir_func or disassembled_ir_func is not a string for {task_id}_{test_type}. Skipping similarity calculation.")

        except Exception as e:
            print(f"LLM disassembler failed for {func_executable_path} with error: {str(e)}")
            traceback.print_exc()  
            continue


        # Create temporary .c file for ir_test
        with tempfile.NamedTemporaryFile(delete=False, suffix=".c", dir=temp_dir) as c_temp_file:
            c_temp_file_path = c_temp_file.name
            c_temp_file.write(ir_test.encode())
            print(f"Temporary .c file for ir_test generated: {c_temp_file_path}")

        # Compile disassembled .ll file to .o - ARM64 (aarch64) cross-compilation
        ll_object_file = disassembled_ll_file_path.replace('.ll', '.o')
        try:
            result_ll_compile = subprocess.run(
                ['clang', '--target=aarch64-linux-gnu', '-c', disassembled_ll_file_path, '-o', ll_object_file],
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT
            )

            if result_ll_compile.returncode != 0:
                print(f"Compilation error for disassembled .ll file ({disassembled_ll_file_path}): {result_ll_compile.stderr}")
                # Clean up if compilation fails
                if os.path.exists(c_temp_file_path): os.remove(c_temp_file_path)
                if os.path.exists(ll_object_file): os.remove(ll_object_file) # In case of partial write
                continue
            else:
                print(f"Compilation successful for disassembled .ll file: {ll_object_file}")
                successful_disassembly_compilations += 1
                stats_by_type[test_type]['successful_disassembly_compilations'] += 1
        except subprocess.TimeoutExpired:
            print(f"Compilation of disassembled .ll file ({disassembled_ll_file_path}) timed out after {COMMAND_TIMEOUT} seconds.")
            if os.path.exists(c_temp_file_path): os.remove(c_temp_file_path)
            if os.path.exists(ll_object_file): os.remove(ll_object_file)
            continue


        # Compile .c file for ir_test to .o - ARM64 (aarch64) cross-compilation
        c_object_file = c_temp_file_path.replace('.c', '.o')
        try:
            result_c_compile = subprocess.run(
                ['clang', f'--target={clang_target}', '-c', c_temp_file_path, '-o', c_object_file],
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT
            )

            if result_c_compile.returncode != 0:
                print(f"Compilation error for .c file ({c_temp_file_path}): {result_c_compile.stderr}")
                # Clean up if compilation fails
                if os.path.exists(c_temp_file_path): os.remove(c_temp_file_path)
                if os.path.exists(ll_object_file): os.remove(ll_object_file)
                if os.path.exists(c_object_file): os.remove(c_object_file) # In case of partial write
                continue
            else:
                print(f"Compilation successful for .c file: {c_object_file}")
        except subprocess.TimeoutExpired:
            print(f"Compilation of .c file ({c_temp_file_path}) timed out after {COMMAND_TIMEOUT} seconds.")
            if os.path.exists(c_temp_file_path): os.remove(c_temp_file_path)
            if os.path.exists(ll_object_file): os.remove(ll_object_file)
            if os.path.exists(c_object_file): os.remove(c_object_file)
            continue


        # Link the two .o files - ARM64 (aarch64) cross-compilation
        output_binary = os.path.join(os.getcwd(), f"linked_binary_{task_id}_{test_type}_arm")
        try:
            result_link = subprocess.run(
                ['clang', f'--target={clang_target}', '-static',
                 '-Wl,--allow-multiple-definition',
                 ll_object_file, c_object_file, '-o', output_binary, '-lm'],
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT
            )

            if result_link.returncode != 0:
                print(f"Linking error for {task_id}_{test_type}: {result_link.stderr}")
                print(f"Skipping execution test, but compilation was successful")
                # Clean up if linking fails
                if os.path.exists(output_binary): os.remove(output_binary) # In case of partial write
                # Continue to next iteration even if linking fails
                continue
            else:
                print(f"Linking successful! Executable: {output_binary}")

                # Execute the compiled binary - requires ARM64 environment or QEMU
                try:
                    # Note: You may need to use qemu-aarch64 or run on actual ARM64 hardware
                    # For cross-compiled ARM64 binaries on x86, use: ['qemu-aarch64', '-L', '/usr/aarch64-linux-gnu/', output_binary]
                    result_execute = subprocess.run(
                        [qemu_binary, '-L', qemu_library_path, output_binary],
                        capture_output=True, text=True, check=False, timeout=COMMAND_TIMEOUT
                    )

                    if result_execute.returncode != 0:
                        print(f"Execution error for {task_id}_{test_type}: {result_execute.stderr}")
                    else:
                        print(f"Execution successful for {task_id}_{test_type}! Output:\n{result_execute.stdout}")
                        successful_executions += 1
                        stats_by_type[test_type]['successful_executions'] += 1
                except FileNotFoundError:
                    print(f"Warning: qemu-aarch64 not found. Skipping execution for {output_binary}.")
                    print(f"To run ARM64 binaries on x86, install qemu-user: sudo apt-get install qemu-user-binfmt")
                except subprocess.TimeoutExpired:
                    print(f"Execution of {output_binary} timed out after {COMMAND_TIMEOUT} seconds.")
                finally:
                    if os.path.exists(output_binary): os.remove(output_binary) # Always remove the executable after testing

        except subprocess.TimeoutExpired:
            print(f"Linking for {task_id}_{test_type} timed out after {COMMAND_TIMEOUT} seconds.")
            if os.path.exists(output_binary): os.remove(output_binary) # Clean up if timeout during linking
        finally:
            # Clean up object files regardless of linking/execution success/failure
            if os.path.exists(c_temp_file_path): os.remove(c_temp_file_path)
            if os.path.exists(ll_object_file): os.remove(ll_object_file)
            if os.path.exists(c_object_file): os.remove(c_object_file)


    print(f"\n--- Summary (ARM64/aarch64 Architecture) ---")
    print(f"Total entries processed: {total_entries}")
    print(f"Number of successfully compiled disassembled .ll files: {successful_disassembly_compilations}")
    print(f"Number of successfully executed linked binaries: {successful_executions}")
    if edit_similarities:
        avg_edit_similarity = sum(edit_similarities) / len(edit_similarities)
        print(f"Average edit similarity between ir_func and disassembled .ll: {avg_edit_similarity:.4f}")
    else:
        print("No edit similarity calculated (no successful llvm-mctoll runs).")
    print("\n--- Per-Type Summary ---")
    for t, stats in stats_by_type.items():
        count = stats['count']
        succ_compile = stats['successful_disassembly_compilations']
        succ_exec = stats['successful_executions']
        sim_list = stats['edit_similarities']
        avg_sim = sum(sim_list) / len(sim_list) if sim_list else 0.0

        print(f"\nType: {t}")
        print(f"  Total entries: {count}")
        print(f"  Successfully compiled: {succ_compile}")
        print(f"  Successfully executed: {succ_exec}")
        print(f"  Average edit similarity: {avg_sim:.4f}")



