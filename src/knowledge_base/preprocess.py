#!/usr/bin/env python3
import os
import sys
import subprocess
import glob
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def remove_ir_attributes(ir_text):
    """
     LLVM IR  noinline、nounwind、optnone  uwtable ，
    。
    """
    attrs = ["noinline", "nounwind", "optnone", "uwtable"]
    pattern = r'\b(?:' + '|'.join(attrs) + r')\b'
    
    modified_text = re.sub(pattern, '', ir_text)
    
    modified_text = re.sub(
        r'attributes\s+#\d+\s*=\s*\{\s*\}\s*(\n|$)',
        '',
        modified_text
    )
    
    #modified_text = re.sub(r'\s+', ' ', modified_text)
    #modified_text = re.sub(r'\}\s+', '}\n', modified_text)
    
    return modified_text


def extract_probe_info(probe_text):
    """
    ，
    "Pseudo Probe Address Conversion results:" 
    "=======================================" 。
    
    :
      probe_text: 
      
    :
      ，
    """
    pattern = r'Pseudo Probe Address Conversion results:(.*?)======================================='
    match = re.search(pattern, probe_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""    



def clean_llvm_ir(llvm_ir):
    cleaned_ir = re.sub(r"^; ModuleID.*", "", llvm_ir, flags=re.MULTILINE)  # comment removed
    cleaned_ir = re.sub(r"^source_filename.*", "", cleaned_ir, flags=re.MULTILINE)
    cleaned_ir = re.sub(r"^target datalayout.*", "", cleaned_ir, flags=re.MULTILINE)
    cleaned_ir = re.sub(r"^target triple.*", "", cleaned_ir, flags=re.MULTILINE)
    cleaned_ir = re.sub(r"^attributes #\d+ = \{.*\}", "", cleaned_ir, flags=re.MULTILINE)
    cleaned_ir = re.sub(r"^;.*", "", cleaned_ir, flags=re.MULTILINE)
    #cleaned_ir = re.sub(r"!llvm\.module\.flags = !\{.*\}", "", cleaned_ir, flags=re.MULTILINE)
    #cleaned_ir = re.sub(r"!llvm\.ident = !\{.*\}", "", cleaned_ir, flags=re.MULTILINE)
    return cleaned_ir.strip()



# zeros_pattern = r"^0+\s"  
# def clean_asm(asm):
#     asm_clean = " "
#     asm = asm.split("Disassembly of section .text:")[-1].strip()
#     if "Disassembly of section .fini:" in asm:
#         asm = asm.split("Disassembly of section .fini:")[0].strip()
#     if '%' in asm.split("Disassembly of section .text.startup:")[0]:
#         asm = asm.replace("Disassembly of section .text.startup:","")
#     else:
#         asm = asm.split("Disassembly of section .text.startup:")[-1].strip()
#     asm = asm.replace("\n\n0000000000000000","")
#     for tmp in asm.split("\n"):
#         tmp_asm = re.sub(r'^\s*[\da-fA-F]+\s+(<[\w@.+-]+>):', r'\1:', tmp, flags=re.MULTILINE)
#         tmp_asm = re.sub(r'^\s*([\da-fA-F]+):\s+([\da-fA-F]{2}(?:\s[\da-fA-F]{2})*)\s+', r'\1:', tmp_asm)
#         #tmp_asm = re.sub(r'^\s*[\da-fA-F]+:\s*', '', tmp_asm, flags=re.MULTILINE)
#         tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
#         asm_clean += tmp_asm + "\n"

#     if len(asm_clean.split("\n")) < 4:
#         raise ValueError("compile fails")
#     asm = asm_clean
#     asm = re.sub(zeros_pattern, "", asm)
#     asm = re.sub(r'\n+', '\n', asm)
#     return asm


def clean_asm_arm(asm):
    asm_clean = ""
    asm = asm.split("Disassembly of section .text:")[-1].strip()
    
    if "Disassembly of section .fini:" in asm:
        asm = asm.split("Disassembly of section .fini:")[0].strip()
    if '%' in asm.split("Disassembly of section .text.startup:")[0]:
        asm = asm.replace("Disassembly of section .text.startup:","")
    else:
        asm = asm.split("Disassembly of section .text.startup:")[-1].strip()
    asm = asm.replace("\n\n0000000000000000","")
    
    for tmp in asm.split("\n"):
        tmp = tmp.strip()
        if not tmp:
            continue
            
        if re.match(r'^\s*[\da-fA-F]+\s+<[\w@.+-]+>:$', tmp):
            asm_clean += tmp + "\n"
            continue
        
        
        if ':' in tmp and not tmp.strip().endswith(':'):
            parts = tmp.split('\t')
            if len(parts) >= 3:
                addr_part = parts[0].strip()
                instruction_name = parts[1].strip()
                operands = parts[2].strip()
                
                addr_match = re.match(r'^([\da-fA-F]+):', addr_part)
                if addr_match:
                    addr = addr_match.group(1)
                    
                    operands = re.sub(r'\s+@.*$', '', operands)
                    
                    full_instruction = f"{instruction_name} {operands}".strip()
                    
                    asm_clean += f"{addr}:\t{full_instruction}\n"
            continue
        
        if tmp.startswith('.word') or tmp.startswith('.byte') or tmp.startswith('.ascii'):
            asm_clean += tmp + "\n"
    
    asm_clean = re.sub(r'\n+', '\n', asm_clean).strip()
    
    if len(asm_clean.split("\n")) < 2:
        return " "
        #raise ValueError("compile fails")
    
    return asm_clean



def process_c_file(c_file):
    """
     C ：
    1.  clang  LLVM IR ；
    2. ；
    3. 。
    """
    base = os.path.splitext(c_file)[0]  # comment removed
    ll_file = f"{base}_0_arm.ll"
    
    #cmd1 = ["clang", "-flto=thin", "-O0", "-fdebug-info-for-profiling", "-fpseudo-probe-for-profiling", "-S", "-emit-llvm", c_file, "-o", ll_file]
    cmd1 = ["clang", "--target=aarch64-linux-gnu", "-O0", "-S", "-emit-llvm", c_file, "-o", ll_file]
    print("：", " ".join(cmd1))
    try:
        subprocess.run(cmd1, check=True, timeout=300)  # comment removed
        with open(ll_file, "r") as f:
            sample_ir = f.read()
        modified_ir = remove_ir_attributes(sample_ir)
        cleaned_ir = clean_llvm_ir(modified_ir)
        with open(ll_file, "w") as f:
            f.write(cleaned_ir)
    except subprocess.TimeoutExpired as e:
        print(f": {e}")
    except subprocess.CalledProcessError as e:
        print(f": {e}")
    
    
    opt_type = ['0', '1', '2', '3']
    for opt in opt_type:
        exe_file = f"{base}_{opt}_arm"
        exe_output = f"{base}_{opt}_1_arm"
        obj_file = f"{base}_{opt}_arm.o"
        asm_file = f"{base}_{opt}_arm.asm"
        probe_file = f"{base}_{opt}_arm.txt"
        
        cmd2 = ["clang", "--target=aarch64-linux-gnu", "-c", c_file, "-o", obj_file, f"-O{opt}"]
        print("：", " ".join(cmd2))
        try:
            subprocess.run(cmd2, check=True, timeout=300)  # comment removed
        except subprocess.TimeoutExpired as e:
            print(f": {e}")
        except subprocess.CalledProcessError as e:
            print(f": {e}")

        # cmd5 = ["ld", obj_file, '-lm', '-lc', "-o", exe_file]
        # try:
        # except subprocess.TimeoutExpired as e:
        # except subprocess.CalledProcessError as e:

        #cmd2 = ["clang", ll_file, "-o", exe_file, f"-O{opt}"]
        #subprocess.run(cmd2, check=True)
        
        # cmd4 = ["llvm-bolt", "--print-pseudo-probes=all", "-update-debug-sections", exe_file, "-o", exe_output]
        # try:
        #     with open(probe_file, "w") as f:
        #     with open(probe_file, "r") as f:
        #         probe_text = f.read()
        #     extracted_info = extract_probe_info(probe_text)
        #     with open(probe_file, "w") as f:
        #         f.write(extracted_info)
        # except subprocess.TimeoutExpired as e:
        # except subprocess.CalledProcessError as e:



        cmd3 = ["llvm-objdump", "-d", obj_file]
        print("：", " ".join(cmd3), ">", asm_file)
        try:
            with open(asm_file, "w") as f:
                subprocess.run(cmd3, check=True, stdout=f, timeout=300)  # comment removed
            # with open(asm_file, "r") as f:
            #     asm_text = f.read()
            # asm_clean = clean_asm_arm(asm_text)
            # with open(asm_file, "w") as f:
            #     f.write(asm_clean)
        except subprocess.TimeoutExpired as e:
            print(f": {e}")
        except subprocess.CalledProcessError as e:
            print(f": {e}")
        

        
def find_all_c_files(root_dir):
    return [str(path) for path in Path(root_dir).rglob("*.c")]


def run_preprocess(input_path: str, jobs: int = 8):
    """Compile one C file or a directory of C files into LLVM IR/object/assembly artifacts."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isdir(input_path):
        c_files = find_all_c_files(input_path)
    else:
        c_files = [input_path]

    if not c_files:
        raise ValueError("No C files were found.")

    print(f"Discovered {len(c_files)} C file(s). Starting preprocessing...")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(process_c_file, c_file): c_file for c_file in c_files}
        for future in as_completed(futures):
            c_file = futures[future]
            try:
                future.result()
                print(f"[OK] {c_file}")
            except subprocess.CalledProcessError as e:
                print(f"[FAILED] {c_file}: {e}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Preprocess C files into LLVM IR and assembly artifacts.")
    parser.add_argument("input_path", help="Input C file or directory containing C files")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Number of worker threads")
    return parser
