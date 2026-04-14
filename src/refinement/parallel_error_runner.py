from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import editdistance


def normalize_spaces(s: str) -> str:
    if not s:
        s = ""
    s = re.sub(r"([.,!?();+\-/*{}^&=!])", r" \1 ", s)
    s = " ".join(s.split())
    return s


def normalize_variables(s: str) -> str:
    return re.sub(r"%[\w\.\-]+", "%VAR", s)


def preprocess(s: str) -> str:
    return normalize_variables(normalize_spaces(s))


def get_normalized_edit_similarity(ground_truth: str, pred: str) -> float:
    if not pred:
        return 0.0
    gt = preprocess(ground_truth)
    out = preprocess(pred)
    ed = editdistance.distance(gt, out)
    denom = max(len(gt), len(out))
    if denom == 0:
        return 1.0
    return 1 - ed / denom


def try_run_ll_file(filename: str, ir_test: str | None = None, ir_func: str | None = None, temp_dir: str = "/tmp"):
    os.makedirs(temp_dir, exist_ok=True)
    ll_object_file = filename.replace(".ll", ".o")
    compile_success = False
    exec_success = False
    similarity = -1.0

    if ir_func:
        try:
            with open(filename, "r") as f:
                gen_code = f.read()
            similarity = get_normalized_edit_similarity(ir_func, gen_code)
        except Exception:
            similarity = -1.0

    c_temp_file_path = None
    c_object_file = None
    output_binary = None

    try:
        result_compile = subprocess.run(["clang", "-c", filename, "-o", ll_object_file, "-lm", "-lc"], capture_output=True, text=True, timeout=15)
        if result_compile.returncode != 0:
            return False, False, result_compile.stderr, similarity
        compile_success = True

        if ir_test:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".c", dir=temp_dir) as c_temp_file:
                c_temp_file_path = c_temp_file.name
                c_temp_file.write(ir_test.encode())

            c_object_file = c_temp_file_path.replace(".c", ".o")
            result_c = subprocess.run(["clang", "-c", c_temp_file_path, "-o", c_object_file, "-lm", "-lc"], capture_output=True, text=True, timeout=15)
            if result_c.returncode != 0:
                return True, False, result_c.stderr, similarity

            pid = os.getpid()
            output_binary = filename.replace(".ll", f"_{pid}_exe")
            result_link = subprocess.run(["clang", ll_object_file, c_object_file, "-lm", "-lc", "-o", output_binary], capture_output=True, text=True, timeout=15)
            if result_link.returncode != 0:
                return True, False, result_link.stderr, similarity

            result_exec = subprocess.run([output_binary], capture_output=True, text=True, timeout=15)
            if result_exec.returncode != 0:
                return True, False, result_exec.stderr, similarity
            exec_success = True

        return compile_success, exec_success, "", similarity
    except subprocess.TimeoutExpired:
        return False, False, "TimeoutExpired", similarity
    finally:
        for path in [ll_object_file, output_binary, c_temp_file_path, c_object_file]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


def _run_one_case(args):
    task_id, opt, ir_test, ir_func, filename, temp_dir = args
    if not os.path.exists(filename):
        return (opt, False, False, -1.0, "File not found", filename, task_id)
    compile_success, exec_success, error_msg, similarity = try_run_ll_file(filename, ir_test=ir_test, ir_func=ir_func, temp_dir=temp_dir)
    return (opt, compile_success, exec_success, similarity, error_msg, filename, task_id)


def run_parallel_evaluation(
    json_path: str,
    input_ll_dir: str,
    file_pattern: str = "onlyfunc_{task_id}_{opt}_rag.ll",
    max_workers: int | None = None,
    temp_dir: str = "/tmp",
):
    with open(json_path, "r") as f:
        testsets = json.load(f)

    edit_sim_dict = defaultdict(list)
    edit_sim_all = []
    success_counter_compile = defaultdict(int)
    success_counter_exec = defaultdict(int)

    tasks = []
    for testset in testsets:
        task_id = testset["task_id"]
        opt = testset["type"]
        ir_test = testset["ir_test"]
        ir_func = testset["ir_func"]
        filename = testset.get("output_ll_path") or os.path.join(input_ll_dir, file_pattern.format(task_id=task_id, opt=opt))
        tasks.append((task_id, opt, ir_test, ir_func, filename, temp_dir))

    cpu = os.cpu_count() or 8
    resolved_workers = max_workers or max(1, min(cpu - 1, 16))
    print(f"Total tasks: {len(tasks)} | max_workers: {resolved_workers}")

    with ProcessPoolExecutor(max_workers=resolved_workers) as ex:
        futures = [ex.submit(_run_one_case, task) for task in tasks]
        for fut in as_completed(futures):
            opt, compile_success, exec_success, similarity, err, filename, task_id = fut.result()
            if compile_success:
                success_counter_compile[opt] += 1
            else:
                print(f"[{opt}] COMPILE FAIL | id={task_id} | {filename}\n{err}\n")
            if exec_success:
                success_counter_exec[opt] += 1
            if similarity is not None and similarity >= 0:
                edit_sim_dict[opt].append(similarity)
                edit_sim_all.append(similarity)

    print("\n=== Compile & Executable Summary ===")
    for opt in ["O0", "O1", "O2", "O3"]:
        compile_count = success_counter_compile[opt]
        exec_count = success_counter_exec[opt]
        sims = edit_sim_dict[opt]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        print(f"count:{len(sims)} {opt}: Compile Success = {compile_count}, Executable Success = {exec_count}, Avg Similarity = {avg_sim:.4f}")

    total_compile = sum(success_counter_compile.values())
    total_exec = sum(success_counter_exec.values())
    avg_all = sum(edit_sim_all) / len(edit_sim_all) if edit_sim_all else 0.0
    print(f"\nTotal: Compile Success = {total_compile}, Executable Success = {total_exec}, Avg Similarity = {avg_all:.4f}")
