import subprocess
import os
import re
import json
from collections import defaultdict
import tempfile
import editdistance
from concurrent.futures import ProcessPoolExecutor, as_completed


# -----------------------------
# Edit similarity helpers
# -----------------------------
def normalize_spaces(s: str) -> str:
    if not s:
        s = ""
    s = re.sub(r"([.,!?();+\-/*{}^&=!])", r" \1 ", s)  # comment removed
    s = " ".join(s.split())
    return s


def normalize_variables(s: str) -> str:
    s = re.sub(r"%[\w\.\-]+", "%VAR", s)
    return s


def preprocess(s: str) -> str:
    s = normalize_spaces(s)
    s = normalize_variables(s)
    return s


def get_normalized_edit_similarity(ground_truth: str, pred: str) -> float:
    if not pred:
        return 0.0
    gt = preprocess(ground_truth)
    out = preprocess(pred)
    ed = editdistance.distance(gt, out)
    denom = max(len(gt), len(out))
    if denom == 0:
        return 1.0
    normalized = ed / denom
    return 1 - normalized


# -----------------------------
# Core: compile/link/execute one .ll
# -----------------------------
def try_run_ll_file(filename: str, ir_test: str = None, ir_func: str = None):
    ll_object_file = filename.replace(".ll", ".o")
    compile_success = False
    exec_success = False
    similarity = -1.0

    # similarity: compare IR text
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
        # 1) compile ll -> o
        result_compile = subprocess.run(
            ["clang", "-c", filename, "-o", ll_object_file, "-lm", "-lc"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result_compile.returncode != 0:
            return False, False, result_compile.stderr, similarity
        compile_success = True

        # 2) compile test C -> o, link, exec
        if ir_test:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".c", dir="/data2/zxa/tmp"
            ) as c_temp_file:
                c_temp_file_path = c_temp_file.name
                c_temp_file.write(ir_test.encode())

            c_object_file = c_temp_file_path.replace(".c", ".o")
            result_c = subprocess.run(
                ["clang", "-c", c_temp_file_path, "-o", c_object_file, "-lm", "-lc"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result_c.returncode != 0:
                return True, False, result_c.stderr, similarity

            pid = os.getpid()
            output_binary = filename.replace(".ll", f"_{pid}_exe")

            result_link = subprocess.run(
                ["clang", ll_object_file, c_object_file, "-lm", "-lc", "-o", output_binary],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result_link.returncode != 0:
                return True, False, result_link.stderr, similarity

            result_exec = subprocess.run(
                [output_binary], capture_output=True, text=True, timeout=15
            )
            if result_exec.returncode != 0:
                return True, False, result_exec.stderr, similarity

            exec_success = True

        return compile_success, exec_success, "", similarity

    except subprocess.TimeoutExpired:
        return False, False, "TimeoutExpired", similarity

    finally:
        try:
            if ll_object_file and os.path.exists(ll_object_file):
                os.remove(ll_object_file)
        except Exception:
            pass

        try:
            if output_binary and os.path.exists(output_binary):
                os.remove(output_binary)
        except Exception:
            pass

        try:
            if c_temp_file_path and os.path.exists(c_temp_file_path):
                os.remove(c_temp_file_path)
        except Exception:
            pass

        try:
            if c_object_file and os.path.exists(c_object_file):
                os.remove(c_object_file)
        except Exception:
            pass


# -----------------------------
# Worker (must be top-level for ProcessPool)
# -----------------------------
def _run_one_case(args):
    """
    ：///
    :
      (opt, compile_success, exec_success, similarity, error_msg, filename, task_id)
    """
    task_id, opt, ir_test, ir_func, filename = args

    if not os.path.exists(filename):
        return (opt, False, False, -1.0, "File not found", filename, task_id)

    compile_success, exec_success, error_msg, similarity = try_run_ll_file(
        filename, ir_test=ir_test, ir_func=ir_func
    )
    return (opt, compile_success, exec_success, similarity, error_msg, filename, task_id)


# -----------------------------
# Main (parallel)
# -----------------------------
def main():
    json_path = "/data2/zxa/rag_final_dataset/new_eval_code/new-eval-c-500-0919.json"
    with open(json_path, "r") as f:
        testsets = json.load(f)

    edit_sim_dict = defaultdict(list)  # comment removed
    edit_sim_all = []  # comment removed
    success_counter_compile = defaultdict(int)
    success_counter_exec = defaultdict(int)

    tasks = []
    for testset in testsets:
        i = testset["task_id"]
        opt = testset["type"]
        ir_test = testset["ir_test"]
        ir_func = testset["ir_func"]

        filename = f"/data2/zxa/rag_final_dataset/rag-data/geimini3-rag-mbpp-x86/onlyfunc_{i}_{opt}_rag.ll"
        tasks.append((i, opt, ir_test, ir_func, filename))

    cpu = os.cpu_count() or 8
    max_workers = max(1, min(cpu - 1, 16))

    print(f"Total tasks: {len(tasks)} | max_workers: {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_one_case, t) for t in tasks]

        for fut in as_completed(futures):
            opt, compile_success, exec_success, similarity, err, filename, task_id = fut.result()

            if compile_success:
                success_counter_compile[opt] += 1
            else:
                print(f"[{opt}] COMPILE FAIL | id={task_id} | {filename}\n{err}\n")

            if exec_success:
                success_counter_exec[opt] += 1
            elif compile_success and (err and err != ""):
                pass

            if similarity is not None and similarity >= 0:
                edit_sim_dict[opt].append(similarity)
                edit_sim_all.append(similarity)

    print("\n=== Compile & Executable Summary ===")
    for opt in ["O0", "O1", "O2", "O3"]:
        compile_count = success_counter_compile[opt]
        exec_count = success_counter_exec[opt]
        sims = edit_sim_dict[opt]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        print(
            f"count:{len(sims)}"
            f"{opt}: Compile Success = {compile_count}, "
            f"Executable Success = {exec_count}, "
            f"Avg Similarity = {avg_sim:.4f}"
        )

    total_compile = sum(success_counter_compile.values())
    total_exec = sum(success_counter_exec.values())
    avg_all = sum(edit_sim_all) / len(edit_sim_all) if edit_sim_all else 0.0
    print(
        f"\nTotal: Compile Success = {total_compile}, "
        f"Executable Success = {total_exec}, "
        f"Avg Similarity = {avg_all:.4f}"
    )


if __name__ == "__main__":
    main()
