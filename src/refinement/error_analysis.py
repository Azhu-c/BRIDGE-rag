from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Dict

# ---------------------------------------------------------------------------
# Built‑in verifier helper
# ---------------------------------------------------------------------------

def run_opt_verify(ir_text: str) -> List[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as tmp:
        tmp.write(ir_text)
        path = tmp.name
    try:
        proc = subprocess.run(["opt", "-passes=verify", "-disable-output", path], capture_output=True, text=True)
    except FileNotFoundError:
        return ["[Tool Missing] LLVM 'opt' not found – built‑in verifier skipped"]
    finally:
        os.unlink(path)
    if proc.returncode == 0 and not proc.stderr.strip():
        return []
    return [l.strip() for l in proc.stderr.splitlines() if l.strip()]

# ---------------------------------------------------------------------------
# Build variable → type map
# ---------------------------------------------------------------------------

def build_var_type_map(lines: List[str]) -> Dict[str, str]:
    var_types: Dict[str, str] = {}

    # — function parameters
    func_re = re.compile(r"^define\s+\w+\s+@\w+\s*\(([^)]*)\)")
    param_re = re.compile(r"(ptr|i\d+|float|double)\s+%([\w$.]+)")
    for line in lines:
        m = func_re.match(line)
        if m:
            for typ, name in param_re.findall(m.group(1)):
                var_types[name] = typ

    # — result definitions
    def_re = re.compile(r"^\s*%([\w$.]+)\s*=\s*([a-zA-Z][a-zA-Z0-9_]*)")
    cast_ops = {
        "trunc", "zext", "sext", "fptrunc", "fpext", "bitcast",
        "addrspacecast", "ptrtoint", "inttoptr"
    }

    for line in lines:
        m = def_re.match(line)
        if not m:
            continue
        var, op = m.groups()
        op = op.lower()
        if op == "alloca":
            var_types[var] = "ptr"  # alloca always yields a pointer
            continue
        if op in cast_ops:
            to_m = re.search(r"\bto\s+(ptr|i\d+|float|double)\b", line)
            if to_m:
                var_types[var] = to_m.group(1)
                continue
        # fallback heuristic: first scalar or ptr token
        typ_m = re.search(r"\b(i\d+|float|double|ptr)\b", line)
        if typ_m:
            var_types.setdefault(var, typ_m.group(1))
    return var_types

# ---------------------------------------------------------------------------
# Main heuristic checks
# ---------------------------------------------------------------------------

def collect_source_errors(ir_text: str) -> List[str]:
    lines = ir_text.splitlines()
    var_types = build_var_type_map(lines)
    errors: List[str] = []

    # 1. labels & branches
    label_re = re.compile(r"^([\w$.][\w$.]*)\s*:")
    labels: set[str] = set()
    for i, ln in enumerate(lines, 1):
        m = label_re.match(ln.strip())
        if m:
            if m.group(1) in labels:
                errors.append(f"[Duplicate Label] %{m.group(1)} re‑declared at line {i}")
            labels.add(m.group(1))

    br_re = re.compile(r"br\s+(?:i1\s+%?[\w$.]+,\s+label\s+%([\w$.]+),\s+label\s+%([\w$.]+)|label\s+%([\w$.]+))")
    for i, ln in enumerate(lines, 1):
        for dest in filter(None, sum(br_re.findall(ln), ())):
            if dest not in labels:
                errors.append(f"[Undefined Label] branch to %{dest} at line {i}")

    # 2. invalid call @label
    call_re = re.compile(r"call\s+[^@]*@([\w$.]+)\s*\(")
    for i, ln in enumerate(lines, 1):
        m = call_re.search(ln)
        if m and m.group(1) in labels:
            errors.append(f"[Invalid Call] call to basic block '@{m.group(1)}' at line {i}")

    # 3. SSA redefs & binop type mismatches
    seen: set[str] = set()
    for i, ln in enumerate(lines, 1):
        d = re.match(r"^\s*%([\w$.]+)\s*=", ln)
        if d:
            var = d.group(1)
            if var in seen:
                errors.append(f"[SSA Violation] %{var} redefined at line {i}")
            seen.add(var)

    binop_re = re.compile(r"\b(?:add|sub|mul|and|or|xor|sdiv|udiv|shl|lshr|ashr)\s+(i\d+)\s+%([\w$.]+)")
    for i, ln in enumerate(lines, 1):
        m = binop_re.search(ln)
        if m:
            expect, var = m.groups()
            actual = var_types.get(var)
            if actual and actual != expect:
                errors.append(f"[Type Mismatch] %{var} defined as {actual} but used as {expect} at line {i}")

    # 4. store dest ptr check
    store_re = re.compile(r"store\s+[^,]+,\s+([^,]+)")
    for i, ln in enumerate(lines, 1):
        m = store_re.search(ln)
        if m:
            dest = m.group(1).strip()
            if dest.startswith("ptr"):
                continue
            if dest.startswith("%") and var_types.get(dest[1:]) == "ptr":
                continue
            errors.append(f"[Store Dest Not Ptr] {dest} (line {i}) is not a pointer")

    # 5. load src ptr check (skip if src produced by alloca)
    load_re = re.compile(r"load\s+[^,]+,\s+ptr\s+%([\w$.]+)")
    for i, ln in enumerate(lines, 1):
        m = load_re.search(ln)
        if m:
            var = m.group(1)
            if var_types.get(var) == "ptr":
                continue
            errors.append(f"[Type Mismatch] load expects ptr but %{var} is {var_types.get(var, '<unknown>')} (line {i})")

    # 6. inline inttoptr
    if "inttoptr" in ir_text:
        inline_re = re.compile(r"inttoptr\s*\([^)]*\)")
        for i, ln in enumerate(lines, 1):
            if inline_re.search(ln):
                errors.append(f"[Inline inttoptr] avoid inline cast at line {i}; cast once and reuse a ptr value")

    return errors

