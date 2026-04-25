"""CFG-guided assembly partitioning utilities.

This module consolidates the earlier standalone scripts for CFG DOT generation,
DOT analysis, and assembly block grouping. It is used by the RAG lifter to
retrieve examples for each structural assembly block instead of retrieving only
once for the whole function.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import angr
import networkx as nx
from networkx.drawing.nx_agraph import read_dot, to_agraph


def _get_block_asm(block) -> str:
    lines = []
    for insn in block.capstone.insns:
        lines.append(f"{hex(insn.address)}: {insn.mnemonic} {insn.op_str}")
    return "\\l".join(lines) + "\\l"


def generate_cfg_dot(binary_path: str | Path, output_dot: str | Path) -> Path:
    """Generate a CFG DOT file for a binary using angr CFGFast.

    Args:
        binary_path: Path to the target binary.
        output_dot: Destination path for the generated DOT file.

    Returns:
        The generated DOT path.
    """
    binary_path = Path(binary_path)
    output_dot = Path(output_dot)
    output_dot.parent.mkdir(parents=True, exist_ok=True)

    project = angr.Project(str(binary_path), load_options={"auto_load_libs": False})
    cfg = project.analyses.CFGFast()
    graph = cfg.graph

    for node in graph.nodes():
        if hasattr(node, "block") and node.block is not None:
            graph.nodes[node]["asm"] = _get_block_asm(node.block)
            graph.nodes[node]["label"] = f"{hex(node.addr)}"
        else:
            graph.nodes[node]["asm"] = ""
            graph.nodes[node]["label"] = f"{hex(node.addr)}"

    agraph = to_agraph(graph)
    agraph.write(str(output_dot))
    return output_dot


def _load_cfg(dot_path: str | Path):
    graph = read_dot(str(dot_path))
    cfg = nx.DiGraph()
    exit_nodes = set()

    for node_name, attrs in graph.nodes(data=True):
        label = str(attrs.get("label", node_name)).strip('"')
        addr_match = re.search(r"0x[0-9a-fA-F]+", label)
        if not addr_match:
            continue

        addr = int(addr_match.group(0), 16)
        asm_code = str(attrs.get("asm", "")).lower()
        cfg.add_node(node_name, label=label, addr=addr, asm=asm_code)

        if "ret" in asm_code:
            exit_nodes.add(node_name)

    for src, dst in graph.edges():
        if src in cfg.nodes and dst in cfg.nodes:
            cfg.add_edge(src, dst)

    return cfg.nodes, cfg.edges(), exit_nodes


def _parse_addr(node: str) -> int:
    match = re.search(r"\+0x([0-9a-fA-F]+)", node)
    if match:
        return int(match.group(1), 16)
    match = re.search(r"0x([0-9a-fA-F]+)", node)
    return int(match.group(1), 16) if match else 0


def _format_addr(node: str, nodes) -> str:
    node_attrs = nodes[node] if node in nodes else {}
    if "addr" in node_attrs:
        return hex(int(node_attrs["addr"]))
    match = re.search(r"0x[0-9a-fA-F]+", str(node_attrs.get("label", node)))
    return match.group(0).lower() if match else "0x0"


def _build_graph(nodes: Iterable[str], edges: Iterable[tuple[str, str]]):
    graph = defaultdict(list)
    outdegree = defaultdict(int)
    for src, dst in edges:
        graph[src].append(dst)
        outdegree[src] += 1
    return graph, outdegree


def _merge_groups(groups: dict[str, set[str]], exit_nodes: set[str]) -> dict[str, set[str]]:
    def sort_group(group: set[str]) -> list[str]:
        return sorted(group, key=_parse_addr)

    merged = True
    while merged:
        merged = False
        new_groups = {}
        used = set()
        group_items = list(groups.items())

        for i, (root_i, nodes_i) in enumerate(group_items):
            if i in used:
                continue
            sorted_i = sort_group(nodes_i)
            merged_flag = False

            for j in range(i + 1, len(group_items)):
                if j in used:
                    continue
                root_j, nodes_j = group_items[j]
                sorted_j = sort_group(nodes_j)

                common_node = None
                for ni, nj in zip(reversed(sorted_i), reversed(sorted_j)):
                    if ni == nj:
                        common_node = ni
                    else:
                        break

                if common_node and common_node not in exit_nodes:
                    new_root = min(root_i, root_j, key=_parse_addr)
                    new_groups[new_root] = nodes_i | nodes_j
                    used.add(i)
                    used.add(j)
                    merged = True
                    merged_flag = True
                    break

            if not merged_flag:
                new_groups[root_i] = nodes_i
                used.add(i)

        groups = new_groups

    return groups


def _find_structural_groups(nodes, edges, exit_nodes: set[str]) -> dict[str, set[str]]:
    graph, outdegree = _build_graph(nodes, edges)
    branch_nodes = sorted([n for n in nodes if outdegree[n] >= 2], key=_parse_addr)
    groups = {}
    assigned = set()

    def dfs(cur: str, visited: set[str], current_root: str):
        if cur in visited:
            return
        visited.add(cur)
        cur_addr = _parse_addr(cur)

        if outdegree[cur] <= 1:
            groups[current_root].add(cur)
            assigned.add(cur)
            for nxt in graph[cur]:
                dfs(nxt, visited, current_root)
        else:
            successors = graph[cur]
            has_backedge = any(_parse_addr(nxt) <= cur_addr for nxt in successors)
            has_cycle_edge = any(cur in graph[nxt] for nxt in successors)
            if has_backedge or has_cycle_edge:
                groups[current_root].add(cur)
                assigned.add(cur)
                for nxt in successors:
                    dfs(nxt, visited, current_root)

    for root in branch_nodes:
        if root in assigned:
            continue
        groups[root] = {root}
        assigned.add(root)
        visited = set()
        for succ in graph[root]:
            dfs(succ, visited, root)

    return _merge_groups(groups, exit_nodes)


def new_partition(dot_file: str | Path) -> list[list[str]]:
    """Partition a CFG DOT file into structural address groups.

    Returns addresses as strings such as ``0x401000``.
    """
    nodes, edges, exit_nodes = _load_cfg(dot_file)
    groups = _find_structural_groups(nodes, edges, exit_nodes)
    result = []
    for _, group in groups.items():
        result.append([_format_addr(node, nodes) for node in sorted(group, key=_parse_addr)])
    return result


def _address_keys(address: str) -> set[str]:
    value = int(address, 16)
    return {hex(value).lower(), hex(value & 0xFFF).lower()}


def split_asm_by_address_groups(asm: str, address_groups: list[list[str]]) -> list[str]:
    """Split assembly text according to CFG-derived address groups.

    Address matching supports both full addresses and low-12-bit addresses to
    remain compatible with older DOT grouping scripts.
    """
    lines = asm.splitlines()
    block_map: dict[str, list[str]] = defaultdict(list)
    current_keys: set[str] | None = None
    address_pattern = re.compile(r"^\s*(?:0x)?([0-9a-fA-F]+):")

    for line in lines:
        match = address_pattern.match(line)
        if match:
            full_addr = hex(int(match.group(1), 16)).lower()
            current_keys = _address_keys(full_addr)
        if current_keys:
            for key in current_keys:
                block_map[key].append(line.rstrip("\n"))

    asm_blocks = []
    used_blocks = set()
    for group in address_groups:
        merged_lines = []
        for addr in group:
            for key in _address_keys(addr):
                if key in block_map and key not in used_blocks:
                    merged_lines.extend(block_map[key])
                    used_blocks.add(key)
                    break
        if merged_lines:
            asm_blocks.append("\n".join(merged_lines))

    return asm_blocks


def partition_asm_from_binary(
    asm: str,
    binary_path: str | Path,
    dot_path: str | Path,
) -> list[str]:
    """Generate CFG DOT from a binary and split assembly into structural blocks."""
    generate_cfg_dot(binary_path, dot_path)
    address_groups = new_partition(dot_path)
    return split_asm_by_address_groups(asm, address_groups)


def get_asm_blocks(
    asm: str,
    binary_path: str | Path | None = None,
    dot_path: str | Path | None = None,
    enable_partition: bool = True,
) -> list[str]:
    """Return structural assembly blocks, falling back to the whole function."""
    if not enable_partition or not binary_path or not dot_path or not Path(binary_path).exists():
        return [asm]

    try:
        blocks = partition_asm_from_binary(asm, binary_path, dot_path)
        return blocks or [asm]
    except Exception as exc:  # Keep lifting robust when CFG recovery fails.
        print(f"Warning: ASM partitioning failed for {binary_path}: {exc}")
        return [asm]
