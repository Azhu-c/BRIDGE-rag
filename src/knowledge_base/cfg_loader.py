from collections import defaultdict

def build_cfg_from_nodes(nodes):
    cfg = {}
    label_to_probe = {}
    for node in nodes:
        label = node['label']
        cfg[label] = node.get('next_label', [])
        label_to_probe[label] = node.get('probe_indexes', [])
    return cfg, label_to_probe

def build_reverse_cfg(cfg):
    reverse_cfg = defaultdict(list)
    for src, dests in cfg.items():
        for dest in dests:
            reverse_cfg[dest].append(src)
    return reverse_cfg

def find_back_edges(cfg, start='entry'):
    visited = set()
    stack = []
    back_edges = []

    def dfs(node):
        if node in visited:
            return
        print(f"Visiting: {node}")  # comment removed
        visited.add(node)
        stack.append(node)
        for succ in cfg.get(node, []):
            if succ in stack:
                back_edges.append((node, succ))
            elif succ not in visited:
                dfs(succ)
        stack.pop()

    dfs(start)
    return back_edges

def compute_natural_loop(cfg, head, tail):
    reverse_cfg = build_reverse_cfg(cfg)
    loop_nodes = set()
    worklist = [tail]
    loop_nodes.add(head)
    while worklist:
        node = worklist.pop()
        if node not in loop_nodes:
            loop_nodes.add(node)
            worklist.extend(reverse_cfg[node])
    return loop_nodes

def find_structure_exit(cfg, body):
    exits = set()
    for node in body:
        for succ in cfg.get(node, []):
            if succ not in body:
                exits.add(succ)
    return exits

def identify_unique_structures(cfg, label_to_probe, start='entry'):
    structures = []
    seen = set()

    back_edges = find_back_edges(cfg, start)

    # 1. do-while / while
    for tail, head in back_edges:
        loop_nodes = compute_natural_loop(cfg, head, tail)
        exits = find_structure_exit(cfg, loop_nodes)
        structure_nodes = frozenset(loop_nodes.union(exits))
        if structure_nodes not in seen:
            seen.add(structure_nodes)
            structures.append({
                'type': 'do-while' if tail in cfg.get(head, []) else 'while',
                'nodes': [
                    {'label': n, 'probe_indexes': label_to_probe.get(n, [])}
                    for n in structure_nodes
                ]
            })

    # 2. if-else
    for node, succs in cfg.items():
        if len(succs) == 2:
            then_branch, else_branch = succs
            then_succs = set(cfg.get(then_branch, []))
            else_succs = set(cfg.get(else_branch, []))
            join = then_succs & else_succs
            if join:
                structure_nodes = frozenset({node, then_branch, else_branch}.union(join))
                if structure_nodes not in seen:
                    seen.add(structure_nodes)
                    structures.append({
                        'type': 'if-else',
                        'nodes': [
                            {'label': n, 'probe_indexes': label_to_probe.get(n, [])}
                            for n in structure_nodes
                        ]
                    })

    # 3. switch-case
    for node, succs in cfg.items():
        if len(succs) >= 3:
            for potential_exit in succs:
                branches = [s for s in succs if s != potential_exit]
                if branches and all(potential_exit in cfg.get(b, []) for b in branches):
                    structure_nodes = frozenset({node}.union(branches).union({potential_exit}))
                    if structure_nodes not in seen:
                        seen.add(structure_nodes)
                        structures.append({
                            'type': 'switch',
                            'nodes': [
                                {'label': n, 'probe_indexes': label_to_probe.get(n, [])}
                                for n in structure_nodes
                            ]
                        })

    return structures


def re_probes(nodes):
    cfg, label_to_probe = build_cfg_from_nodes(nodes)
    if len(cfg) > 1:
        structures = identify_unique_structures(cfg, label_to_probe)
        re_probes = []
        for s in structures:
            labels = [node['label'] for node in s['nodes']]
            probe_indexes = []
            for node in s['nodes']:
                probe_indexes.extend(node['probe_indexes'])
            re_probes.append(probe_indexes)
            #print(" ".join(labels))
            #print(" ".join(map(str, probe_indexes)))
        return re_probes

    else:
        result = [list(label_to_probe.values())[0]]
        return result

"""
# 
nodes =  [{'label': 'entry', 'content': 'entry:\n  %i = alloca i32, align 4\n  store i32 0, ptr %i, align 4\n  br label %start_loop', 'probe_indexes': [1], 'next_label': ['start_loop']}, {'label': 'start_loop', 'content': 'start_loop:                                       ; preds = %if.end, %if.then2, %entry\n  %0 = load i32, ptr %i, align 4\n  %cmp = icmp slt i32 %0, 5\n  br i1 %cmp, label %if.then, label %if.end4', 'probe_indexes': [2], 'next_label': ['if.then', 'if.end4']}, {'label': 'if.then', 'content': 'if.then:                                          ; preds = %start_loop\n  %1 = load i32, ptr %i, align 4\n  %cmp1 = icmp eq i32 %1, 3\n  br i1 %cmp1, label %if.then2, label %if.end', 'probe_indexes': [3], 'next_label': ['if.then2', 'if.end']}, {'label': 'if.then2', 'content': 'if.then2:                                         ; preds = %if.then\n  %2 = load i32, ptr %i, align 4\n  %inc = add nsw i32 %2, 1\n  store i32 %inc, ptr %i, align 4\n  br label %start_loop', 'probe_indexes': [4], 'next_label': ['start_loop']}, {'label': 'if.end', 'content': 'if.end:                                           ; preds = %if.then\n  %3 = load i32, ptr %i, align 4\n  %call = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %3)\n  %4 = load i32, ptr %i, align 4\n  %inc3 = add nsw i32 %4, 1\n  store i32 %inc3, ptr %i, align 4\n  br label %start_loop', 'probe_indexes': [5], 'next_label': ['start_loop']}, {'label': 'if.end4', 'content': 'if.end4:                                          ; preds = %start_loop\n  ret void\n}', 'probe_indexes': [7], 'next_label': []}]

re = re_probes(nodes)
print(re)
"""
