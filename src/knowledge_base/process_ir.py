import re

def process_llvm_ir_cfg(ir_file):
    with open(ir_file, "r", encoding='utf-8') as f:
        ir_text = f.read()

    functions = []
    
    func_pattern = re.compile(
        r'(define\s+.*?@(?P<fname>\w+)\s*\(.*?\)\s*[^{]*\{(?P<body>.*?^\})\s*)',
        re.DOTALL | re.MULTILINE
    )
    bb_pattern = re.compile(r'^\s*([\w\.\$]+):', re.MULTILINE)
    probe_pattern = re.compile(
        r'call\s+void\s+@llvm\.pseudoprobe\s*\(\s*i64\s*[-+]?\d+\s*,\s*i64\s*([-+]?\d+)',
        re.IGNORECASE
    )
    label_pattern = re.compile(r'label\s+%([^,\s]+)')

    for func_match in func_pattern.finditer(ir_text):
        func_name = func_match.group("fname")
        body = func_match.group("body")
        
        blocks = []
        block_matches = list(bb_pattern.finditer(body))
        
        if not block_matches:
            block_text = body.strip()
            
            probe_indexes = []
            next_labels = []  # comment removed
            filtered_lines = []
            for line in block_text.splitlines():
                line_str = line.strip()
                if line_str.startswith("br"):
                    found = label_pattern.findall(line)
                    next_labels.extend(found)
                m = probe_pattern.search(line)
                if m:
                    probe_indexes.append(int(m.group(1)))
                    continue
                filtered_lines.append(line)
            
            block_text_filtered = "\n".join(filtered_lines)
            blocks.append({
                "label": "entry",
                "content": block_text_filtered,
                "probe_indexes": probe_indexes,
                "next_label": next_labels  # comment removed
            })
        else:
            for i, match in enumerate(block_matches):
                label = match.group(1)
                start = match.start()
                end = block_matches[i+1].start() if i + 1 < len(block_matches) else len(body)
                block_text = body[start:end].strip()
                
                probe_indexes = []
                next_labels = []  # comment removed
                filtered_lines = []
                for line in block_text.splitlines():
                    line_str = line.strip()
                    if line_str.startswith("br"):
                        found = label_pattern.findall(line)
                        next_labels.extend(found)
                    m = probe_pattern.search(line)
                    if m:
                        probe_indexes.append(int(m.group(1)))
                        continue
                    filtered_lines.append(line)
                
                block_text_filtered = "\n".join(filtered_lines)
                blocks.append({
                    "label": label,
                    "content": block_text_filtered,
                    "probe_indexes": probe_indexes,
                    "next_label": next_labels  # comment removed
                })
        
        functions.append({
            "name": func_name,
            "blocks": blocks
        })
    
    return functions

"""
# 
if __name__ == "__main__":
    #with open("/data2/zxa/100c_files/s_75_0.ll", "r") as f:
        #ir_text = f.read()
    
    funcs = process_llvm_ir_cfg("/data2/zxa/100c_files/s_52_0.ll")
    print(funcs)

    for func in funcs:
        print(":", func["name"])
        for block in func["blocks"]:
            print("  :", block["label"])
            print("    :", block["probe_indexes"])
            print("    :")
            print(block["content"])
            print("-" * 40)
"""
