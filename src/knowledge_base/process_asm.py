import re

def extract_probe_details(probe_text):
    """
    ，，：
      -  Address、FUNC  Index ，
      - （ ".LBB06"）， "name" 。
      
    Address ： "0x" ，。
    
    :
      probe_text: ，。
    
    :
      ， "address", "func", "index", "name"。
    """
    details = []
    lines = probe_text.splitlines()
    
    first_line_pattern = re.compile(
        r"Address:\s*(0x[0-9a-fA-F]+)\s+FUNC:\s*([\w\.]+)\s+Index:\s*(\d+)"
    )
    second_line_pattern = re.compile(r"Probe is in\s+([^\s]+)")
    
    for i in range(0, len(lines), 2):
        probe_info = {}
        first_line = lines[i].strip()
        match1 = first_line_pattern.search(first_line)
        if match1:
            addr_hex = match1.group(1)  # comment removed
            addr_str = addr_hex[2:]
            addr_last4 = addr_str[-6:] if len(addr_str) >= 4 else addr_str
            func_name = match1.group(2)
            index = int(match1.group(3))
            probe_info["address"] = addr_last4
            probe_info["func"] = func_name
            probe_info["index"] = index
        else:
            continue
        
        block_name = ""
        if i+1 < len(lines):
            second_line = lines[i+1].strip()
            match2 = second_line_pattern.search(second_line)
            if match2:
                block_name = match2.group(1)
        probe_info["name"] = block_name
        
        details.append(probe_info)
    
    return details


def process_asm_with_probe_details(asm_text, probe_details):
    """
     probe ， ASM 。
    
    ：
      1.  ASM ， ASM ，
         （： "0000000000001140 <factorial>:"）。
      2. ， probe_details  func 。
      3.  probe  address（）， ASM 。
         ASM （）。
      4.  probe  address ，，
          index  probe  Index， probe  name。
      5. ，， ASM ，：
             { "name": , "index": probe  Index, "content":  ASM  }
    
    :
      asm_text: ， ASM 
      probe_details:  extract_probe_details ，，
                      "address", "func", "index", "name" ，
                      "address" 。
                     
    :
      ，， ASM ，：
             { "name": <>, "index": <probe index>, "content": <> }
    """
    func_pattern = re.compile(r'^\s*<([\w@.+-]+)>:', re.MULTILINE)
    functions = {}
    func_matches = list(func_pattern.finditer(asm_text))
    #print(func_matches)
    for i, match in enumerate(func_matches):
        func_name = match.group(1)
        start_idx = match.start()
        end_idx = func_matches[i+1].start() if i+1 < len(func_matches) else len(asm_text)
        func_content = asm_text[start_idx:end_idx]
        functions[func_name] = func_content

    probes_by_func = {}
    for pd in probe_details:
        probes_by_func.setdefault(pd["func"], []).append(pd)

    results = {}
    for func_name, asm_func in functions.items():
        if func_name not in probes_by_func:
            continue
        probe_map = {}
        for pd in probes_by_func[func_name]:
            addr = pd["address"]
            #probe_map[addr] = {"index": pd["index"], "name": pd["name"]}
            if addr not in probe_map:
                probe_map[addr] = {"name": pd["name"], "index": [pd["index"]]}
            else:
                if pd["index"] not in probe_map[addr]["index"]:
                    probe_map[addr]["index"].append(pd["index"])
        
        lines = asm_func.splitlines()
        blocks = []
        current_block_lines = []
        current_block_info = None  # comment removed

        for line in lines:
            stripped_line = line.lstrip()
            if len(stripped_line) >= 4:
                node_addr = stripped_line[:6]
                #print(node_addr)
                if node_addr in probe_map:
                    if current_block_info is not None:
                        block_content = "\n".join(current_block_lines)
                        blocks.append({
                            "name": current_block_info["name"],
                            "index": sorted(current_block_info["index"]),
                            "content": block_content
                        })
                    current_block_info = probe_map[node_addr]
                    current_block_lines = [line]  # comment removed
                    continue
            if current_block_info is not None:
                current_block_lines.append(line)
        if current_block_info is not None and current_block_lines:
            block_content = "\n".join(current_block_lines)
            blocks.append({
                "name": current_block_info["name"],
                "index": current_block_info["index"],
                "content": block_content
            })
        results[func_name] = blocks

    return results



def probe_asm(probe_file, asm_file):
    with open(probe_file, "r") as f:
        probe_text = f.read()
    probe_details = extract_probe_details(probe_text)
    #print(probe_details) 
    with open(asm_file, "r") as f:
        asm_text = f.read()
    asm_blocks = process_asm_with_probe_details(asm_text, probe_details) 
    return asm_blocks

