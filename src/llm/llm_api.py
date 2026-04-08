import openai  # comment removed
import re
import os
from openai import APITimeoutError, APIError

def call_llm_disassembler(client, prompt):
    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",  # comment removed
            messages=[
                {"role": "system", "content": "You are an expert reverse engineer and compiler architect. Your task is to perform binary lifting and de-optimization. I will provide you with a function's assembly code, which was compiled with an aggressive optimization level (e.g., -O3), and several examples of similar assembly/unoptimized LLVM IR pairs."},
                {"role": "user", "content": prompt}
            ],
            stream=False,  # comment removed
            timeout=300  # comment removed
        )

    except APITimeoutError:
        print("⚠️ LLM  300 ，，")
        return "__TIMEOUT__"  # comment removed
    except APIError as e:
        print(f"⚠️ LLM API : {e}")
        return "__Failed__"

    response = []  # comment removed
    is_answering = False  # comment removed
    ir_block = stream.choices[0].message.content
    """
    for chunk in stream:
        if not getattr(chunk, 'choices', None):
            # choices，（token）
            continue

        delta = chunk.choices[0].delta

        # reasoning_content
        if not hasattr(delta, 'reasoning_content') and not hasattr(delta, 'content'):
            continue

        # 
        if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
            continue

        # 
        if not getattr(delta, 'reasoning_content', None) and not is_answering:
            is_answering = True

        # 
        #if getattr(delta, 'reasoning_content', None):
            #reasoning_content += delta.reasoning_content  # 

        # 
        if getattr(delta, 'content', None):
            response.append(delta.content)  # response
    """
    #ir_block = "".join(response)
    #print(ir_block)
    code_pattern = re.compile(r"```ir\n(.*?)\n```", re.DOTALL)
    match = code_pattern.search(ir_block)
    code_pattern1 = re.compile(r"```llvm\n(.*?)\n```", re.DOTALL)
    match1 = code_pattern1.search(ir_block)
    if match:
        return match.group(1).strip()
    elif match1:
        return match1.group(1).strip()
    else:
        return ir_block
