from __future__ import annotations

import re
from typing import Optional

from openai import OpenAI, APITimeoutError, APIError


def create_openai_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_llvm_code_block(response_text: str) -> str:
    code_pattern = re.compile(r"```ir\n(.*?)\n```", re.DOTALL)
    match = code_pattern.search(response_text)
    code_pattern_alt = re.compile(r"```llvm\n(.*?)\n```", re.DOTALL)
    match_alt = code_pattern_alt.search(response_text)
    if match:
        return match.group(1).strip()
    if match_alt:
        return match_alt.group(1).strip()
    return response_text.strip()


def call_llm_disassembler(
    client: OpenAI,
    prompt: str,
    model_name: str = "deepseek-chat",
    system_prompt: Optional[str] = None,
    timeout: int = 300,
) -> str:
    system_prompt = system_prompt or (
        "You are an expert reverse engineer and compiler architect. "
        "Your task is to perform binary lifting and de-optimization."
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            timeout=timeout,
        )
    except APITimeoutError:
        print("Warning: LLM request timed out.")
        return "__TIMEOUT__"
    except APIError as exc:
        print(f"Warning: LLM API error: {exc}")
        return "__FAILED__"

    response_text = response.choices[0].message.content or ""
    return extract_llvm_code_block(response_text)
