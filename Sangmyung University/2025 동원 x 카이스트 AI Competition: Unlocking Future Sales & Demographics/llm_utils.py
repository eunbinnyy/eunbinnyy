# -*- coding: utf-8 -*-
import os
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from dotenv import load_dotenv

import time
class TransientOpenAIError(Exception):
    pass

# .env 자동 로드
load_dotenv()

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or api_key). Please set your environment variable.")
    return OpenAI(api_key=api_key)

@retry(
    reraise=True,
    stop=stop_after_attempt(int(os.getenv("RETRY_MAX", "5"))),
    wait=wait_exponential(multiplier=float(os.getenv("RETRY_BASE_DELAY", "1.0")), min=1, max=30),
    retry=retry_if_exception_type(TransientOpenAIError),
)
def request_with_retry(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 900,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    force_json: bool = False,
    max_retries: int = 4,
    base_delay: float = 1.0,
):
    client = OpenAI()  # OPENAI_API_KEY 사용

    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            if force_json:
                # ★ JSON 강제: 지원 모델에서 유효
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            return {"choices":[{"message":{"content": resp.choices[0].message.content}}]}
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    # 이 리턴에 올 일은 없음
    raise last_err

def repair_json_with_llm(bad_text: str, model: str, max_tokens: int = 1200) -> str:
    """
    깨진 JSON 텍스트를 LLM으로 '수리'해서 유효 JSON으로 반환.
    JSON만 출력하도록 강제.
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a strict JSON fixer. Output ONLY a valid JSON object. No comments, no markdown."},
            {"role": "user",
             "content": f"Fix the following into a single valid JSON object. Keep structure/keys/arrays intact as much as possible.\n\n{bad_text}"}
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content
