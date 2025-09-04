# -*- coding: utf-8 -*-
import json
from typing import Dict
import json

_ATTRS = (
    "age_range, gender, household_size, life_stage, region, income_bracket, "
    "diet_health_orientation, cooking_frequency, brand_loyalty, price_sensitivity, "
    "promotion_responsiveness, channel_pref, social_influence, eco_consciousness, "
    "allergy_or_restriction, novelty_seeking, ad_model_affinity, seasonality_sensitivity"
)

def build_persona_prompt_labels(product: Dict, n_personas: int = 8) -> str:
    """싱글턴 프롬프트(라벨/속성/가중치만). JSON만 출력하도록 강제."""
    prod_name = (product.get("product_name") or "").strip()
    feat = (product.get("product_feature") or "").strip()
    cat1 = (product.get("category_level_1") or "").strip()
    cat2 = (product.get("category_level_2") or "").strip()
    cat3 = (product.get("category_level_3") or "").strip()

    system = "You are a careful market research assistant. Output ONLY a strict minified JSON object."

    user = f"""
Return ONE VALID JSON ONLY (minified). Create EXACTLY {n_personas} personas for the product.
Provide attributes (EXACTLY 10 keys) and attribute_weights (sum=1). Shares across personas should sum ≈1.00.

HARD RULES:
- Output MUST be raw JSON only (NO markdown fences, NO prose).
- Use only safe tokens for string values: lowercase letters [a-z], digits [0-9], hyphen(-), slash(/), plus(+).
  Examples: "25-44", "mixed", "2-3/wk", "mid", "core-health".
- Do NOT use quotes inside values. Do NOT use commas inside values. Do NOT use newlines.
- channel_pref must be an object with keys online, hypermarket, convenience, department, others; fractions sum to 1.00.
- EXCLUDE numeric behavior fields (purchase_probability, monthly_frequency, avg_units). LABELS ONLY.

RECOMMENDED (if relevant): include promotion_responsiveness, novelty_seeking, ad_model_affinity, seasonality_sensitivity among the 10 attributes.

PRODUCT:
- name: {prod_name}
- feature: {feat}
- category_level_1: {cat1}
- category_level_2: {cat2}
- category_level_3: {cat3}

SCHEMA (example shape; keep minified):
{{"personas":[{{"id":"P01","label":"core-health","share":0.1,
"attributes":{{"age_range":"25-44","gender":"mixed","household_size":"3","life_stage":"working",
"region":"metro","income_bracket":"mid","diet_health_orientation":"balanced","cooking_frequency":"2-3/wk",
"brand_loyalty":"medium","channel_pref":{{"online":0.25,"hypermarket":0.35,"convenience":0.2,"department":0.1,"others":0.1}}}},
"attribute_weights":{{"age_range":0.1,"gender":0.1,"household_size":0.1,"life_stage":0.1,"region":0.1,
"income_bracket":0.1,"diet_health_orientation":0.1,"cooking_frequency":0.1,"brand_loyalty":0.1,"channel_pref":0.1}}}}]}}
""".strip()

    return json.dumps({"system": system, "user": user}, ensure_ascii=False)


# --- add this to src/prompts.py ---

import json

def build_persona_prompt_numeric(prod: dict, n_personas: int = 6) -> str:
    """
    숫자 전용(personas=N, share/sig/channel만) JSON을 요구하는 싱글턴 프롬프트.
    문자열은 id만 허용(P01 형태). 나머지는 모두 number/array.
    """
    name = prod["product_name"]
    feature = prod["product_feature"]
    cat1 = prod["category_level_1"]; cat2 = prod["category_level_2"]; cat3 = prod["category_level_3"]

    system = (
        "You are a market research assistant that outputs only STRICT RFC8259 JSON. "
        "Do not include any prose, markdown fences, or explanations."
    )

    # 값에 한글/문장 안 씁니다. 모델이 문자열을 오염시키지 않게 숫자만 요구.
    user = f"""
Create EXACTLY {n_personas} consumer personas for the product below.
Return ONLY minified JSON matching this structure:

{{
  "personas":[
    {{"id":"P01","share":0.10,"sig":{{"promo":0.50,"novelty":0.40,"ad":0.30,"season":0.20}},"channel":[0.20,0.30,0.20,0.15,0.15]}},
    ...
  ]
}}

Rules:
- id must be "P01","P02",... zero-padded.
- share: number in [0,1]. Sum of shares across personas should be ~1.0 (±0.02).
- sig: object with numbers in [0,1] for keys promo, novelty, ad, season. (no other keys)
- channel: array of 5 numbers in [0,1] for [online, hypermarket, convenience, department, others], sum ~1.0 (±0.02).
- NO other keys. NO strings except id. NO commas/newlines/quotes inside values (values are numbers).
- Keep output short. Minified JSON only.

Context:
- product_name: {name}
- category: {cat1} > {cat2} > {cat3}
- features: {feature}
- Assume KR market. If features mention 광고(모델/기간) or 시즌성(추석 등), reflect that in sig and channel.
""".strip()

    return json.dumps({"system": system, "user": user}, ensure_ascii=False)
