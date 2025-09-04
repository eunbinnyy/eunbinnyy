# -*- coding: utf-8 -*-
"""
v7 NUMERIC-SCHEMA: 숫자 전용(personas=N, share/sig/channel) JSON으로 오류 근본 차단
- 1차: response_format=json_schema (숫자만)
- 2차: tool function-call(schema)
- 3차: response_format=json_object
- 4차: LLM repair
- 실패시 템플릿 fallback
- personas 개수는 끝까지 N명 강제(pad/trim), share/channel 정규화
- 캠페인/바이럴 bump, 카테고리별 배율(cat_g), 소형 앙상블(amp 리스트 median)
"""

import os
import re
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI

from .prompts import build_persona_prompt_numeric   # 숫자 전용 프롬프트
from .llm_utils import request_with_retry           # preflight ping 용
from .anchors import anchor_for
from .feature_signals import multipliers_from_feature, soft_ramp

# =============== 공용 유틸 ===============
def log(msg: str): 
    print(msg, flush=True)

def _norm(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 1e-12, None)
    return a / a.sum()

def _try_parse_json(s: str) -> dict:
    s = s.strip().replace("```json", "```").replace("```", "")
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        s = s[a:b+1]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return json.loads(s)

def sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", str(name))
    return re.sub(r"\s+", "_", s).strip("_") or "file"

# =============== 곡선 & 페르소나 shaping ===============
def base_curve(feature: str, cat1: str, name: str,
               ramp=True, smooth=True, shrink=0.30) -> np.ndarray:
    m = multipliers_from_feature(feature, cat1, name)
    if ramp:
        m = _norm(m * soft_ramp(k=0.9, mid=2.5))
    if smooth:
        pad = 1
        xpad = np.pad(m, (pad, pad), mode="edge")
        m = np.convolve(xpad, np.ones(3) / 3, "valid")
        m = _norm(m)
    if shrink > 0:
        u = np.ones(12) / 12
        m = _norm((1 - shrink) * m + shrink * u)
    return m

def persona_signals_numeric(personas: list) -> dict:
    agg = {"promo": 0.0, "novelty": 0.0, "ad": 0.0, "season": 0.0}
    ssum = 0.0
    for p in personas or []:
        w = float(p.get("share", 0.0) or 0.0)
        ssum += w
        sig = p.get("sig", {}) or {}
        for k in agg:
            agg[k] += w * float(sig.get(k, 0.0) or 0.0)
    if ssum > 0:
        for k in agg:
            agg[k] = float(np.clip(agg[k] / ssum, 0.0, 1.0))
    return agg

def apply_persona_shape(m: np.ndarray, sig: dict, cat1: str, amp: float = 1.0) -> np.ndarray:
    adj = np.ones_like(m)
    early = (0.28 * sig.get("promo", 0) + 0.22 * sig.get("novelty", 0) + 0.18 * sig.get("ad", 0)) * amp
    adj[0] *= (1 + early)
    if len(adj) > 1:
        adj[1] *= (1 + 0.6 * early)
    hol_base = 0.06
    if "참치" in cat1:
        hol_base = 0.16
    elif "조미소스" in cat1:
        hol_base = 0.12
    elif "축산캔" in cat1:
        hol_base = 0.10
    hol = hol_base * (0.5 + 0.5 * sig.get("season", 0)) * amp
    if len(adj) >= 4:
        adj[2] *= (1 + hol)
        adj[3] *= (1 + 0.8 * hol)
    adj = np.clip(adj, 0.80, 1.35)
    out = _norm(m * adj)
    pad = 1
    xpad = np.pad(out, (pad, pad), mode="edge")
    out = np.convolve(xpad, np.ones(3) / 3, "valid")
    return _norm(out)

# =============== LLM preflight ===============
def _preflight_llm(model: str):
    key = os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY/api_key")
    resp = request_with_retry(
        model=model,
        system="You output JSON only.",
        user='{"ping":1}',
        force_json=True,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
    )
    _ = _try_parse_json(resp["choices"][0]["message"]["content"])

# =============== JSON 스키마/호출 루틴 ===============
def _numeric_schema(n: int) -> dict:
    ratio = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    persona = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "share", "sig", "channel"],
        "properties": {
            "id": {"type": "string", "pattern": r"^P\d{2}$"},
            "share": ratio,
            "sig": {
                "type": "object",
                "additionalProperties": False,
                "required": ["promo", "novelty", "ad", "season"],
                "properties": {"promo": ratio, "novelty": ratio, "ad": ratio, "season": ratio},
            },
            "channel": {"type": "array", "minItems": 5, "maxItems": 5, "items": ratio},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["personas"],
        "properties": {"personas": {"type": "array", "minItems": n, "maxItems": n, "items": persona}},
    }

def _ask_schema_first(model: str, system: str, user: str, schema: dict, max_tokens: int) -> dict:
    client = OpenAI()
    # 1) json_schema
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "PersonaBatch", "schema": schema, "strict": True}},
        )
        return _try_parse_json(r.choices[0].message.content)
    except Exception as e1:
        # 2) tool/function-call
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + "\n\nCall function emit_personas with JSON args."},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                tools=[{"type": "function", "function": {"name": "emit_personas", "description": "Return personas JSON.", "parameters": schema}}],
                tool_choice={"type": "function", "function": {"name": "emit_personas"}},
            )
            tc = r.choices[0].message.tool_calls
            if not tc:
                raise RuntimeError("No tool_calls")
            return _try_parse_json(tc[0].function.arguments)
        except Exception as e2:
            raise RuntimeError(f"schema+tool failed: {e1} | {e2}")

def _json_object_then_repair(model: str, system: str, user: str, max_tokens: int, target_n: int) -> dict:
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    txt = r.choices[0].message.content
    try:
        return _try_parse_json(txt)
    except Exception:
        # minimal repair 요청 (숫자 배열만 유지)
        fix_sys = "Return ONLY strict minified JSON."
        fix_user = (
            f"Fix into valid JSON with EXACTLY {target_n} personas: "
            '{"personas":[{"id":"P01","share":0.1,'
            '"sig":{"promo":0.5,"novelty":0.4,"ad":0.3,"season":0.2},'
            '"channel":[0.2,0.3,0.2,0.15,0.15]}]}'
        )
        r2 = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": fix_sys}, {"role": "user", "content": fix_user}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return _try_parse_json(r2.choices[0].message.content)

# =============== 페르소나 배열 가드 ===============
def _pad_or_trim_personas_numeric(personas: list, target_n: int) -> list:
    if not personas:
        return personas
    if len(personas) > target_n:
        personas = personas[:target_n]
    while len(personas) < target_n:
        base = personas[len(personas) % len(personas)]
        newp = json.loads(json.dumps(base, ensure_ascii=False))
        newp["id"] = f"P{len(personas) + 1:02d}"
        # 작은 지터
        newp["share"] = float(np.clip(float(newp.get("share", 0.0)) + np.random.uniform(-0.02, 0.02), 0.0, 1.0))
        sig = newp.get("sig", {}) or {}
        for k in ("promo", "novelty", "ad", "season"):
            sig[k] = float(np.clip(float(sig.get(k, 0.0)) + np.random.uniform(-0.03, 0.03), 0.0, 1.0))
        newp["sig"] = sig
        ch = newp.get("channel", [0.2, 0.3, 0.2, 0.15, 0.15])
        ch = np.array([float(x) for x in (ch if len(ch) == 5 else [0.2, 0.3, 0.2, 0.15, 0.15])], float)
        ch = _norm(np.clip(ch, 1e-9, None))
        newp["channel"] = [float(round(x, 4)) for x in ch]
        personas.append(newp)
    # 정규화
    s = sum(float(p.get("share", 0.0)) for p in personas) or 1.0
    for p in personas:
        p["share"] = float(round(float(p.get("share", 0.0)) / s, 4))
        ch = np.array([float(x) for x in p.get("channel", [0.2, 0.3, 0.2, 0.15, 0.15])], float)
        ch = _norm(np.clip(ch, 1e-9, None))
        p["channel"] = [float(round(x, 4)) for x in ch]
    return personas

# =============== 캠페인/바이럴 bump & 카테고리 보정 ===============
MONTH_RANGE_PAT = re.compile(r"(\d{1,2})\s*[-~]\s*(\d{1,2})\s*월")
MONTH_SINGLE_PAT = re.compile(r"(?<!\d)(\d{1,2})\s*월")

def _campaign_bump(feature_text: str, amp: float = 0.15) -> np.ndarray:
    """
    특징 문구에서 광고/바이럴 언급을 읽어 초기 1~3개월에 가중치 적용.
    - '광고'가 있으면 +amp, +0.7*amp, +0.5*amp
    - '광고 X' & '바이럴'이면 -0.5*amp, -0.35*amp
    """
    text = (feature_text or "").lower()
    bump = np.ones(12, dtype=float)

    has_ad = ("광고" in text) and not ("광고 x" in text or "광고x" in text)
    has_previral = ("바이럴" in text)

    if has_ad:
        bump[0] *= (1.0 + amp)
        if len(bump) > 1:
            bump[1] *= (1.0 + 0.7 * amp)
        if len(bump) > 2:
            bump[2] *= (1.0 + 0.5 * amp)

    if (("광고 x" in text) or ("광고x" in text)) and has_previral:
        bump[0] *= max(0.7, 1.0 - 0.5 * amp)
        if len(bump) > 1:
            bump[1] *= max(0.75, 1.0 - 0.35 * amp)

    return np.clip(bump, 0.6, 1.5)

def _parse_cat_g(s: str):
    d = {}
    if not s:
        return d
    for tok in s.split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                d[k.strip()] = float(v)
            except:
                pass
    return d

def _apply_cat_g(base_total: float, cat1: str, cat_g_map: dict) -> float:
    return base_total * float(cat_g_map.get(cat1, 1.0))

# =============== 메인 ===============
def main():
    ap = argparse.ArgumentParser(description="v7 NUMERIC-SCHEMA")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--n_personas", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=600)     # 짧게 유지
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--global_g", type=float, default=25.0)
    ap.add_argument("--use_persona_shape", action="store_true")
    ap.add_argument("--pshape_amp", type=float, default=1.2)
    ap.add_argument("--suffix", type=str, default="v7_numeric")
    ap.add_argument("--skip_preflight", action="store_true")
    ap.add_argument("--log_calls", action="store_true")
    ap.add_argument("--cat_g", type=str, default="", help="카테고리별 g 배율, 예: '우유류=0.92,조미소스=1.06,참치=1.10,축산캔=1.02'")
    ap.add_argument("--campaign_amp", type=float, default=0.15, help="캠페인/바이럴 신호의 초기월 증감 강도 (0~0.3 권장)")
    ap.add_argument("--ensemble_amps", type=str, default="", help="persona shape 앙상블용 amp 리스트. 예: '1.0,1.4'")
    args = ap.parse_args()

    log(f"[ARGS] {args}")

    artifacts = "artifacts"; os.makedirs(artifacts, exist_ok=True)
    out_dir = "submissions_v7"; os.makedirs(out_dir, exist_ok=True)

    if not args.skip_preflight:
        log("LLM preflight..."); _preflight_llm(args.model); log("✅ LLM preflight OK.")

    if not os.path.exists("product_info.csv"):
        raise FileNotFoundError("Missing product_info.csv")
    df = pd.read_csv("product_info.csv")
    req = {"product_name", "product_feature", "category_level_1", "category_level_2", "category_level_3"}
    if not req.issubset(df.columns):
        raise ValueError(f"product_info.csv must contain: {req}")
    log(f"Loaded product_info.csv rows={len(df)}")

    order = df["product_name"].tolist()
    if os.path.exists("sample_submission.csv"):
        try:
            ss = pd.read_csv("sample_submission.csv")
            if "product_name" in ss.columns:
                order = ss["product_name"].tolist()
                log("Use sample_submission order.")
        except Exception as e:
            log(f"[WARN] sample_submission read failed: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_log = os.path.join(artifacts, f"personas_v7_{ts}.jsonl")
    call_log_path = os.path.join(artifacts, f"llm_call_report_{ts}.csv")
    forecasts = {}; call_report = []

    # 카테고리 보정 맵 파싱
    cat_g_map = _parse_cat_g(args.cat_g)

    with tqdm(total=len(df), desc="Products", unit="prod") as pbar:
        for _, r in df.iterrows():
            prod = {
                "product_name": r["product_name"],
                "product_feature": r["product_feature"],
                "category_level_1": r["category_level_1"],
                "category_level_2": r["category_level_2"],
                "category_level_3": r["category_level_3"],
            }
            pname = prod["product_name"]; safe = sanitize_filename(pname)

            # 1) 프롬프트
            packed = json.loads(build_persona_prompt_numeric(prod, n_personas=int(args.n_personas)))
            system, user = packed["system"], packed["user"]

            # 2) 스키마 → 툴콜 → JSON_OBJECT → 리페어
            try:
                schema = _numeric_schema(int(args.n_personas))
                data = _ask_schema_first(args.model, system, user, schema, int(args.max_tokens))
                status = "LLM_OK"
            except Exception as e_schema:
                log(f"[WARN] {pname}: schema/tool failed → json_object. {repr(e_schema)}")
                try:
                    data = _json_object_then_repair(args.model, system, user, int(args.max_tokens), int(args.n_personas))
                    status = "REPAIRED"
                except Exception as e2:
                    log(f"[ERROR] {pname}: object+repair failed → fallback. {repr(e2)}")
                    # 템플릿 N명
                    n = int(args.n_personas or 6)
                    personas = []
                    for i in range(n):
                        personas.append({
                            "id": f"P{i + 1:02d}",
                            "share": 1.0 / n,
                            "sig": {"promo": 0.5, "novelty": 0.4, "ad": 0.3, "season": 0.3},
                            "channel": [0.22, 0.30, 0.22, 0.13, 0.13],
                        })
                    data = {"personas": personas}
                    status = "FALLBACK"

            # 3) N명 강제 + 정규화 + 로그
            personas = data.get("personas", [])
            personas = _pad_or_trim_personas_numeric(personas, int(args.n_personas))
            with open(persona_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"product": prod, "personas": personas}, ensure_ascii=False) + "\n")
            call_report.append({"product_name": pname, "status": status})
            log(f"[{status}] {pname} personas={len(personas)}")

            # 4) 기본 곡선
            base_m = base_curve(r["product_feature"], r["category_level_1"], r["product_name"])
            # 4-1) 캠페인/바이럴 반영
            cb = _campaign_bump(r["product_feature"], amp=float(args.campaign_amp))
            base_m = _norm(base_m * cb)

            # 5) (옵션) 페르소나 보정 앙상블
            variants = []
            ens_amps = []
            if args.ensemble_amps:
                try:
                    ens_amps = [float(x) for x in args.ensemble_amps.split(",") if x.strip()]
                except:
                    ens_amps = []

            if ens_amps:
                for ampv in ens_amps:
                    m = base_m.copy()
                    if args.use_persona_shape:
                        sig = persona_signals_numeric(personas)
                        m = apply_persona_shape(m, sig, r["category_level_1"], amp=ampv)
                    variants.append(m)
            else:
                m = base_m.copy()
                if args.use_persona_shape:
                    sig = persona_signals_numeric(personas)
                    m = apply_persona_shape(m, sig, r["category_level_1"], amp=float(args.pshape_amp))
                variants.append(m)

            # 6) 앵커 × 글로벌 g × 카테고리 보정
            base_total = anchor_for(r["category_level_1"], r["product_name"], r["product_feature"]) * float(args.global_g)
            total = _apply_cat_g(base_total, r["category_level_1"], cat_g_map)

            # 7) 변형별 산출 → median
            month_arrays = [(v * total).round().astype(int) for v in variants]
            if len(month_arrays) == 1:
                forecasts[pname] = month_arrays[0].tolist()
            else:
                stack = np.vstack(month_arrays)
                forecasts[pname] = np.median(stack, axis=0).round().astype(int).tolist()

            pbar.update(1)

    # 8) 제출 저장
    cols = ["product_name"] + [f"months_since_launch_{i}" for i in range(1, 13)]
    sub = pd.DataFrame([[nm] + forecasts.get(nm, [0] * 12) for nm in order], columns=cols)
    out_path = os.path.join("submissions_v7", f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.suffix}.csv")
    sub.to_csv(out_path, index=False, encoding="utf-8-sig")

    if call_report:
        pd.DataFrame(call_report).to_csv(call_log_path, index=False, encoding="utf-8-sig")
        log(f"📒 Call report → {call_log_path}")
    log(f"✅ Wrote {out_path}")

if __name__ == "__main__":
    main()
