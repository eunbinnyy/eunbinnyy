# -*- coding: utf-8 -*-
import re, numpy as np
MONTHS = 12
def _norm(a): a = np.clip(a, 1e-9, None); return a / a.sum()
def _def(): return np.ones(MONTHS, dtype=float)

def _decay_boost(start_m, end_m, base=1.12, tail=1.03):
    mul = _def(); span = max(1, end_m - start_m)
    for i in range(start_m, end_m+1):
        pos = i - start_m
        w = base - (base - tail) * (pos / span)
        mul[i-1] *= w
    return mul

def _shift_calendar_to_launch(cal_m):
    if cal_m <= 6: return 1, 2
    m = cal_m - 6; return m, m

def multipliers_from_feature(feature_text: str, cat1: str, name: str) -> np.ndarray:
    txt = f"{feature_text or ''} {name or ''}"
    mul = _def()

    for m1, m2 in re.findall(r"(\d{1,2})\s*[-~–]\s*(\d{1,2})\s*월", txt):
        a, b = int(m1), int(m2)
        sa, _ = _shift_calendar_to_launch(a); _, sb = _shift_calendar_to_launch(b)
        mul *= _decay_boost(sa, sb, base=1.15, tail=1.05)

    for m in re.findall(r"(?<![-~–])\b(\d{1,2})\s*월\b", txt):
        a = int(m); sa, sb = _shift_calendar_to_launch(a)
        mul *= _decay_boost(sa, sb, base=1.10, tail=1.03)

    if re.search(r"안유진|모델", txt):
        mul[0] *= 1.08; mul[1] *= 1.06; mul[2] *= 1.03
    if re.search(r"SNS|바이럴", txt, re.I):
        mul[0] *= 1.06; mul[1] *= 1.04

    if "참치" in cat1 or re.search(r"참치캔|축산캔|통조림", txt):
        mul[2] *= 1.03; mul[3] *= 1.03   # Sep, Oct
    if "조미" in cat1:
        for i in (2,3,6,7): mul[i] *= 1.02

    mul = np.clip(mul, 0.85, 1.20)
    return _norm(mul)

def soft_ramp(k=0.9, mid=2.5):
    m = np.arange(1, MONTHS+1, dtype=float)
    r = 1.0 / (1.0 + np.exp(-k*(m - mid)))
    return r / r.mean()
