# -*- coding: utf-8 -*-
import re

BASE_ANCHORS = {
    "우유류":   60000,
    "참치":    120000,
    "조미소스": 80000,
    "축산캔":   70000,
    "커피":     90000,
    "기타":     80000,
}

NAME_MULTIPLIERS = [
    (r"(그릭|Greek|프로틴|단백질|제로)", 1.08),
    (r"(프리미엄|고급)", 0.92),
    (r"(900g|340g|대용량|중대용량)", 1.05),
    (r"(90g|200g|소용량)", 0.95),
]

def anchor_for(cat1: str, name: str, feature: str = "") -> float:
    base = BASE_ANCHORS.get(cat1, BASE_ANCHORS["기타"])
    txt = f"{name or ''} {feature or ''}"
    mult = 1.0
    for pat, w in NAME_MULTIPLIERS:
        if re.search(pat, txt, re.I): mult *= float(w)
    return float(base) * mult
