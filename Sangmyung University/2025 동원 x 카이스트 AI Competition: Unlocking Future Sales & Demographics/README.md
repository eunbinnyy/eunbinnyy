
# 📈 2025 동원 x 카이스트 AI Competition: Unlocking Future Sales & Demographics
**소비자 페르소나 기반 신제품 월별 수요 예측 (24.07 ~ 25.06)**

본 리포지토리는 LLM을 활용해 **숫자 전용(single-turn) 페르소나**를 생성하고,  
**계절성·프로모션·바이럴 신호**를 반영한 **월별 수요 곡선** 위에 투영하여  
출시 후 12개월 판매량을 예측하는 파이프라인 구현체입니다.

- 주최: **동원그룹 × KAIST**
- 주제: **소비자 페르소나 기반 신제품 월별 수요 예측**
- 핵심 아이디어:  
  1) LLM으로 **형식 보장된(minified, RFC8259) 숫자 JSON 페르소나** 생성  
  2) 제품 특징/카테고리에서 **월별 분포 곡선(12차원)** 생성  
  3) **광고/바이럴·시즌성 키워드** → 초기/시즌 **bump** 적용  
  4) **카테고리별 앵커(총량)** × **글로벌 스케일** 보정  
  5) **소형 앙상블(median)**로 강건성 확보  
- 제출 형식: `sample_submission.csv` 스키마(12개 월 컬럼)

---

## 🗂️ 데이터 (대회 제공 예시)
product_image/ # 제품 이미지 예시 (9개)
product_info.csv # [product_name, product_feature, category_level_1/2/3]
sample_submission.csv # [product_name, months_since_launch_1..12]
> `months_since_launch_k`의 “launch”는 **출시 시점이 아니라 예측 시작 기준 월(24년 7월)**을 의미합니다.

---
## 🧠 방법 개요

### 1) LLM 기반 **숫자 전용(personas=N) 페르소나**
- 싱글턴 프롬프트로 `N`명의 페르소나를 **한 번에** 생성  
- 출력 스키마(예시, 실제는 `prompts.build_persona_prompt_numeric`가 생성):
{"personas":[
  {"id":"P01","share":0.10,
   "sig":{"promo":0.50,"novelty":0.40,"ad":0.30,"season":0.20},
   "channel":[0.20,0.30,0.20,0.15,0.15]}
]}
- 강제 JSON 전략 (깨짐 방지, main.py):
  - response_format=json_schema
  - tool/function-call(schema)
  - response_format=json_object
  - LLM repair
  - 템플릿 fallback
- 생성된 personas는 share 합 1.0으로 정규화, channel 합도 1.0로 정규화

### 2) 월별 수요 곡선(Base Curve)
- feature_signals.multipliers_from_feature(feature, cat1, name)
  - 제품 특징문(예: “9~10월”, “모델/SNS/바이럴”, 카테고리별 시즌성 등)에서 월별 가중치 도출
  - 스무딩, 균질화(shrink)로 과도한 피크 방지
- soft_ramp(k, mid)로 출시 초반 램프업 형상을 더함

### 3) 캠페인/바이럴 bump (초기 1~3개월)
- 특징문에서 "광고", "SNS/바이럴" 신호를 파싱 → 초기 가중치 증감
- "광고 X" 등 부정 신호는 과대 추정 방지

### 4) 페르소나 shape (옵션)
- --use_persona_shape 시, 페르소나들의 sig(promo/novelty/ad/season)를 가중합 → 초기/시즌 민감도 보정(apply_persona_shape)

### 5) 규모 앵커 × 카테고리 보정
- anchors.anchor_for(cat1, name, feature) → 카테고리/제품특성 기반 총량(앵커)
- --global_g로 전사 스케일, --cat_g로 카테고리별 배율 보정
  예: 우유류=0.92,조미소스=1.06,참치=1.10,축산캔=1.02

### 6) 소형 앙상블 & 중앙값 결합
- --ensemble_amps "1.0,1.4" 처럼 페르소나 shape 강도를 달리해 여러 분포 생성
- 월별 벡터를 median으로 합성 → 안정성↑

---

## 🏗️ 리포지토리 구조
.

├─ main.py                          # 메인 파이프라인 

├─ prompts.py                       # build_persona_prompt_numeric / (labels variant 포함)

├─ llm_utils.py                     # OpenAI 호출 재시도/프리플라이트/JSON repair

├─ anchors.py                       # anchor_for: 카테고리/제품 기반 총량 앵커

├─ feature_signals.py               # 곡선/램프/시즌성 신호 추출

├─ product_info.csv                 # 대회 제공

├─ sample_submission.csv            # 대회 제공(스키마/순서 참조)

├─ artifacts/                       # LLM 응답(jsonl), 호출 리포트(csv)

└─ submissions_v7/                  # 제출 파일(.csv) 저장 위치

✅ 본 README는 아래 파일 버전 기준으로 작성되었습니다:
prompts.build_persona_prompt_numeric,
llm_utils.request_with_retry, llm_utils.repair_json_with_llm,
anchors.anchor_for,
feature_signals.multipliers_from_feature, feature_signals.soft_ramp.

---

## 🧩 내부 모듈 요약
prompts.py
- build_persona_prompt_numeric(prod, n)
  - 숫자 전용 JSON만 허용하는 싱글턴 프롬프트 구성
  - id, share, sig{promo,novelty,ad,season}, channel[5] 이외 키 금지
- (옵션) build_persona_prompt_labels(product, n_personas)
  - 라벨/속성 중심(문자열) 프롬프트. 본 파이프라인은 numeric만 사용

llm_utils.py
- request_with_retry(...)
  - OpenAI chat.completions 재시도/JSON 강제(response_format)
- repair_json_with_llm(bad_text, model)
  - 깨진 JSON을 LLM으로 수리 (출력은 json_object)

anchors.py
- anchor_for(cat1, name, feature)
  - 카테고리별 기본 총량 + 이름/특징 키워드(그릭/단백질/용량 등)로 배율 보정

feature_signals.py
- multipliers_from_feature(feature_text, cat1, name)
  - "9-10월", "광고/모델", "SNS/바이럴", 카테고리(참치/조미 등) 신호로 월별 가중치
- soft_ramp(k, mid)
  - 출시 초기 램프업(시그모이드 형태) 곡선
 
---

## 🔍 트러블슈팅

- Missing OPENAI_API_KEY: 환경변수 또는 .env에 키 설정 필요
- LLM JSON 파싱 실패: 파이프라인에서 자동으로 schema→tool→json_object→repair→fallback 순으로 복구
- product_info.csv must contain ...: 필수 컬럼 확인
- 비상식적 피크: --campaign_amp/--pshape_amp/--ensemble_amps 완화 또는 feature_signals의 clip/shrink 조정

---

## ⚙️ 설치 & 실행
0) 요구 사항
- Python 3.9+
- pip install -U numpy pandas tqdm openai tenacity python-dotenv

1) OpenAI API 키
.env 또는 환경변수로 설정 Windows (PowerShell)
: $env:OPENAI_API_KEY="YOUR_KEY"
