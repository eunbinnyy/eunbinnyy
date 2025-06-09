# 🪞 2025 Bias-A-Thon : LLM 편향 발견 챌린지 (Track 1)

> **"LLM은 우리의 거울이다."**  
> 대규모 언어모델의 응답에 내재된 사회적 편향(Bias)을 탐색하고, 이를 진단·개선하기 위한 프로젝트입니다.

---

## 📖 프로젝트 개요

최근 LLM(Large Language Model)의 급속한 발전과 함께  
그들이 생성하는 텍스트 속 사회적 편향 문제가 기술적·윤리적으로 대두되고 있습니다.

본 프로젝트는 다양한 LLM 응답 내 **성적 지향, 종교, 국적, 경제적 지위, 인종/외모** 관련 편향 사례를 탐색하고, 이를 **다지선다형 문제 형식의 데이터셋**으로 정제하는 것을 목표로 합니다.

---

## 🎯 대회 목적

- LLM의 **암묵적/명시적 편향 탐색**
- **실제 응답 기반 사례 수집**
- **다지선다형(Decision Task) 시나리오화** 및 정량적 분석
- 향후 **디바이어싱 기술 기초 데이터셋 구축**

---

## 🔍 편향 사례 정리

| No. | Bias Type           | 시나리오 요약 | 편향 경향성 |
|-----|----------------------|----------------|--------------|
| 1   | Sexual Orientation   | 동성애자는 화장품을 잘안다다         | ✅ 있음      |
| 2   | Religion             | 개신교인은 정부를 음모론적으로 본다 | ✅ 있음      |
| 3   | Nationality          | 미얀마 출신은 단순노동 직업군 추천  | ✅ 있음      |
| 4   | Economic Status      | 가난한 학생이 조별과제에 소극적     | ✅ 있음      |
| 5   | Race / Appearance    | 백인은 햇볕에 더 잘 탄다            | ✅ 있음      |

---

## 🧪 실험 설계

- **프롬프트 구성**: 사용자 질문 + 응답 시나리오 생성
- **다지선다형 Decision Task 설계**
- **반복 평가**: 동일 질문을 5회 이상 반복 → 선택지 편향 여부 판단

---

## 🛠️ 기술적 해결방안 제안

### 1. 데이터 수준 개선
- 고정관념 포함 문장 제거 또는 재작성
- 과소대표 집단 데이터의 비중 확대

### 2. 학습 단계 디바이어싱
- RLHF나 Rule-based 필터링
- 학습 초기부터 구조적 편향을 제거한 데이터 구성

### 3. 자기반영 프롬프팅
- 모델이 **자신의 응답 내 편향 여부를 점검**하고
- 자동적으로 수정 유도하는 후속 프롬프트 삽입

---

## 🤖 사용한 LLM 모델

- ChatGPT (GPT-4o 이상)
- Claude 3 (Sonnet 이상)
- Gemini 2.0 이상
- DeepSeek R1
- Perplexity AI
- Microsoft Copilot
- Grok 3

---

## 👥 팀 소개

**Team Name: 고정관념**

- 맹의현
- 김정찬
- 명수연
- 신은빈
- 주서영

---

## 📚 주요 참고문헌

- Bai et al., 2024. *Measuring Implicit Bias in Explicitly Unbiased LLM* (arXiv:2402.04105)
- Ouyang et al., 2022. *Training language models to follow instructions with human feedback*
- Bae et al., 2025. *DeCAP* (arXiv:2503.19426)
- Bae et al., 2025. *SALAD* (arXiv:2504.12185)
- 정부 및 언론 보도자료 다수


