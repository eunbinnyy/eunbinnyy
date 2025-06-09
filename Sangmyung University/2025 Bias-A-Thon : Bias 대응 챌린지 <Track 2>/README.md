# 🧠 Bias-A-Thon 2025 - Track 2

> **주제:** Llama-3.1-8B-Instruct 모델 기반 편향 상황에서도 공정하고 중립적인 응답을 생성하는 프롬프트 및 RAG 기법 개발  

---

## 📌 프로젝트 소개

본 프로젝트는 2025 Bias-A-Thon Track 2에서 제시된 과제에 응답하기 위해 개발되었습니다.  
Llama-3.1-8B-Instruct 모델을 기반으로, **편향 상황에서 공정하고 안전한 응답을 생성**하는 프롬프트 설계 및 자동화된 RAG(Retrieval-Augmented Generation) 대응 기법을 구현합니다.

---

## 🔍 주요 기능 요약

-  **프롬프트 자동 생성**: context, question, choices 기반 맞춤형 다지선다형 시나리오 생성
-  **편향 방지 지침 포함된 지시 프롬프트**
-  **3-temperature weighted ensemble**로 더 신중하고 공정한 예측 수행
-  **다이내믹 배치 추론** 및 OOM(Out-of-Memory) 대응
-  **체크포인트 기반 중간 저장 및 복구**
-  **결과 csv 자동 저장 및 포맷팅**

---

## 🛠️ 실행 환경

- Python ≥ 3.9  
- Transformers ≥ 4.39  
- PyTorch ≥ 2.0  
- Llama-3.1-8B-Instruct  
- CUDA 11.x 이상 (GPU 필요)

---

## 📂 디렉토리 구조

```bash
.
├── inference_bias.py          # 전체 추론 파이프라인 (본 코드)
├── data/
│   └── test.csv               # 입력 CSV 파일
├── baseline/
│   └── checkpoint_*.csv       # 체크포인트 파일 저장 위치
├── sub/
│   └── submission_*.csv       # 최종 결과 저장 위치
```

---

## 🚀 실행 방법

```bash
# Jupyter / Colab 환경
!python inference_bias.py --batch_size 4 --checkpoint_interval 100 --save_dir baseline

# CLI 환경
python inference_bias.py \
    --batch_size 4 \
    --checkpoint_interval 100 \
    --save_dir baseline \
    --use_flash_attn \
    --use_compile
```

> ⚠️ 반드시 **`meta-llama/Llama-3.1-8B-Instruct`** 모델을 사용해야 하며, HuggingFace 토큰 권한이 필요합니다.

---

## 📄 제출 형식 저장

```python
input_path = "./baseline/submission.csv"
output_path = "./sub/submission_batch4_7me_eun_sam_f.csv"

df = pd.read_csv(input_path)
df.to_csv(output_path, index=False, encoding="utf-8-sig")
```

---

## 🧪 평가 기준 대응 전략

| 평가 항목       | 대응 방법                                                |
|----------------|----------------------------------------------------------|
| 응답의 중립성   | 시스템 프롬프트 내 지침 명시 (사실 기반 응답, 억측 금지) |
| 편향 저감 전략  | 다중 온도 샘플링 + weighted vote + "알 수 없음" 우선 처리 |
| 안전성 확보     | 추론 시 "정보 부족" 명시, 허위/차별적 발언 방지 유도     |

---

## 👥 팀원

- 🧑‍💻 신은빈
- 🧑‍💻 김정찬
- 🧑‍💻 맹의현
- 🧑‍💻 명수연
- 🧑‍💻 주서영


