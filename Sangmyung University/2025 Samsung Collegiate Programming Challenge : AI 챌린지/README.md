# 🏆 2025 Samsung Collegiate Programming Challenge : AI 챌린지 (멀티모달 QA)

삼성전자가 주최하는 **2025 SCPC : AI 챌린지** 예선에 참가하여  
사용자의 일상 사진을 이해하고, 주어진 **선다형 질문(MCQ)**에 대해 올바른 정답을 선택하는  
**멀티모달 AI 모델**을 개발한 프로젝트입니다.  

본 코드는 예선에서 **상위 10%** 성적을 기록하였습니다. 🚀  

---

## 📌 대회 개요
- **대회명** : 2025 Samsung Collegiate Programming Challenge : AI Challenge
- **주제** : 사용자의 스마트폰 갤러리에 저장된 일상 사진을 기반으로,  
  주어진 질문에 대해 올바른 선택지를 예측하는 멀티모달 AI 모델 개발
- **입력** : 이미지 + 질문(텍스트) + 선택지 4개 (A, B, C, D)
- **출력** : 정답 선택지 (A, B, C, D 중 하나)
- **평가 방식** : Private 리더보드 정확도 (Accuracy)

---

## 📂 데이터셋 구조
open/
├── train_input_images/ # 학습 이미지 (예: TRAIN_000.jpg ~ TRAIN_059.jpg)
├── test_input_images/ # 테스트 이미지 (예: TEST_000.jpg ~ TEST_851.jpg)
├── train.csv # 학습 데이터 (이미지 경로, 질문, 선택지, 정답 포함)
├── test.csv # 테스트 데이터 (이미지 경로, 질문, 선택지 포함)
└── sample_submission.csv # 제출 형식 (ID, answer)

---

## ⚙️ 모델 및 접근 방법
본 프로젝트에서는 **InstructBLIP** 계열 모델을 활용하여 이미지와 텍스트를 동시에 이해하는 **멀티모달 추론 파이프라인**을 구축하였습니다.

1. **모델 선택**
   - [`Salesforce/instructblip-flan-t5-xl`](https://huggingface.co/Salesforce/instructblip-flan-t5-xl) 사용
   - Hugging Face `transformers` 기반 로드

2. **프롬프트 엔지니어링**
   - 동일 질문/선택지에 대해 **3가지 프롬프트 변형** 생성
   - 예: "Answer:", "Your answer:", "Please select A, B, C, or D." 등

3. **Prompt Ensemble**
   - 각 프롬프트별 출력 → 정규식 기반 답변 추출
   - 최종 답변은 **다수결 투표(Majority Voting)**로 결정

4. **추론 파이프라인**
   - 학습 데이터로 **Train Accuracy** 확인
   - Test 데이터에 대해 답변 예측 후 `submission.csv` 생성
     
---

## 📂 핵심 아이디어

멀티모달 추론 : 이미지 + 질문 + 선택지를 동시에 입력

Prompt Ensemble : 다양한 프롬프트 구성으로 모델 일관성 확보

다수결 투표 : 불확실성 완화 및 안정적인 정답 추론

---

## 📊 결과
- **리더보드 성적** : 상위 10%
- **주요 성능 향상 요인**
  - 프롬프트 다양화 + 다수결 앙상블
  - 불확실 출력에 대한 정규식 기반 후처리

---

## 🚀 실행 방법
```bash
# 1. 환경 설치
pip install torch torchvision transformers pandas pillow tqdm

# 2. 저장된 모델 로드
python main.ipynb

# 3. 결과 확인
cat ./sub/submission.csv
