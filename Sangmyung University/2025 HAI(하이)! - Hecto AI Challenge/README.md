# 🚗 HAI 2025: 중고차 이미지 차종 분류 AI 모델

## 📌 개요

본 프로젝트는 `HAI(하이)! - Hecto AI Challenge: 2025 상반기 헥토 채용 AI 경진대회`에 참가하여 수행한 중고차 차종 분류 AI 모델 개발 과제입니다. 실제 중고차 플랫폼에서 수집된 차량 이미지를 기반으로, "총 396종의 차량 차종(class)"을 구분하는 모델을 개발하였습니다.

해당 과제는 헥토 그룹의 우수 인재 선발 및 채용 연계 목적의 AI 챌린지 시리즈 중 하나로, 이미지 분류 기반의 정밀한 차종 인식 기술을 요구합니다. 본 프로젝트에서는 다양한 데이터 증강, 앙상블 전략, 부정형 이미지 필터링 등의 고도화된 전략을 활용해 "Public 리더보드 30위, Private 리더보드 24위 (상위 4%)"의 성과를 달성하였습니다.

---

## 🏆 대회 성과

- ✅ Public 리더보드 : 30등 / 상위 4%
- ✅ Private 리더보드 : 24등 / 상위 4%
- 🔍 396개 차종 분류 문제에서 Log Loss 최적화를 위한 다양한 실험 및 개선 진행

---

## 🗂️ 프로젝트 구조
├── code_final.ipynb # 전체 학습 및 추론 파이프라인 통합 notebook

├── data/

│   ├── train/ # 학습 이미지 (396개 클래스 폴더, 총 33,137장)

│   └── test/ # 테스트 이미지 (8,258장)

├── test.csv # 테스트셋 이미지 경로 및 ID

├── sample_submission.csv # 제출 양식

└── submission.csv 

---

## 🧠 모델링 전략

### 1. 전체 파이프라인
- Google Colab 기반 작업 디렉토리 설정
- 학습 전체 파이프라인은 하나의 Jupyter Notebook(`code_final.ipynb`)으로 구성

### 2. 백본 모델
- `ConvNeXt-Base`, `ConvNeXt-Large` 기반의 모델 구조
- classifier layer를 `396` 클래스에 맞춰 수정
- `EMA` 적용 없이 PyTorch 기본 저장 방식 사용

### 3. 손실 함수 및 학습 전략
- `MultiFocalLoss`: gamma 조합 (1.0, 2.0) + 가중치 (0.6, 0.4)
- `AdamW` 옵티마이저 + `CosineAnnealingLR` 스케줄러
- `GradScaler`, `autocast`로 AMP 학습
- `EarlyStopping` 조건 적용 (patience 5)

### 4. 이미지 증강 및 입력 변형
- `AutoAugment`, `Horizontal Flip`, `CenterCrop`
- Canny Edge version 이미지 추가 사용
- 각 이미지에 대해 3가지 버전 (원본 / Flip / Edge) 학습

### 5. 비정형 이미지 삭제
- test dataset에서 실내, 내비게이션, 차키, 외부환경 이미지들 제거
- 수작업으로 확인 후 학습 데이터에서 제외

---

## ⚙️ 주요 라이브러리

- Python 3.10  
- PyTorch 2.x  
- torchvision  
- scikit-learn  
- pandas, numpy  
- matplotlib  
- tqdm  
- PIL (Image)  
- OpenCV (Canny edge)  
- Google Colab

---

## 🧪 평가 기준 (Log Loss)

```math
\text{LogLoss} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(p_{ij})
