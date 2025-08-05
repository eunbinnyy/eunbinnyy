# Boost up AI 2025 - CYP3A4 효소 저해 예측모델 개발

> 인공지능 기반 CYP3A4 효소 저해율 예측모델 개발 및 최적화

---

## 🏅 대회 개요
- **대회명**: Boost up AI 2025 신약개발 경진대회
- **주최 기관**: 생명연구자원센터
- **기간**: 2025년
- **참가 결과**: 최종 제출 완료

---

## 🚩 프로젝트 목표
- 화합물의 구조 데이터를 기반으로 CYP3A4 효소 저해율(% inhibition)을 정확히 예측하는 AI 모델 개발
- 신약 개발 과정에서 효율적이고 정확한 예측 모델 제안

---

## 📚 데이터셋 소개

### 학습 데이터 (`train.csv`)
- 데이터 수: 1,681종
- 주요 컬럼:
  - `ID`: 화합물의 고유 ID
  - `Canonical_Smiles`: 화합물 분자 구조 데이터
  - `Inhibition`: CYP3A4 효소 저해율(%)

### 평가 데이터 (`test.csv`)
- 데이터 수: 127종 (예시)
- 주요 컬럼:
  - `ID`: 화합물의 고유 ID
  - `Canonical_Smiles`: 화합물 분자 구조 데이터

### 제출 형식 (`sample_submission.csv`)
- `ID`: 화합물의 고유 ID
- `Inhibition`: 예측된 CYP3A4 효소 저해율(%)

---

## 🛠️ 사용한 모델 및 방법론

### 1. 특성 추출 (RDKit)
- 분자 구조로부터 RDKit descriptor 및 fingerprint를 이용한 특성 추출
  - Morgan Fingerprints
  - MACCS Keys
  - 기본 분자 특성 (MolMR, 분자량 등)

### 2. 데이터 전처리
- RobustScaler를 활용한 특성 스케일링

### 3. 모델 학습 및 최적화
- **LightGBM**: Optuna를 활용한 하이퍼파라미터 최적화 (RMSE 최소화)
- **ChemProp**: CLI 기반으로 GPU 환경에서 학습
  - 앙상블 크기: 3
  - 은닉층 크기: 500
  - 깊이: 5
  - 드롭아웃 비율: 0.3

### 4. 앙상블 예측
- Optuna 기반 앙상블 가중치 최적화 (LightGBM, ChemProp 예측값 결합)

---

## 📈 최종 모델 성능

|모델|전처리 및 기법|최적화 방법|
|---|---|---|
|LightGBM|RDKit 특성 추출 및 스케일링|Optuna (RMSE)|
|ChemProp|분자 특성 기반 학습|CLI 기반 하이퍼파라미터 조정|
|앙상블|LightGBM + ChemProp|Optuna 기반 가중치 최적화|

---

## 🗃️ 프로젝트 파일 구조
project

├── data

│   ├── train.csv

│   ├── test.csv

│   └── sample_submission.csv

├── chemprop_model

│   └── best.pt

├── chemprop_train.csv

├── chemprop_test.csv

├── chemprop_preds.csv

├── sub

│   └── submission.csv

└── src

└── base_optuna.py

---

## 모델 학습
python base_optuna.py

