
# 🧬 MAI 2024: 유전자 발현 예측 AI (H&E 조직 이미지 기반)

> 고려대학교 의료원 주최 '제1회 MAI(Medical Artificial Intelligence) 경진대회' 참가 프로젝트  
> **H&E 염색된 조직 이미지**로부터 **3467개의 유전자 발현 수치**를 예측하는 AI 모델 개발

---

## 🎯 목표 및 개요

- 의료 영상(H&E stained tissue image)에서 유전자 발현 정보를 정량적으로 예측
- 이미지-유전자 간 관계를 학습할 수 있는 비전 모델 설계 및 최적화
- 실제 의료 연구 및 진단 보조 시스템 개발 가능성을 검증하는 것이 목표

---

## 🧠 주요 특징

- 모델: `ViT (Vision Transformer)` 기반 회귀 모델
- 증강: CutMix, FMix, MixUp
- 전략: EarlyStopping, TTA (Test Time Augmentation), Self-Supervised Pretraining
- 앙상블: ViT + EfficientNet + MobileNet 등 다양한 모델 실험
- 성능지표: MSE (Mean Squared Error)

---

## 🧪 결과

| 데이터셋 | Public Score | Private Score |
|----------|--------------|---------------|
| MAI test | **0.53607**    | **0.54317**  |

> 대회 최종 상위 10% 이내 성과 기록

---

## 🛠️ 기술 스택

- Python 3.10
- PyTorch, torchvision, transformers
- Albumentations (이미지 증강)
- ViT (google/vit-base-patch16-224-in21k)
- Colab 환경 기반 학습

---

## 📁 코드 구조
├── train # 학습 파이프라인

├── inference # 추론 및 TTA

├── utils/ # 데이터 증강 함수 등

├── dataset/ # csv 및 이미지 샘플

├── notebooks/ # 실험 기록 노트북

├── models/ # 저장된 모델 체크포인트

└── result/ # 예측 결과 csv

---

