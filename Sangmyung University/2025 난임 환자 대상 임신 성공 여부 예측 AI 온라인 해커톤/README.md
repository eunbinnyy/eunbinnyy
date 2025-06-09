# 🧬 난임 환자 임신 성공 여부 예측 AI 모델

## 🏆 성과

> Lg aimers 6기에 참가하여 수상에는 아쉽게 들지 못했지만, **상위 10% 이내의 성적을 기록**

## 🩺 프로젝트 소개

난임 치료는 많은 환자에게 심리적·경제적 부담을 동반합니다. 본 프로젝트는 실제 난임 환자 데이터를 바탕으로 **임신 성공 여부를 예측**하는 AI 모델을 개발함으로써, 보다 정밀한 치료 전략 수립에 기여하고자 합니다.

## 🎯 목적

- **임신 성공(출산까지 진행된 임신)** 여부를 예측하는 고성능 AI 모델 개발
- 다양한 모델을 활용한 **앙상블**을 통해 성능 극대화
- 주요 특성 분석을 통해 **의학적 인사이트 제공**


## 🧾 사용 데이터

- 학습 데이터: `train_preprocessed.csv`
- 테스트 데이터: `test_preprocessed.csv`
- 제출 양식: `sample_submission.csv`

## 🧪 모델 구성

1. **PyCaret 기반 AutoML**
2. **AutoGluon 기본 모델**
3. **AutoGluon + Feature Engineering**
4. **위 세 가지 모델의 앙상블**

## ⚙️ 주요 라이브러리

- PyCaret
- AutoGluon
- Optuna
- Scikit-learn
- Pandas / NumPy


## 👥 팀 소개

**Team Name: LGgram에어팟연결안됨**

- 맹의현
- 김진석
- 명수연
- 신은빈
- 유민균

## 📦 실행 방법

```bash
pip install pycaret autogluon optuna
python main.py

