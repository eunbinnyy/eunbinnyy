import os
import re
import pandas as pd
import numpy as np
from pycaret.classification import *
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 로드
train = pd.read_csv("./data/train_preprocessed.csv", index_col=0)
test = pd.read_csv("./data/test_preprocessed.csv", index_col=0)
sample_submission = pd.read_csv("./data/sample_submission.csv")

X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

# 특수문자 제거 및 중복 컬럼 처리
def clean_column_names(df):
    df.columns = [re.sub(r'\W+', '_', col) for col in df.columns]
    return df

def rename_duplicate_columns(df):
    seen = {}
    new_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    df.columns = new_columns
    return df

X = clean_column_names(X)
test = clean_column_names(test)

X = rename_duplicate_columns(X)
test = rename_duplicate_columns(test)

# PyCaret 모델 학습 및 예측
clf1 = setup(
    data=pd.concat([X, y], axis=1),
    target='임신 성공 여부',
    session_id=42,
    use_gpu=False,
    feature_selection=False,
    silent=True,
    verbose=False
)

best_model = compare_models(include=['lightgbm', 'xgboost', 'catboost'], fold=5)
final_model = finalize_model(best_model)
pycaret_proba = predict_model(final_model, data=test, raw_score=True)
pycaret_proba_1 = pycaret_proba.iloc[:, -1]

# AutoGluon 기본 모델
train_data = TabularDataset(pd.concat([X, y], axis=1))
test_data = TabularDataset(test)

predictor = TabularPredictor(
    label='임신 성공 여부',
    eval_metric='roc_auc',
    path="./autogluon_models"
).fit(
    train_data,
    presets='best_quality',
    time_limit=10800,
    num_gpus=1
)

autogluon_proba = predictor.predict_proba(test_data)
autogluon_proba_1 = autogluon_proba[1]

# Feature Engineering + AutoGluon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

train_scaled = pd.DataFrame(X_scaled, columns=X.columns)
train_scaled['임신 성공 여부'] = y.values
train_scaled_data = TabularDataset(train_scaled)
test_scaled_data = TabularDataset(pd.DataFrame(test_scaled, columns=test.columns))

predictor_fe = TabularPredictor(
    label='임신 성공 여부',
    eval_metric='roc_auc',
    path="./autogluon_models"
).fit(
    train_scaled_data,
    presets='best_quality',
    time_limit=10800,
    num_gpus=1
)

autogluon_fe_proba = predictor_fe.predict_proba(test_scaled_data)
autogluon_fe_proba_1 = autogluon_fe_proba[1]

# 앙상블 및 저장
sample_submission['probability'] = (
    pycaret_proba_1 * 0.33 +
    autogluon_proba_1 * 0.33 +
    autogluon_fe_proba_1 * 0.34
)

sample_submission.to_csv("./final_submission_ensemble.csv", index=False)
print("✅ 최종 앙상블 제출 파일이 저장되었습니다: final_submission.csv")
