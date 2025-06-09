import pandas as pd
import numpy as np

def create_features(icu_df: pd.DataFrame, labs_df: pd.DataFrame) -> pd.DataFrame:
    """
    icu_df: ICU 환자 기본 정보 (hadm_id, age, gender, AKI label 등 포함)
    labs_df: 크레아티닌 등 labevents 정보
    return: Feature가 추가된 dataframe
    """
    # 크레아티닌의 최대값, 최소값, 평균, 표준편차 계산
    labs_df = labs_df.dropna(subset=['value'])
    lab_stats = labs_df.groupby('subject_id')['value'].agg(['mean', 'std', 'min', 'max']).reset_index()
    lab_stats.columns = ['subject_id', 'creatinine_mean', 'creatinine_std', 'creatinine_min', 'creatinine_max']

    # ICU 기본정보와 병합
    merged = pd.merge(icu_df, lab_stats, on='subject_id', how='left')
    merged = merged.dropna(subset=['creatinine_mean'])

    # 성별 인코딩
    merged['gender'] = merged['gender'].map({'M': 1, 'F': 0})
    return merged


# model_train
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_and_evaluate(df: pd.DataFrame):
    features = ['anchor_age', 'gender', 'creatinine_mean', 'creatinine_std', 'creatinine_min', 'creatinine_max']
    target = 'AKI_label'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[Classification Report]\n", classification_report(y_test, y_pred))
    print("\n[Confusion Matrix]\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, "aki_rf_model.pkl")
    print("✅ 모델 저장 완료: aki_rf_model.pkl")


# eda_plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_distribution(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='creatinine_mean', hue='AKI_label', kde=True, bins=30)
    plt.title('Creatinine Mean Distribution by AKI Label')
    plt.xlabel('Creatinine Mean')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('creatinine_mean_distribution.png')
    print("📊 저장 완료: creatinine_mean_distribution.png")
