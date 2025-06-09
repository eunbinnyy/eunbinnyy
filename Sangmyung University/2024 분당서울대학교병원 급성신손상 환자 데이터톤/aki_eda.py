import pandas as pd
import numpy as np
import gzip
import gc
import os
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything()

# ───────────────────────
# Step 1. Load Core Tables
# ───────────────────────

def load_csv_gz(path):
    with gzip.open(path) as f:
        return pd.read_csv(f)

icu_stay = load_csv_gz('/content/drive/MyDrive/MIMIC-IV/3.0/icu/icustays.csv.gz')
icu_stay = icu_stay.drop_duplicates(subset=['hadm_id'], keep='first')

admissions = load_csv_gz('/content/drive/MyDrive/MIMIC-IV/3.0/hosp/admissions.csv.gz')[['hadm_id', 'deathtime']]
patients = load_csv_gz('/content/drive/MyDrive/MIMIC-IV/3.0/hosp/patients.csv.gz')[['subject_id', 'gender', 'anchor_age']]

# Merge info
icu_stay = pd.merge(icu_stay, patients, on='subject_id', how='left')
icu_stay = pd.merge(icu_stay, admissions, on='hadm_id', how='left')

# ───────────────────────
# Step 2. Mortality Labeling
# ───────────────────────
icu_stay['intime'] = pd.to_datetime(icu_stay['intime'])
icu_stay['deathtime'] = pd.to_datetime(icu_stay['deathtime'])
icu_stay['mortality_sec'] = (icu_stay['deathtime'] - icu_stay['intime']).dt.total_seconds()
icu_stay = icu_stay[(icu_stay['mortality_sec'].isnull()) | (icu_stay['mortality_sec'] > 0)]
icu_stay['mortality_3day'] = icu_stay['mortality_sec'] < 86400 * 3

print(icu_stay[['hadm_id', 'gender', 'anchor_age', 'mortality_3day']].head())

# ───────────────────────
# Step 3. Lab: Creatinine 추출
# ───────────────────────

# Creatinine 관련 itemid 확인
d_labitems = load_csv_gz('/content/drive/MyDrive/MIMIC-IV/3.0/hosp/d_labitems.csv.gz')
creatinine_ids = d_labitems[d_labitems['label'].str.lower().str.contains('creatinine')]['itemid'].tolist()

# labevents에서 Creatinine 추출 (메모리 절약용)
def filter_labevents_creatinine(file_path, ids, chunksize=1000000):
    result = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize, compression='gzip'):
        chunk = chunk[chunk['itemid'].isin(ids)]
        result.append(chunk)
    return pd.concat(result, ignore_index=True)

labevents_creatinine = filter_labevents_creatinine(
    '/content/drive/MyDrive/MIMIC-IV/3.0/hosp/labevents.csv.gz',
    creatinine_ids
)

print(labevents_creatinine[['subject_id', 'itemid', 'charttime', 'value']].head())

# ───────────────────────
# Step 4. AKI 라벨링 준비 (예: SCr 추이 기반 라벨링)
# ───────────────────────

labevents_creatinine['charttime'] = pd.to_datetime(labevents_creatinine['charttime'])
labevents_creatinine['value'] = pd.to_numeric(labevents_creatinine['value'], errors='coerce')

# 향후 creatinine 추이 분석 및 AKI 정의 기준 적용 가능 (KDIGO 기준 등)
# 예: 최대/최소 비율, 48시간 내 1.5배 이상 상승 등

# ───────────────────────
# Step 5. ICD 기반 AKI 여부 병합
# ───────────────────────

diagnoses = load_csv_gz('/content/drive/MyDrive/MIMIC-IV/3.0/hosp/diagnoses_icd.csv.gz')
d_icd = load_csv_gz('/content/drive/MyDrive/MIMIC-IV/3.0/hosp/d_icd_diagnoses.csv.gz')

# AKI 관련 ICD 코드 필터링
aki_keywords = ['acute kidney injury', 'acute renal failure', 'aki']
aki_codes = d_icd[d_icd['long_title'].str.lower().str.contains('|'.join(aki_keywords))]['icd_code'].tolist()
aki_patients = diagnoses[diagnoses['icd_code'].isin(aki_codes)]

aki_hadm_ids = aki_patients['hadm_id'].unique()
icu_stay['AKI_label'] = icu_stay['hadm_id'].isin(aki_hadm_ids).astype(int)

print(icu_stay['AKI_label'].value_counts())

# ───────────────────────
# Step 6. 저장 및 후속 분석용 Export
# ───────────────────────

icu_stay.to_csv('icu_stay_final.csv', index=False)
labevents_creatinine.to_csv('creatinine_labs.csv', index=False)

print("✅ EDA 및 전처리 완료")
