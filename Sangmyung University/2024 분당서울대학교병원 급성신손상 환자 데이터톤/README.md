# 🩺 Acute Kidney Injury (AKI) Prediction Challenge

> **Team:** 산할머니  
> **Hosted by:** 분당서울대학교병원 × MIMIC-IV Datathon  
> **Homepage:** [https://snubh-datathon.imweb.me](https://snubh-datathon.imweb.me)

---

## 📌 Competition Overview

- **Goal:** Develop an AI model to predict Acute Kidney Injury (AKI)
- **Preliminary:** Predict AKI using MIMIC-IV dataset
- **Final:** Apply to real clinical dataset from Bundang Seoul National University Hospital

---

## 👥 Team Introduction

**Team Name:** 산할머니  
**Affiliation:** 상명대학교 휴먼지능정보공학전공  

Our team comprises four undergraduate AI engineering students with diverse specialties in Vision, LLM, Multimodal, BCI, and Data Science.  
We actively participate in paper reading groups and project-based learning, and we have experience in multiple AI competitions.

> 🏆 Previous Awards:
> - Big Data AI Competition  
> - Generative Intelligence Model Contest  
> - Solar Astronomical Event Detection  
> - Biohealth AI Challenge (Oral Surgery Risk Prediction)

**Goal:** Develop a publishable-quality model through collaboration with SNUBH.

---

## 🧪 Model Development Plan

### 🔹 Preliminary Round: MIMIC-IV Dataset

1. EDA & Preprocessing
   - Handle missing values and outliers
   - Feature selection based on importance
2. Imbalance handling: Oversampling / Undersampling
3. Model Candidates:
   - TabNet, ResNet (Tabular), XGBoost, LightGBM, CatBoost, Logistic Regression
4. Hyperparameter tuning with HyperOpt
5. Ensemble top 3–5 models

### 🔸 Final Round: SNUBH Real Clinical Data

1. If structured (CSV): same pipeline
2. If unstructured (CT/Ultrasound): vision model approach
   - Apply augmentation: rotate, crop, flip
3. Normalize images, handle imbalance
4. Use StratifiedKFold for robustness
5. Apply CNNs: ResNet, EfficientNet, ViT

---

## 🧬 Data Handling (MIMIC-IV)

Used large-scale structured files:

- `icustays.csv`, `admissions.csv`, `patients.csv`
- `labevents.csv` (Creatinine, Hemoglobin, BUN, etc.)
- `diagnoses_icd.csv` (ICD-9/10 AKI codes)
- `chartevents.csv`, `outputevents.csv`

Used `pandas.read_csv(..., chunksize=...)` to manage large data efficiently.

### 💡 Sample Code

```python
import pandas as pd
chunksize = 100000
filtered = pd.DataFrame()
for chunk in pd.read_csv("labevents.csv.gz", chunksize=chunksize, compression='gzip'):
    f_chunk = chunk[chunk['itemid'].isin([50912, 52546])]
    filtered = pd.concat([filtered, f_chunk], ignore_index=True)

