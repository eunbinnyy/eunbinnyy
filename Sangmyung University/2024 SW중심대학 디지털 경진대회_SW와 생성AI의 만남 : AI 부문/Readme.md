# 🗣️ Fake Voice Detection using Deep Learning

> **Project Title:** Detecting AI-generated Fake Voices in 5-second English Audio Clips  
> **Competition Task:** Classify real vs fake voices using deep learning and audio features  
> **Key Features:** Mel-spectrogram, MFCC, CNN + Residual Blocks, K-Fold CV

---

## 🧠 Background

Recent advances in generative AI have enabled highly realistic fake voices that can mimic public figures or alter real speech. These fake audios pose major threats such as misinformation, reputation damage, or financial fraud.

To counter this, we propose a deep learning-based fake voice detection system that can distinguish AI-generated voices from real human voices in short audio clips.

---

## 🧪 Task Overview

- **Input:** 5-second English audio clips
- **Output:** Binary label – Real (0) or Fake (1)
- **Data:** Real and fake voices recorded in controlled environments
- **Challenge:** Evaluation set may contain mixed voices and noisy conditions

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `1_data_loader.py` | Loads `.wav` files and returns audio signals using `librosa` |
| `2_feature_engineering.py` | Extracts mel-spectrograms and MFCC features with random padding |
| `3_model.py` | CNN model architecture with residual blocks for audio classification |
| `4_train_eval.py` | Training pipeline with StratifiedKFold CV and model ensembling |
| `5_predict_submission.py` | Aggregates test predictions and generates `submission.csv` |

---

## 🔍 Techniques Used

- `librosa` for audio feature extraction
- Mel-spectrograms and MFCCs for input representation
- Random padding as augmentation
- Deep CNN + residual architecture
- K-Fold cross validation for robust training
- Ensembling for prediction refinement

---

## 🛠 Requirements

```bash
pip install librosa numpy pandas matplotlib scikit-learn tensorflow
