import librosa
import numpy as np
import pandas as pd

sr = 20000

def load_audio(file_names, path):
    audios = []
    for fname in file_names:
        y, _ = librosa.load(path + fname, sr=sr)
        audios.append(y)
    return np.array(audios)
