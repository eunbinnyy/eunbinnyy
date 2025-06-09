def random_pad(mels, pad_size, mfcc=True):
    pad_width = pad_size - mels.shape[1]
    left = int(pad_width * np.random.rand())
    right = pad_width - left
    mels = np.pad(mels, ((0, 0), (left, right)), mode='constant')
    mels = (mels - mels.min()) / (mels.max() - mels.min())
    return mels

def extract_features(audio_list, size=40, pad_size=40, repeat=5):
    mels_all, mfccs_all = [], []
    for y in audio_list:
        mels = librosa.feature.melspectrogram(y, sr=sr, n_mels=size)
        mels = librosa.power_to_db(mels, ref=np.max)
        mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=size)
        for _ in range(repeat):
            mels_all.append(random_pad(mels, pad_size, mfcc=False))
            mfccs_all.append(random_pad(mfcc, pad_size, mfcc=True))
    return np.array(mels_all), np.array(mfccs_all)
