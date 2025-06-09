from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def train_kfold(X_mel, X_mfcc, y, X_test_mel, X_test_mfcc, n_splits=5):
    acc_list, preds_list = [], []
    skf = StratifiedKFold(n_splits=n_splits)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_mel, y)):
        print(f"\n[FOLD {fold+1}]")
        model_mel = build_model()
        model_mfcc = build_model()

        x_tr, x_val = X_mel[tr_idx], X_mel[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model_mel.fit(x_tr, y_tr, epochs=50, batch_size=32, validation_data=(x_val, y_val), verbose=0)
        preds_val_mel = model_mel.predict(x_val)

        x_tr, x_val = X_mfcc[tr_idx], X_mfcc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model_mfcc.fit(x_tr, y_tr, epochs=50, batch_size=32, validation_data=(x_val, y_val), verbose=0)
        preds_val_mfcc = model_mfcc.predict(x_val)

        preds_val = preds_val_mel + preds_val_mfcc
        acc = accuracy_score(y_val, np.argmax(preds_val, axis=1))
        acc_list.append(acc)

        preds_test = model_mel.predict(X_test_mel) + model_mfcc.predict(X_test_mfcc)
        preds_list.append(preds_test)

    return np.mean(acc_list), sum(preds_list)
