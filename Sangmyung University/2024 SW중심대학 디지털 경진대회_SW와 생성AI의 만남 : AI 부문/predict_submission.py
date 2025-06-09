import pandas as pd
def generate_submission(preds, sample_csv_path, output_csv_path):
    submission = pd.read_csv(sample_csv_path)
    submission['label'] = np.argmax(preds, axis=1)
    submission.to_csv(output_csv_path, index=False)
    print(f"Submission saved to {output_csv_path}")
