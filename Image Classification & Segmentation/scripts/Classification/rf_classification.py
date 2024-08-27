import os
import cv2
import pandas as pd
import numpy as np

from sklearn import ensemble, metrics, model_selection

def load_data(file_path: str)-> pd.DataFrame:
    data = []
    labels = []
    sample_ids = []

    file_names = os.listdir(file_path)
    file_names.sort()

    for idx, file_name in enumerate(file_names):
        img_path = os.path.join(file_path, file_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (120, 120))
        img_flattened = img_resized.flatten()

        label = 1 if 'dog' in file_name else 0

        data.append(img_flattened)
        labels.append(label)
        sample_ids.append(idx)

    df = pd.DataFrame(data)
    df['sample_id'] = sample_ids
    df['target'] = labels

    return df

def perform_cv(data: pd.DataFrame, n_splits: int = 5):
    X = data.drop(columns=['sample_id', 'target'])
    y = data['target']

    skf = model_selection.StratifiedKFold(n_splits=n_splits)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{n_splits}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        accuracy = metrics.accuracy_score(y_val, y_pred)
        fold_results.append(accuracy)

        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
    
    print(f"Mean Accuracy across folds: {np.mean(fold_results):.4f}")

if __name__ == '__main__':
    # data = load_data('../../../data/cat_dog/train/')
    # data.to_csv('../../../data/cat_dog/train.csv', index=False)
    data = pd.read_csv('../../../data/cat_dog/train.csv')
    perform_cv(data, n_splits=5)
