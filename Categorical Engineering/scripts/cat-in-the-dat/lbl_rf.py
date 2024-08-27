import pandas as pd
import config

from sklearn import ensemble
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

def run(fold):
    df = pd.read_csv(config.DATA_DIR + 'preprocessed_train.csv')
    features = [
        f for f in df.columns if f not in ('id', 'target', 'kfold')
    ]

    for col in features:
        lbl = LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])
    
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, df_train['target'].values)

    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid['target'].values, valid_preds)

    print(f'Fold: {fold}, AUC: {auc}')
    
if __name__ == '__main__':
    for fold in range(config.NUM_FOLD):
        run(fold)