import pandas as pd
import config

from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

def run(fold):
    df = pd.read_csv(config.DATA_DIR + 'preprocessed_train.csv')
    features = [
        f for f in df.columns if f not in ('id', 'target', 'kfold')
    ]
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = OneHotEncoder()

    full_data = pd.concat([df_train[features], df_valid[features]],
                          axis=0)
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(x_train, df_train['target'].values)

    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid['target'].values, valid_preds)
    print(f'Fold: {fold}, AUC: {auc}')

if __name__ == '__main__':
    for fold in range(config.NUM_FOLD):
        run(fold)