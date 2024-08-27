import itertools
import pandas as pd
import xgboost as xgb
import config

from sklearn import metrics
from sklearn import preprocessing

def feature_engineering(df, cat_cols):
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df[c1 + '_' + c2] = df[c1].astype(str) + '_' + df[c2].astype(str)
    return df

def run(fold):
    df = pd.read_csv(config.DATA_DIR + 'train_folds.csv')
    
    num_cols = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week']
    cat_cols = [c for c in df.columns if c not in num_cols and c not in ('income', 'kfold')]

    df = feature_engineering(df, cat_cols)

    features = [
        f for f in df.columns if f not in ('kfold', 'income')
    ]

    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            df[col] = lbl.fit_transform(df[col])

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(n_jobs=-1,
                              max_depth=7)
    model.fit(x_train, df_train['income'].values)
    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid['income'].values, valid_preds)
    print(f'Fold: {fold}, AUC = {auc}')

if __name__ == '__main__':
    for fold in range(config.NUM_FOLD):
        run(fold)