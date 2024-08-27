import copy
import config
import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def mean_target_encoding(data: pd.DataFrame):
    df = copy.deepcopy(data)
    num_cols = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week']
    features = [f for f in df.columns if f not in num_cols 
                and f not in ('income', 'kfold')]
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            df[col] = lbl.fit_transform(df[col])

    encoded_dfs = []
    for fold in range(config.NUM_FOLD):
        df_train = df[df['kfold'] != fold].reset_index(drop=True)
        df_valid = df[df['kfold'] == fold].reset_index(drop=True)
        for col in features:
            mapping_dict = dict(
                df_train.groupby(col)['income'].mean()
            )
            df_valid[col + '_enc'] = df_valid[col].map(mapping_dict)
        encoded_dfs.append(df_valid)
    encoded_dfs = pd.concat(encoded_dfs, axis=0)
    return encoded_dfs

def run(df: pd.DataFrame, fold: int):
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    features = [f for f in df.columns if f not in ('kfold', 'income')]
    x_train = df_train[features].values
    x_valid = df_valid[features].values
    model = xgb.XGBClassifier(n_jobs=-1, 
                              max_depth=7)
    model.fit(x_train, df_train['income'].values)
    valid_preds = model.predict_proba(x_valid)[:, 1]    
    auc = metrics.roc_auc_score(df_valid['income'].values, valid_preds)
    print(f'Fold: {fold}, AUC: {auc}')

if __name__ == '__main__':
    df = pd.read_csv(config.DATA_DIR + 'train_folds.csv')
    df = mean_target_encoding(df)
    for fold in range(config.NUM_FOLD):
        run(df, fold)