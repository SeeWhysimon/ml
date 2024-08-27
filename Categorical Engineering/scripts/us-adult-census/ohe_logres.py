import pandas as pd
import config

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv(config.DATA_DIR + 'train_folds.csv')
    
    num_cols = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week']
    df = df.drop(num_cols, axis=1)
    
    features = [
        f for f in df.columns if f not in ('kfold', 'income')
    ]

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train['income'].values)
    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid['income'].values, valid_preds)
    print(f'Fold: {fold}, AUC = {auc}')

if __name__ == '__main__':
    for fold in range(config.NUM_FOLD):
        run(fold)