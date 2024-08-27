import pandas as pd
import config
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv(config.DATA_DIR + 'adult.csv')
    
    income_mapping = {
        '<=50K': 0,
        '>50K': 1
    }
    df['income'] = df['income'].map(income_mapping).astype(int)
    
    features = [
        f for f in df.columns if f not in ('income')
    ]

    df = df.replace('?', 'NONE')

    # Fill all missing values with 'NONE' or -1
    for col in features:
        if df[col].dtype == 'object':
            df.loc[:, col] = df[col].fillna('NONE')
        else:
            df.loc[:, col] = df[col].fillna(-1)
    
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['income'].values
    kf = model_selection.StratifiedKFold(n_splits=config.NUM_FOLD)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    df.to_csv(config.DATA_DIR + 'train_folds.csv', index=False)