import pandas as pd
import config
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    df = pd.read_csv(config.DATA_DIR + 'train.csv')

    features = [
        f for f in df.columns if f not in ('id', 'target')
    ]

    # Fill all missing values with 'NONE' or -1
    for col in features:
        if df[col].dtype == 'object':
            df.loc[:, col] = df[col].fillna('NONE')
        else:
            df.loc[:, col] = df[col].fillna(-1)

    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=config.NUM_FOLD)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    df.to_csv(config.DATA_DIR + 'preprocessed_train.csv', index=False)