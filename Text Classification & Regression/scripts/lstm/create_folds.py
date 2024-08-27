import pandas as pd
from sklearn import model_selection

import config

def create_folds(imdb_path: str):
    df = pd.read_csv(imdb_path)
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df["sentiment"].values

    kf = model_selection.StratifiedKFold(n_splits=config.FOLD_NUM)
    for fold_, (train_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, "kfold"] = fold_
    
    return df