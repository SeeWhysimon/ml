import pandas as pd
from sklearn.model_selection import KFold

data = pd.read_csv('../../data/mnist/mnist.csv')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
data['kfold'] = -1

for fold, (train_index, val_index) in enumerate(kf.split(data)):
    data.loc[val_index, 'kfold'] = fold

data.to_csv('../../data/mnist/mnist_train_folds.csv', index=False)