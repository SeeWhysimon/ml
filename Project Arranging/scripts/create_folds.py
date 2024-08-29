import config
import pandas as pd
from sklearn.model_selection import KFold

data = pd.read_csv(config.ORIGINAL_FILE)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
data['kfold'] = -1

for fold, (train_index, val_index) in enumerate(kf.split(data)):
    data.loc[val_index, 'kfold'] = fold

data.to_csv(config.TRAINING_FILE, index=False)