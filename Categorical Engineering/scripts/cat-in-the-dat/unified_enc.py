import pandas as pd
import config
from sklearn import preprocessing

train = pd.read_csv(config.DATA_DIR + 'train.csv')
test = pd.read_csv(config.DATA_DIR + 'test.csv')

test.loc[:, 'target'] = -1
data = pd.concat([train, test]).reset_index(drop=True)

features = [x for x in train.columns if x not in ['id', 'target']]
for feat in features:
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feat].fillna('NONE').astype(str).values
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
    train = data[data.target != -1].reset_index(drop=True)
    test = data[data.target == -1].reset_index(drop=True)

train.to_csv(config.DATA_DIR + 'uni_enc_train.csv')
test.to_csv(config.DATA_DIR + 'uni_enc_test.csv')