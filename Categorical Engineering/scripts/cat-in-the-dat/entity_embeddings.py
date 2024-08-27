import os
import gc
import joblib
import config
import pandas as pd
import numpy as np

from sklearn import metrics, preprocessing
from keras import layers, optimizers, callbacks, utils
from keras import models
from keras import backend as K

def creat_model(data: pd.DataFrame, catcols: pd.DataFrame.columns):
    inputs = []
    outputs = []
    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil(num_unique_values/2), 50))
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(num_unique_values + 1,
                               embed_dim,
                               name=c)(inp)
        out = layers.SpatialDropout1D(0.3)(out)
        out = layers.Reshape(target_shape=(embed_dim,))(out)
        inputs.append(inp)
        outputs.append(out)
    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=y)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def run(fold):
    df = pd.read_csv(config.DATA_DIR + 'preprocessed_train.csv')
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df[feat] = lbl_enc.fit_transform(df[feat].values)
    
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    model = creat_model(df, features)
    
    x_train = [df_train[features].values[:, k] for k in range(len(features))]
    x_valid = [df_valid[features].values[:, k] for k in range(len(features))]
    y_train = df_train['target'].values
    y_valid = df_valid['target'].values
    y_train_cat = utils.to_categorical(y_train)
    y_valid_cat = utils.to_categorical(y_valid)

    model.fit(x_train, 
              y_train_cat,
              validation_data=(x_valid, y_valid_cat),
              verbose=1,
              batch_size=1024,
              epochs=3)
    valid_preds = model.predict(x_valid)[:, 1]
    print(metrics.roc_auc_score(y_valid, valid_preds))
    K.clear_session()

if __name__ == '__main__':
    for fold in range(config.NUM_FOLD):
        run(fold)