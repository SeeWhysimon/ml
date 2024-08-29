import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('../data/original_data.csv')

# Suppose the last column is the target
target = df.columns[-1]
y = df[target].values

features = df.columns.drop([target, 'sample_id'])
X = df[features].values

model = RandomForestRegressor()

model.fit(X, y)

importances = model.feature_importances_
idxs = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(idxs)), importances[idxs], align='center')
plt.yticks(range(len(idxs)), [features[i] for i in idxs])
plt.xlabel('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

threshold = 0.1
selecete_features = features[importances > threshold]

df_transformed = df[['sample_id'] + list(selecete_features) + [target]]

df_transformed.to_csv('../data/select_by_coeffs_importance.csv', index=False)