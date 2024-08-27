import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv('../../data/Feature Selection/original_data.csv')

# Suppose the last column is the target
target = df.columns[-1]
y = df[target].values

features = df.columns.drop([target, 'sample_id'])
X = df[features].values

model = RandomForestRegressor()

sfm = SelectFromModel(estimator=model, threshold=0.1)
X_transformed = sfm.fit_transform(X, y)

support = sfm.get_support()

print([
    x for x, y in zip(features, support) if y == True
])