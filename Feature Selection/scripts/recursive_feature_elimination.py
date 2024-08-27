import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../../data/Feature Selection/original_data.csv')

# Suppose the last column is the target
target = df.columns[-1]
y = df[target].values

features = df.drop(columns=['sample_id', target])
X = features.values

model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=3)

X_transformed = rfe.fit_transform(X, y)

selected_features = features.columns[rfe.get_support()]

df_transformed = pd.DataFrame(data={
    'sample_id': df['sample_id'],
    **{feature: X_transformed[:, i] for i, feature in enumerate(selected_features)},
    target: y
})

df_transformed.to_csv('../../data/Feature Selection/recursive_feature_elimination.csv', index=False)