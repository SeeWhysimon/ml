import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('../data/customer_transactions.csv')

poly_feat_df = pd.DataFrame()
poly_feat_df['customer_id'] = df['customer_id']

pf = preprocessing.PolynomialFeatures(degree=2, 
                                      interaction_only=False, 
                                      include_bias=False)

ori_features = ['transaction_amount', 'transaction_quantity']

poly_features = pf.fit_transform(df[ori_features])

feature_names = pf.get_feature_names_out(['transaction_amount', 'transaction_quantity'])

new_feature_names = feature_names[2:]
new_poly_features = poly_features[:, 2:]
poly_feat_df[new_feature_names] = new_poly_features

poly_feat_df.to_csv('../data/polynomial_feature_engineering.csv', index=False)