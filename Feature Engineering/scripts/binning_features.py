import pandas as pd

df = pd.read_csv('../../data/Feature Engineering/customer_transactions.csv')
new_df = pd.DataFrame()

new_df['customer_id'] = df['customer_id']

# Bin the 'transaction_amount' into 10 equal-width bins, label them with integers from 0 to 9
new_df['transaction_amount_bin_10'] = pd.cut(df['transaction_amount'], bins=10, labels=False)

new_df['transaction_amount_bin_100'] = pd.cut(df['transaction_amount'], bins=100, labels=False)

new_df.to_csv('../../data/Feature Engineering/binning_features.csv', index=False)