import numpy as np
import pandas as pd

df = pd.read_csv('../data/customer_transactions.csv')
new_df = pd.DataFrame()

new_df['customer_id'] = df['customer_id']
new_df['log_transaction_amount'] = df['transaction_amount'].apply(np.log1p)
new_df['log_transaction_amount_bin_10'] = pd.cut(new_df['log_transaction_amount'], 
                                                 bins=10,
                                                 labels=False)

ori_var = df['transaction_amount'].var()
new_var = new_df['log_transaction_amount'].var()

print(f'Original variance: {ori_var}')
print(f'New variance after log: {new_var}')

new_df.to_csv('../data/log_transformation.csv', index=False)