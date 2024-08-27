from tsfresh.feature_extraction import feature_calculators as fc
import pandas as pd

df = pd.read_csv('../../data/Feature Engineering/customer_transactions.csv')

result_df = pd.DataFrame()

result_df['customer_id'] = df['customer_id']

result_df['abs_energy'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: fc.abs_energy(x.values))
result_df['count_above_mean'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: fc.count_above_mean(x.values))
result_df['count_below_mean'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: fc.count_below_mean(x.values))
result_df['mean_abs_change'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: fc.mean_abs_change(x.values))
result_df['mean_change'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: fc.mean_change(x.values))

result_df = result_df.drop_duplicates(subset='customer_id')

result_df.to_csv('../../data/Feature Engineering/tsfresh_feature_engineering.csv', index=False)

print("Feature engineering with tsfresh complete. Processed data saved to: '../../data/Feature Engineering/feat_eng_tsfresh.csv'")