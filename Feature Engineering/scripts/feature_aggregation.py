import pandas as pd
import numpy as np

file_path = '../../data/Feature Engineering/customer_transactions.csv'
df = pd.read_csv(file_path)

aggregated_features = df.groupby('customer_id', as_index=False).agg(
    total_transaction_amount=('transaction_amount', 'sum'),
    average_discount_amount=('discount_amount', 'mean'),
    total_quantity=('transaction_quantity', 'sum'),
    average_transaction_duration=('transaction_duration', 'mean'),
    transaction_count=('transaction_amount', 'count')
)

aggregated_features['percentile_10'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: np.percentile(x, 10))
aggregated_features['percentile_60'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: np.percentile(x, 60))
aggregated_features['percentile_90'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: np.percentile(x, 90))

aggregated_features = aggregated_features.drop_duplicates(subset='customer_id')

output_file_path = '../../data/Feature Engineering/aggregated_features.csv'
aggregated_features.to_csv(output_file_path, index=False)

print("Feature engineering complete. Processed data saved to:", output_file_path)