import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate random customer IDs
customer_ids = [f'C{str(i).zfill(4)}' for i in range(1, 101)]  # 100 customers

# Define the time range for generating transactions
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 8, 15)

# Generate random transaction records
transactions = []
for _ in range(10000):  # Generate 10,000 transaction records
    customer_id = random.choice(customer_ids)
    transaction_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
    transaction_amount = round(np.random.uniform(10, 1000), 2)  # Random transaction amount between 10 and 1000
    discount_amount = round(transaction_amount * np.random.uniform(0, 0.3), 2)  # Random discount amount between 0% and 30%
    transaction_quantity = np.random.randint(1, 10)  # Random transaction quantity between 1 and 10
    transaction_duration = round(np.random.uniform(5, 60), 2)  # Transaction duration between 5 and 60 minutes
    transaction_type = random.choice(['purchase', 'refund', 'subscription'])  # Transaction type
    transactions.append([
        customer_id,
        transaction_date,
        transaction_amount,
        discount_amount,
        transaction_quantity,
        transaction_duration,
        transaction_type
    ])

# Convert the transactions list to a DataFrame
df = pd.DataFrame(transactions, columns=[
    'customer_id', 
    'transaction_date', 
    'transaction_amount', 
    'discount_amount', 
    'transaction_quantity', 
    'transaction_duration', 
    'transaction_type'
])

# Sort the DataFrame by transaction date
df.sort_values(by='transaction_date', inplace=True)

# Display the first few rows of the DataFrame
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('../../data/Feature Engineering/customer_transactions.csv', index=False)