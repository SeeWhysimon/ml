import numpy as np
import pandas as pd

def generate_data(num_samples: int, means: list, vars: list):
    """
    Generates a DataFrame with specified mean and variance for each column.

    Args:
        num_samples (int): The number of samples (rows) to generate.
        means (list): A list of means for each column of the DataFrame.
        vars (list): A list of variances for each column of the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the generated data, 
                    where the first column is 'sample_id' and each subsequent 
                    column contains normally distributed data with specified mean and variance.
    
    Raises:
        ValueError: If the lengths of `means` and `vars` do not match.
        """
    # Check the length of both means and vars
    if len(means) != len(vars):
        raise Exception('Parameter list length mismatch!')
    
    sample_id = np.arange(1, num_samples + 1)
    data = {'sample_id': sample_id}

    for i in range(len(means)):
        col_name = f'feat_{i+1}'
        data[col_name] = np.random.normal(means[i], np.sqrt(vars[i]), num_samples)
    
    df = pd.DataFrame(data)
    return df
    
if __name__ == '__main__':
    num_samples = 5000
    num_features = 10
    means = np.random.uniform(-5, 5, size=num_features).tolist()
    vars = np.random.uniform(0, 20, size=num_features).tolist()
    
    df = generate_data(num_samples, means, vars)
    
    # Random operation on features
    df['feat_6'] = df['feat_1'].values * 2 + 3
    df['feat_4'] = np.exp(df['feat_1'].values)
    df['feat_8'] = df['feat_3'].values * 7 - 10

    df.to_csv('../../data/Feature Selection/original_data.csv', index=False)