import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def remove_by_var(df: pd.DataFrame,
                  threshold: float,
                  verbose: bool = False) -> pd.DataFrame:
    """
    Removes features from the DataFrame that have a variance lower than the specified threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The variance threshold below which features are removed.
        verbose (bool): If True, prints the names of removed features.

    Returns:
        pd.DataFrame: The DataFrame with low variance features removed.
    """
    # Select data with variance > threshold
    var_thresh = VarianceThreshold(threshold=threshold)
    reduced_data = var_thresh.fit_transform(df)
    
    # Get the features that were kept
    selected_features = df.columns[var_thresh.get_support()]
    
    # Get the features that were removed
    removed_features = df.columns[~var_thresh.get_support()]
    
    # Create a DataFrame from the reduced data
    reduced_df = pd.DataFrame(reduced_data, columns=selected_features)

    if verbose:
        print('Removed features:')
        print(removed_features)

    return reduced_df

if __name__ == '__main__':
    ori_data = pd.read_csv('../../data/Feature Selection/original_data.csv')

    threshold = 5

    transformed_df = remove_by_var(ori_data, threshold=threshold, verbose=True)

    transformed_df.to_csv('../../data/Feature Selection/var_thresh_0.5.csv', index=False)
