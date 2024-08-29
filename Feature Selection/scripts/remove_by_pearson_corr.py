import pandas as pd
import numpy as np

def get_corr_matrix(df: pd.DataFrame, 
                    verbose: bool = False):
    corr_matrix = df.corr()
    
    if verbose:
        print('Pearson Correlation Matrix:')
        print(corr_matrix)

    return corr_matrix

def remove_by_preason_corr(df: pd.DataFrame, 
                           threshold: float = 0.1, 
                           verbose: bool = False):
    corr_matrix = get_corr_matrix(df)

    # Select the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    if verbose:
        print('Columns dropped:')
        print(to_drop)

    df_reduced = df.drop(columns=to_drop)
    return df_reduced

if __name__ == '__main__':
    ori_df = pd.read_csv('../data/original_data.csv')
    reduced_df = ori_df.drop(columns=['sample_id'])
    
    corr_matrix = get_corr_matrix(reduced_df, verbose=True)

    threshold = 0.5
    
    reduced_df = remove_by_preason_corr(reduced_df, threshold=threshold, verbose=True)
    reduced_df.insert(0, 'sample_id', ori_df['sample_id'])
    reduced_df.to_csv('../data/preason_corr_thresh_0.5.csv', index=False)