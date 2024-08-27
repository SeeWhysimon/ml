import pandas as pd

from sklearn.feature_selection import chi2,\
                                      f_classif,\
                                      f_regression,\
                                      mutual_info_classif,\
                                      mutual_info_regression,\
                                      SelectKBest,\
                                      SelectPercentile

class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        if problem_type == 'classification':
            valid_scoring = {
                'f_classif': f_classif,
                'chi2': chi2,
                'mutual_info_classif': mutual_info_classif
            }
        elif problem_type == 'regression':
            valid_scoring = {
                'f_regression': f_regression,
                'mutual_info_regression': mutual_info_regression
            }
        
        if scoring not in valid_scoring:
            raise Exception('Invalid scoring function')
        
        # Use SelectKBest if n_features is int
        # Use SelectPercentile if n_features is float
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=(n_features * 100)
            )
        else:
            raise Exception('Invalid type of feature')
    
    def fit(self, X, y):
        return self.selection.fit(X, y)
    
    def transform(self, X):
        return self.selection.transform(X)
    
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)
    
if __name__ == '__main__':
    df = pd.read_csv('../../data/Feature Selection/original_data.csv')

    # Suppose the last column is the target
    target = df.columns[-1]
    y = df[target].values

    features = df.columns.drop([target, 'sample_id'])
    X = df[features].values

    ufs = UnivariateFeatureSelection(
        n_features=0.5, # Select top 5 features
        problem_type='regression',
        scoring='f_regression'
    )

    X_transformed = ufs.fit_transform(X, y)
    
    selected_features = features[ufs.selection.get_support()]
    dropped_features = features[~ufs.selection.get_support()]

    print(selected_features)
    
    df_transformed = pd.DataFrame(data={
        'sample_id': df['sample_id'].values,
        **{feature: X_transformed[:, i] for i, feature in enumerate(selected_features)},
        target: y
    })
    
    df_transformed.to_csv('../../data/Feature Selection/univariate_feature_selection.csv', index=False)