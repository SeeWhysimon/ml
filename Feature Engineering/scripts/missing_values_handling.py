import numpy as np
from sklearn import impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

# Random numpy array with 10 samples and 6 features, values ranging from 1 to 15 
X = np.random.randint(1, 15, (10, 6))

X = X.astype(float)

# Randomly assign 10 elements to NaN (missing)
X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan

# Use linear regressor to fill nan values
regression_imputer = impute.IterativeImputer(estimator=LinearRegression(),
                                             max_iter=15,
                                             tol=1e-4,
                                             imputation_order='random',
                                             random_state=42,
                                             n_nearest_features=4,
                                             initial_strategy='median')
regression_filled_X = regression_imputer.fit_transform(X)

# Use 2 nearest neighbours to fill nan values
knn_imputer = impute.KNNImputer(n_neighbors=2)
knn_filled_X = knn_imputer.fit_transform(X)

# Use mean values to fill nan values
mean_imputer = impute.SimpleImputer(strategy='mean')
mean_filled_X = mean_imputer.fit_transform(X)

print(f'Original: \n{X}')
print(f'Regression filled with parameters: \n{regression_filled_X}')
print(f'KNN filled: \n{knn_filled_X}')
print(f'Mean filled: \n{mean_filled_X}')