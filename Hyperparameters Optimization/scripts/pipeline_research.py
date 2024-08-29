import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data = pd.read_csv('../data/mobile phone price/train.csv')

X = data.drop('price_range', axis=1)
y = data['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline that includes data scaling and a classifier (initially set to SVC)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the data
    ('clf', SVC())  # Step 2: Use Support Vector Classifier (SVC) as the initial model
])

# Define a parameter grid for searching the best hyperparameters
param_grid = [
    {
        'clf': [SVC()],  # First model: Support Vector Classifier
        'clf__C': [0.1, 1, 10],  # C parameter values to test
        'clf__gamma': [0.001, 0.01, 0.1],  # Gamma parameter values to test
        'clf__kernel': ['rbf', 'linear']  # Different kernel types to test
    },
    {
        'clf': [RandomForestClassifier()],  # Second model: Random Forest Classifier
        'clf__n_estimators': [10, 50, 100],  # Number of trees in the forest
        'clf__max_depth': [None, 10, 20]  # Maximum depth of the tree
    }
]

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))