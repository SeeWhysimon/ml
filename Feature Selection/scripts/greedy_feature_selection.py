import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class GreedyFeatureSelection:
    def evaluate_score(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        auc = r2_score(y, predictions)
        return auc
    
    def _feature_selection(self, X: np.array, y: np.array):
        good_features = []
        best_scores = []

        num_features = X.shape[1]
        while True:
            this_feature = None
            best_score = 0

            for feature in range(num_features):
                if feature in good_features:
                    continue
                
                selected_features = good_features + [feature]
                x_train = X[:, selected_features]
                
                score = self.evaluate_score(x_train, y)
                
                if score > best_score:
                    this_feature = feature
                    best_score = score
                
                if this_feature is not None:
                    good_features.append(this_feature)
                    best_scores.append(best_score)
                
                if len(best_scores) > 2:
                    if best_scores[-1] < best_scores[-2]:
                        break
        
        return best_scores, good_features
    
    def __call__(self, X: np.array, y: np.array):
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores, features

if __name__ == '__main__':
    df = pd.read_csv('../data/original_data.csv')

    target = df.columns[-1]
    y = df[target].values
    X = df.drop(columns=[target, 'sample_id']).values
    feature_names = df.drop(columns=[target, 'sample_id']).columns

    gfs = GreedyFeatureSelection()
    X_transformed, scores, selected_features = gfs(X, y)
    
    df_transformed = pd.DataFrame(X_transformed, columns=feature_names[selected_features])
    df_transformed[target] = y
    
    df_transformed.to_csv('../data/greedy_feature_selection.csv', index=False)
    
    print("Transformed features saved to 'selected_features_data.csv'.")
