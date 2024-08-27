import numpy as np

from functools import partial
from scipy.optimize import fmin
from sklearn import metrics

class OptimizeAUC:
    def __init__(self):
        self.coef_ = None
    
    def _auc(self, coef, X, y):
        predictions = np.dot(X, coef)
        auc_score = metrics.roc_auc_score(y, predictions)
        return -1 * auc_score
    
    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)

        # Initialize coefficients
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1).flatten()
        
        # Optimize
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)
    
    def predict(self, X):
        return np.dot(X, self.coef_)
    
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn import ensemble, linear_model, metrics, model_selection

if __name__ == "__main__":
    X, y = make_classification(n_samples=10000, n_features=25)
    xfold1, xfold2, yfold1, yfold2 = model_selection.train_test_split(X, y, test_size=0.5, stratify=y)
    
    logreg = linear_model.LogisticRegression()
    rf = ensemble.RandomForestClassifier()
    xgbc = xgb.XGBClassifier()

    logreg.fit(xfold1, yfold1)
    rf.fit(xfold1, yfold1)
    xgbc.fit(xfold1, yfold1)

    logreg_pred = logreg.predict_proba(xfold2)[:, 1]
    rf_pred = logreg.predict_proba(xfold2)[:, 1]
    xgbc_pred = xgbc.predict_proba(xfold2)[:, 1]

    avg_pred = (logreg_pred + rf_pred + xgbc_pred) / 3

    fold2_preds = np.column_stack((logreg_pred, rf_pred, xgbc_pred, avg_pred))

    aucs_fold2 = []
    for i in range(fold2_preds.shape[1]):
        auc = metrics.roc_auc_score(yfold2, fold2_preds[:, i])
        aucs_fold2.append(auc)

    print(f"Fold-2: LR AUC = {aucs_fold2[0]}") 
    print(f"Fold-2: RF AUC = {aucs_fold2[1]}") 
    print(f"Fold-2: XGB AUC = {aucs_fold2[2]}") 
    print(f"Fold-2: Average Pred AUC = {aucs_fold2[3]}")

    logreg = linear_model.LogisticRegression()
    rf = ensemble.RandomForestClassifier()
    xgbc = xgb.XGBClassifier()

    logreg.fit(xfold2, yfold2)
    rf.fit(xfold2, yfold2)
    xgbc.fit(xfold2, yfold2)

    logreg_pred = logreg.predict_proba(xfold1)[:, 1]
    rf_pred = logreg.predict_proba(xfold1)[:, 1]
    xgbc_pred = xgbc.predict_proba(xfold1)[:, 1]

    avg_pred = (logreg_pred + rf_pred + xgbc_pred) / 3

    fold1_preds = np.column_stack((logreg_pred, rf_pred, xgbc_pred, avg_pred))

    aucs_fold1 = []
    for i in range(fold1_preds.shape[1]):
        auc = metrics.roc_auc_score(yfold1, fold1_preds[:, i])
        aucs_fold1.append(auc)

    print(f"Fold-1: LR AUC = {aucs_fold1[0]}") 
    print(f"Fold-1: RF AUC = {aucs_fold1[1]}") 
    print(f"Fold-1: XGB AUC = {aucs_fold1[2]}") 
    print(f"Fold-1: Average Pred AUC = {aucs_fold1[3]}")

    opt = OptimizeAUC()
    opt.fit(fold1_preds[:, :-1], yfold1)
    opt_preds_fold2 = opt.predict(fold2_preds[:, :-1])
    auc = metrics.roc_auc_score(yfold2, opt_preds_fold2) 
    print(f"Optimized AUC, Fold 2 = {auc}") 
    print(f"Coefficients = {opt.coef_}")

    opt = OptimizeAUC()
    opt.fit(fold2_preds[:, :-1], yfold2)
    opt_preds_fold1 = opt.predict(fold1_preds[:, :-1])
    auc = metrics.roc_auc_score(yfold1, opt_preds_fold1) 
    print(f"Optimized AUC, Fold 1 = {auc}") 
    print(f"Coefficients = {opt.coef_}")