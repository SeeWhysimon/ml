import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

estimators = [
    ('svm', SVC(probability=True, kernel='rbf', C=1.0)),
    ('dt', DecisionTreeClassifier(max_depth=5))
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

svc = SVC(probability=True, kernel='rbf', C=1.0)
dt = DecisionTreeClassifier(max_depth=5)
lr = LogisticRegression()

stacking_clf.fit(X_train, y_train)
svc.fit(X_train, y_train)
dt.fit(X_train, y_train)
lr.fit(X_train, y_train)

stack_y_pred = stacking_clf.predict(X_test)
svc_y_pred = svc.predict(X_test)
dt_y_pred = dt.predict(X_test)
lr_y_pred = lr.predict(X_test)

stack_accuracy = accuracy_score(y_test, stack_y_pred)
svc_accuracy = accuracy_score(y_test, svc_y_pred)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
lr_accuracy = accuracy_score(y_test, lr_y_pred)

print(f"Stacking model accuracy: {stack_accuracy}")
print(f"SVC accuracy: {svc_accuracy}")
print(f"Decision Tree accuracy: {dt_accuracy}")
print(f"Linear Regressor accuracy: {lr_accuracy}")