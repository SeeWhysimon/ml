import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Base classifier: decition tree
base_clf = DecisionTreeClassifier(max_depth=3)

# Initialize bagging classifier
bagging_clf = BaggingClassifier(estimator=base_clf, n_estimators=50, random_state=42)

bagging_clf.fit(X_train, y_train)

y_pred = bagging_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging classifier accuracy: {accuracy:.2f}")

base_clf.fit(X_train, y_train)
y_base_pred = base_clf.predict(X_test)
base_accuracy = accuracy_score(y_test, y_base_pred)
print(f"Single decision tree accuracy: {base_accuracy:.2f}")
