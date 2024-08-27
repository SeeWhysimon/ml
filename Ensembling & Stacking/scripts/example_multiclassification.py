import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model3 = SVC(probability=True, random_state=42)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

prob1 = model1.predict_proba(X_test)
prob2 = model2.predict_proba(X_test)
prob3 = model3.predict_proba(X_test)

combined_probabilities = (prob1 + prob2 + prob3) / 3

predicted_classes = np.argmax(combined_probabilities, axis=1)

loss = log_loss(y_test, combined_probabilities)

print("Predicted Classes:", predicted_classes)
print("True Classes:     ", y_test)
print("Log-loss:         ", loss)
