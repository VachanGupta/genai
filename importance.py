# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Train Decision Tree on all features
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predictions and accuracy
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

accuracy

# Feature importances
importances = classifier.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

# Store feature importances
feature_importances = {}

for i in sorted_indices:
    feature_importances[cancer.feature_names[i]] = float(importances[i])

json.dumps(feature_importances)

# ---- Train model with top 2 features ----

top2_indices = sorted_indices[:2]

X_train_top2 = X_train[:, top2_indices]
X_test_top2 = X_test[:, top2_indices]

classifier_top2 = DecisionTreeClassifier(random_state=42)
classifier_top2.fit(X_train_top2, y_train)

predictions_top2 = classifier_top2.predict(X_test_top2)
accuracy_top2 = accuracy_score(y_test, predictions_top2)

print("Accuracy with all features:", accuracy)
print("Accuracy with top 2 features:", accuracy_top2)