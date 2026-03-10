data_encoded = pd.get_dummies(data, drop_first=True)
X = data_encoded.drop("LoanApproval", axis=1)
y = data_encoded["LoanApproval"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)