# Split Dataset into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    df[['Temperature','Humidity']],
    df['Snake_Presence'],
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Get probability estimates
y_prob = model.predict_proba(X_test)[:,1]