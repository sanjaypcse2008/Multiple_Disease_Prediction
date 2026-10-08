import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load improved dataset
data = pd.read_csv("dataset/improved_disease_dataset.csv")

# Separate features and target
X = data.drop("Disease", axis=1)
y = data["Disease"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("===================================")
print("Multiple Disease Prediction Model")
print("===================================")
print("Model trained successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of diseases:", y.nunique())
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")