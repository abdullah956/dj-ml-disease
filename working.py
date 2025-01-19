import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np
import joblib

# Load the Dataset
data = pd.read_csv('dataset.csv')

# Data Preprocessing
symptom_columns = [col for col in data.columns if col.startswith('Symptom')]
data[symptom_columns] = data[symptom_columns].fillna('no_symptom')

# Encode Diseases
le = LabelEncoder()
data['Disease'] = le.fit_transform(data['Disease'])

# One-Hot Encode the symptom features
X = pd.get_dummies(data[symptom_columns])
y = data['Disease']

# Save all feature columns
all_features = X.columns

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest for Classification
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test Model
y_pred = model.predict(X_test)

# Classification Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Classification Metrics:")
print(f"Accuracy Score: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# For regression, assuming 'Disease' as a numerical target
# Regression Metrics
y_pred_regression = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_regression)
mse = mean_squared_error(y_test, y_pred_regression)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_regression)
cv_scores = cross_val_score(model, X, y, cv=5)

print("\nRegression Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-Squared (R²) Score: {r2:.2f}")
print(f"Cross-Validation Score: {cv_scores.mean():.2f}")

# # Save Model and Resources
# joblib.dump(model, 'disease_prediction_model.pkl')
# joblib.dump(le, 'label_encoder.pkl')
# joblib.dump(list(all_features), 'model_features.pkl')
