import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test Model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save Model and Resources
joblib.dump(model, 'disease_prediction_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(list(all_features), 'model_features.pkl')
