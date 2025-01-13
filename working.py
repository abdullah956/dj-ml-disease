import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the Dataset
# Replace 'dataset.csv' with the path to your dataset file
data = pd.read_csv('dataset.csv')

# 2. Data Preprocessing

# Fill NaN values in symptom columns with 'no_symptom'
symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3']  # Use only 3 symptoms
data[symptom_columns] = data[symptom_columns].fillna('no_symptom')

# Encode the Disease column to numerical values
le = LabelEncoder()
data['Disease'] = le.fit_transform(data['Disease'])

# 3. Feature Engineering

# Extract symptom features and target variable
X = data[symptom_columns]
y = data['Disease']

# One-Hot Encode the symptom features
X_encoded = pd.get_dummies(X, columns=symptom_columns)

# 4. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 5. Initialize and Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 7. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 8. Save the Trained Model and Label Encoder for Future Use
joblib.dump(model, 'disease_prediction_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model and label encoder have been saved successfully.")

# 9. Take Symptom Inputs from the User and Make a Prediction
def take_input_and_predict():
    # Take symptom inputs
    input_symptoms = {
        'Symptom_1': input("Enter Symptom 1: "),
        'Symptom_2': input("Enter Symptom 2: "),
        'Symptom_3': input("Enter Symptom 3: ")
    }
    
    # Convert input to a DataFrame and one-hot encode
    input_data = pd.DataFrame([input_symptoms])
    input_encoded = pd.get_dummies(input_data)

    # Align the input data with the model's training data (to match the columns)
    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Predict the disease
    predicted_disease = model.predict(input_encoded)

    # Convert the predicted label back to the disease name
    predicted_disease_name = le.inverse_transform(predicted_disease)

    print(f"The predicted disease is: {predicted_disease_name[0]}")

# Call the function to take input and predict the disease
take_input_and_predict()
