import pandas as pd
import joblib
from django.shortcuts import render

def predict_disease(request):
    if request.method == 'POST':
        symptom_1 = request.POST.get('symptom_1', 'no_symptom')
        symptom_2 = request.POST.get('symptom_2', 'no_symptom')
        symptom_3 = request.POST.get('symptom_3', 'no_symptom')

        model = joblib.load('disease_prediction_model.pkl')
        le = joblib.load('label_encoder.pkl')

        input_symptoms = {
            'Symptom_1': symptom_1,
            'Symptom_2': symptom_2,
            'Symptom_3': symptom_3
        }

        input_data = pd.DataFrame([input_symptoms])
        input_encoded = pd.get_dummies(input_data)

        # Load the original model's feature columns and one-hot encode the input accordingly
        training_data = pd.read_csv('dataset.csv')
        original_columns = pd.get_dummies(training_data[['Symptom_1', 'Symptom_2', 'Symptom_3']]).columns.tolist()
        input_encoded = input_encoded.reindex(columns=original_columns, fill_value=0)

        predicted_disease = model.predict(input_encoded)
        predicted_disease_name = le.inverse_transform(predicted_disease)

        return render(request, 'result.html', {'predicted_disease': predicted_disease_name[0]})

    return render(request, 'disease_prediction.html')
