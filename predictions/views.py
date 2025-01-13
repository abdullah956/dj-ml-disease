import pandas as pd
import joblib
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

# Load the pre-trained model and related resources
model = joblib.load('disease_prediction_model.pkl')
le = joblib.load('label_encoder.pkl')
training_data = pd.read_csv('dataset.csv')
original_columns = pd.get_dummies(training_data[['Symptom_1', 'Symptom_2', 'Symptom_3']]).columns.tolist()

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '').strip()

            if not message:
                return JsonResponse({'response': "Please provide symptoms to proceed."})

            symptoms = message.split(',')  # Assuming symptoms are comma-separated
            symptoms_dict = {
                'Symptom_1': symptoms[0] if len(symptoms) > 0 else 'no_symptom',
                'Symptom_2': symptoms[1] if len(symptoms) > 1 else 'no_symptom',
                'Symptom_3': symptoms[2] if len(symptoms) > 2 else 'no_symptom',
            }

            input_data = pd.DataFrame([symptoms_dict])
            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=original_columns, fill_value=0)

            predicted_disease = model.predict(input_encoded)
            predicted_disease_name = le.inverse_transform(predicted_disease)[0]

            return JsonResponse({'response': f"The predicted disease is: {predicted_disease_name}"})
        except Exception as e:
            return JsonResponse({'response': f"Error: {str(e)}"})
    return render(request, 'predict.html')
