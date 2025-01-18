import pandas as pd
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Load Resources
model = joblib.load('disease_prediction_model.pkl')
le = joblib.load('label_encoder.pkl')
model_features = joblib.load('model_features.pkl')

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symptoms = [symptom.strip().lower() for symptom in data.get('message', '').split(',')]

            # Check if there are fewer than 3 symptoms
            if len(symptoms) < 3:
                return JsonResponse({'response': "Please provide at least three symptoms for better prediction."})

            # Map symptoms to all feature columns
            input_dict = {feature: 0 for feature in model_features}
            for symptom in symptoms:
                matched = False
                for feature in model_features:
                    if symptom in feature.lower():
                        input_dict[feature] = 1
                        matched = True
                if not matched:
                    return JsonResponse({'response': f"Symptom '{symptom}' not recognized. Please check spelling."})

            # Prepare input DataFrame
            input_data = pd.DataFrame([input_dict])

            # Predict the disease
            prediction = model.predict(input_data)
            disease = le.inverse_transform(prediction)[0]

            # Avoid "acne" prediction if it's not appropriate
            if disease.lower() == 'acne':
                disease = "The prediction is inconclusive. Please try different symptoms."

            return JsonResponse({'response': f"The predicted disease is: {disease}"})
        except Exception as e:
            return JsonResponse({'response': f"Error: {str(e)}"})

    return render(request, 'predict.html')



def home(request):
    return render(request, 'index.html')

def contact(request):
    return render(request, 'contact.html')

def about(request):
    return render(request, 'about-us.html')