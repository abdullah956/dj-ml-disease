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

            # Get the prediction probabilities
            prediction_proba = model.predict_proba(input_data)[0]
            proba_dict = {disease: prob for disease, prob in zip(le.classes_, prediction_proba)}

            # Sort diseases by probability
            sorted_probabilities = sorted(proba_dict.items(), key=lambda item: item[1], reverse=True)
            top_three = sorted_probabilities[:3]

            # Constructing the response message
            top_three_diseases = [f"{disease}: {prob*100:.2f}%" for disease, prob in top_three]
            disease_suggestion = f"Predicted disease is {disease}. Please see a {disease} specialist."
            probability_message = f"Top 3 predictions: {', '.join(top_three_diseases)}"

            # Returning the response in the desired format
            return JsonResponse({
                'response': f"{disease_suggestion}\n{probability_message}"
            })
        except Exception as e:
            return JsonResponse({'response': f"Error: {str(e)}"})

    return render(request, 'predict.html')





def home(request):
    return render(request, 'index.html')

def contact(request):
    return render(request, 'contact.html')

def about(request):
    return render(request, 'about-us.html')