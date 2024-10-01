from django import forms

class SymptomForm(forms.Form):
    symptom_1 = forms.CharField(label='Symptom 1', max_length=100)
    symptom_2 = forms.CharField(label='Symptom 2', max_length=100)
    symptom_3 = forms.CharField(label='Symptom 3', max_length=100)
