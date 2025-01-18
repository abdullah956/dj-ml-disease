from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

class SignupForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class LoginForm(AuthenticationForm):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)

from django import forms
from .models import PatientHistory

class PatientHistoryForm(forms.ModelForm):
    class Meta:
        model = PatientHistory
        fields = ['medical_history']
        widgets = {
            'medical_history': forms.Textarea(attrs={'rows': 5}),
        }

from .models import Contact

class ContactForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = ['name', 'email', 'subject', 'message']
