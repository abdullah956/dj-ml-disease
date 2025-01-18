from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from .forms import SignupForm, LoginForm

def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Signup successful! Please log in.")
            return redirect('login')
    else:
        form = SignupForm()
    return render(request, 'accounts/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(data=request.POST)
        if form.is_valid():
            user = authenticate(
                username=form.cleaned_data['username'],
                password=form.cleaned_data['password']
            )
            if user is not None:
                login(request, user)
                messages.success(request, "Login successful!")
                return redirect('/')
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, "Logged out successfully!")
    return redirect('home')


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import PatientHistory
from .forms import PatientHistoryForm
from django.contrib import messages

@login_required
def patient_history_view(request):
    try:
        history = PatientHistory.objects.get(user=request.user)
    except PatientHistory.DoesNotExist:
        history = None

    if request.method == 'POST':
        form = PatientHistoryForm(request.POST, instance=history)
        if form.is_valid():
            patient_history = form.save(commit=False)
            patient_history.user = request.user
            patient_history.save()
            messages.success(request, "Your medical history has been saved.")
            return redirect('home')
    else:
        form = PatientHistoryForm(instance=history)

    return render(request, 'accounts/patient_history.html', {'form': form})


# views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import ContactForm

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Create a new contact message object
            contact_message = form.save(commit=False)
            contact_message.save()  # Save it to the database
            messages.success(request, "Your message has been sent successfully!")
            return redirect('contact')
        else:
            messages.error(request, "Something went wrong. Please try again.")
            print(form.errors)  # Optional: Print errors for debugging
    else:
        form = ContactForm()
    
    return render(request, 'contact.html', {'form': form})