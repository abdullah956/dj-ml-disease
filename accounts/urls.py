from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('patient-history/', views.patient_history_view, name='patient_history'),
    path('contact/', views.contact, name='contact'),
]
