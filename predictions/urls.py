from django.urls import path
from . import views

urlpatterns = [
    #path('predict/', views.predict_disease, name='predict'),
    path('', views.home, name='home'),  # Home page
    path('contact/', views.contact, name='contact'),  # Contact page
    path('about/', views.about, name='about'),  # About Us page
    path('predict/', views.chatbot, name='chatbot'),
]
