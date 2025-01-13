from django.urls import path
from . import views

urlpatterns = [
    #path('predict/', views.predict_disease, name='predict'),
    path('predict/', views.chatbot, name='chatbot'),
]
