from django.db import models
from django.contrib.auth.models import User

class PatientHistory(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='history')
    medical_history = models.TextField()

    def __str__(self):
        return f"{self.user.username}'s History"



class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=255)
    message = models.TextField()

    def __str__(self):
        return f"Message from {self.name} ({self.email})"
