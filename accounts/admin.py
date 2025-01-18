from django.contrib import admin

# Register your models here.
from .models import PatientHistory , Contact

admin.site.register(PatientHistory)
admin.site.register(Contact)