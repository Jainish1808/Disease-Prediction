from django.contrib import admin
from .models import HeartDisease
from .models import StrokePrediction

# Register your models here.
admin.site.register(HeartDisease)
admin.site.register(StrokePrediction)