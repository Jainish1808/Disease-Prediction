from django.db import models

class HeartDisease(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    sex = models.IntegerField()
    cp = models.IntegerField()
    trestbps = models.IntegerField()
    chol = models.IntegerField()
    fbs = models.IntegerField()
    restecg = models.IntegerField()
    thalach = models.IntegerField()
    exang = models.IntegerField()
    oldpeak = models.FloatField()
    slope = models.IntegerField()
    ca = models.IntegerField()
    thal = models.IntegerField()
    prediction = models.CharField(max_length=3)
    
    def __str__(self):
        return f"Patient: Name={self.name}, Age={self.age}, Sex={self.sex}, CP={self.cp}"

class StrokePrediction(models.Model):
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()  # Age of the user
    gender = models.BooleanField()  # True for Male (1), False for Female (0)
    hypertension = models.BooleanField()  # True for Yes (1), False for No (0)
    heart_disease = models.BooleanField()  # True for Yes (1), False for No (0)
    ever_married = models.BooleanField()  # True for Yes (1), False for No (0)
    work_type = models.IntegerField()  # e.g., 0 = Govt, 1 = Private
    Residence_type = models.BooleanField()  # True for Urban (1), False for Rural (0)
    avg_glucose_level = models.FloatField()  # Average glucose level
    bmi = models.FloatField()  # Body Mass Index
    smoking_status = models.IntegerField()  # e.g., 0 = Never, 1 = Smoker
    prediction = models.CharField(max_length=3)

    def __str__(self):
        return f"Patient: Name={self.name}, Age={self.age}, Sex={self.gender}"
