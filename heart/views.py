import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .models import HeartDisease
from .models import StrokePrediction

# Load the dataset
heart = pd.read_csv(r"C:\Users\jaini\IntellijIdea\Jainish PYTHON\Python_Project\Disease_Prediction\heart\templates\heart.csv")
stroke = pd.read_csv(r"C:\Users\jaini\IntellijIdea\Jainish PYTHON\Python_Project\Disease_Prediction\heart\templates\stroke.csv")

stroke = stroke.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'])

stroke = stroke.head(1000)
print(stroke.columns)
print(stroke.head())
stroke.columns = stroke.columns.str.strip()
stroke['gender'] = np.where(stroke['gender'] == "Male", 1, np.where(stroke['gender'] == "Female", 0, np.nan))
stroke['smoking_status'] = stroke['smoking_status'].map({'formerly smoked': 1, 'never smoked': 0, 'smokes': 2})
stroke = stroke.fillna(stroke.mean())
print(stroke.isnull().sum())

# Prepare features and target variable
x_heart = heart.drop('target', axis=1)
y_heart = heart['target']

x_stroke = stroke.drop('stroke', axis=1) 
y_stroke = stroke['stroke']


# print(stroke.dtypes)


# Split the dataset into training and testing sets
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(x_heart, y_heart, test_size=0.2, random_state=42)
X_train_stroke, X_test_stroke, y_train_stroke, y_test_stroke = train_test_split(x_stroke, y_stroke, test_size=0.2, random_state=42)

# Train a Logistic Regression model
lr = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
lr.fit(X_train_heart, y_train_heart)

lr_stroke = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
lr_stroke.fit(X_train_stroke, y_train_stroke)

# Calculate accuracy for validation purposes
y_pred_test = lr.predict(X_test_heart)
accuracy = accuracy_score(y_test_heart, y_pred_test)

y_pred_test_stroke = lr_stroke.predict(X_test_stroke)
stroke_accuracy = accuracy_score(y_test_stroke, y_pred_test_stroke)

# Create your views here.
def home(request):
    return render(request, 'home.html')

def heart(request):
    if request.method == 'POST':
        # Collecting input data from the form
        name = request.POST.get('name')
        age = int(request.POST.get('age'))
        sex = int(request.POST.get('sex'))
        cp = int(request.POST.get('cp'))
        trestbps = int(request.POST.get('trestbps'))
        chol = int(request.POST.get('chol'))
        fbs = int(request.POST.get('fbs'))
        restecg = int(request.POST.get('restecg'))
        thalach = int(request.POST.get('thalach'))
        exang = int(request.POST.get('exang'))
        oldpeak = float(request.POST.get('oldpeak'))
        slope = int(request.POST.get('slope'))
        ca = int(request.POST.get('ca'))
        thal = int(request.POST.get('thal'))

        # Create a numpy array of inputs
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Make the prediction
        prediction = lr.predict(features)  # Returns an array (e.g., [0] or [1])

        # Extract the actual prediction value from the array
        prediction_value = prediction[0]

        # Determine result
        if prediction_value == 1:
            result = 'yes'
        else:
            result = 'no'

        # Save prediction to the database
        HeartDisease.objects.create(
            name=name,
            age=age,
            sex=sex,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            restecg=restecg,
            thalach=thalach,
            exang=exang,
            oldpeak=oldpeak,
            slope=slope,
            ca=ca,
            thal=thal,
            prediction=result
        )

        # Convert features to a list to avoid NumPy array ambiguity
        features_list = features.tolist()

        # Return the result to the template
        return render(request, 'heart.html', {'prediction': result, 'accuracy': accuracy, 'features': features_list})

    # If not a POST request, just render the form
    return render(request, 'heart.html')


def stroke(request):
    if request.method == 'POST':
        # Collecting input data from the form
        gender = int(request.POST.get('gender'))
        age = float(request.POST.get('age'))
        hypertension = int(request.POST.get('hypertension'))
        heart_disease = int(request.POST.get('heart_disease'))
        ever_married = int(request.POST.get('ever_married'))
        work_type = int(request.POST.get('work_type'))
        Residence_type = int(request.POST.get('Residence_type'))
        avg_glucose_level = float(request.POST.get('avg_glucose_level'))
        bmi = float(request.POST.get('bmi'))
        smoking_status = int(request.POST.get('smoking_status'))

        # Create a numpy array of inputs
        stroke_features = np.array([[gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status]])

        # Make the prediction
        stroke_prediction = lr_stroke.predict(stroke_features)

        # Extract the actual prediction value from the array
        stroke_prediction_value = stroke_prediction[0]

        # Determine result
        if stroke_prediction_value == 1:
            result = 'yes'
        else:
            result = 'no'

        # Save prediction to the database
        StrokePrediction.objects.create(
            gender=gender,
            age=age,
            hypertension=hypertension,
            heart_disease=heart_disease,
            ever_married=ever_married,
            work_type=work_type,
            Residence_type=Residence_type,
            avg_glucose_level=avg_glucose_level,
            bmi=bmi,
            smoking_status=smoking_status,
            prediction=result
        )

        # Convert features to a list to avoid NumPy array ambiguity
        stroke_features_list = stroke_features.tolist()

        # Return the result to the template
        return render(request, 'stroke.html', {'prediction': result, 'accuracy': stroke_accuracy, 'features': stroke_features_list})

    # If not a POST request, just render the form
    return render(request, 'stroke.html')
