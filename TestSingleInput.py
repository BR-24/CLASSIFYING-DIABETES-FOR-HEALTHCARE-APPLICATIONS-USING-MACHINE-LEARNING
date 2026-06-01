import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load pre-trained models
with open('ann_model.h5', 'rb') as f:
    ann_model = load_model('ann_model.h5')

# Function to get user input
def get_user_input():
    print("Enter the following parameters for diabetes prediction:")
    pregnancies = int(input("Number of pregnancies: "))
    glucose = float(input("Glucose level: "))
    blood_pressure = float(input("Blood pressure: "))
    skin_thickness = float(input("Skin thickness: "))
    insulin = float(input("Insulin level: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = int(input("Age: "))

    return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Function to display prediction results (Yes/No)
def display_result(prediction):
    return "Yes" if prediction == 1 else "No"

# Main function to get input and make predictions
def predict_diabetes():
    user_input = get_user_input()

    # Predict using ANN model
    ann_prediction = ann_model.predict(user_input)
    print(f"ANN Model Prediction: {display_result(ann_prediction[0])}")

    
if __name__ == "__main__":
    predict_diabetes()
