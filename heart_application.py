
import streamlit as st
import pickle
import numpy as np

# Load the trained RandomForest model and scaler
with open('rheart_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('rsc.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the web app
st.title('Heart Disease Prediction App')

# Input fields
age = st.number_input('Age', min_value=0.0, max_value=120.0, value=25.0, step=1.0)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3],
                  format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'][x])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50.0, max_value=250.0, value=120.0, step=1.0)
chol = st.number_input('Serum Cholesterol (chol)', min_value=100.0, max_value=600.0, value=200.0, step=1.0)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=[0, 1, 2],
                       format_func=lambda x: ['Normal', 'ST-T wave abnormality', 'Left Ventricular Hypertrophy'][x])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60.0, max_value=250.0, value=150.0, step=1.0)
exang = st.selectbox('Exercise-Induced Angina (exang)', options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', options=[0, 1, 2],
                     format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
ca = st.number_input('Number of Major Vessels (ca)', min_value=0, max_value=4, value=0, step=1)
thal = st.selectbox('Thalassemia (thal)', options=[0, 1, 2, 3],
                    format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][x])

# Prepare the feature vector
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], dtype=np.float64)

# Scale the features
features_scaled = scaler.transform(features)

# Predict Diabetes
predict_heartdisease = model.predict(features_scaled)

prediction_label = "Yes" if predicted_Diabetes[0] == 1 else "No"
st.write(f'Predicted Heart Disease: {prediction_label}')
