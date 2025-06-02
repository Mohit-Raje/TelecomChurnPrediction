import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder , OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier  , GradientBoostingClassifier
import pickle
import streamlit as st
import joblib

with open('artifacts\model1.pkl' , 'rb') as file:
    model=pickle.load(file)

with open('artifacts\preprocessor1.pkl' , 'rb') as file:
    preprocessor=pickle.load(file)

st.title("📊 Customer Churn Prediction App")

st.markdown("Enter the customer's details below:")

# --- Input Widgets ---
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

multiple_lines = st.selectbox("MultipleLines", ["Yes", "No"])
internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("OnlineSecurity", ["Yes", "No"])
tech_support = st.selectbox("TechSupport", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
payment_method = st.selectbox("PaymentMethod", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])

# --- Create DataFrame ---
user_data = {
    'tenure': tenure,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'TechSupport': tech_support,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([user_data])

# --- Predict ---
if st.button("🔍 Predict Churn"):
    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)[0]

    if prediction == 0:
        st.success("Customer is likely to be **retained**.")
    else:
        st.error("Customer is likely to **churn**.")
