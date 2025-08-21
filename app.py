import streamlit as st
import pickle
import pandas as pd

# Load dataset
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

# Load trained Random Forest model
with open("rfc.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Random Forest Classifier App", layout="wide")

st.title("🌳 Random Forest Classifier Prediction App")

st.write("This app uses a trained Random Forest Classifier to make predictions based on your input.")

# Get feature names from trained model
try:
    model_features = model.feature_names_in_
    st.write("✅ Loaded model with features:", list(model_features))
except AttributeError:
    st.error("Model does not contain feature names. It might have been trained without feature names.")
    st.stop()

# Display dataset preview
if st.checkbox("Show Dataset Preview"):
    st.write("### Dataset Preview")
    st.dataframe(data.head())

# Input form for user
st.write("### Enter Input Features")

user_input = {}
for col in model_features:
    if col not in data.columns:
        st.warning(f"⚠️ Column '{col}' not found in dataset. Using default value = 0.")
        user_input[col] = 0
    elif pd.api.types.is_numeric_dtype(data[col]):
        user_input[col] = st.number_input(f"{col}", value=float(data[col].mean()))
    else:
        user_input[col] = st.selectbox(f"{col}", options=data[col].unique())

# Convert user input to dataframe (must match model features exactly)
input_df = pd.DataFrame([user_input], columns=model_features)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Model Prediction: **{prediction}**")
