import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import re

# Validation functions
def validate_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return bool(re.match(pattern, str(email)))

def validate_phone(phone):
    return str(phone).isdigit() and (7 <= len(str(phone)) <= 15)

# Data cleaning pipeline
def process_data(df):
    df['valid_email'] = df['email'].apply(validate_email)
    df['valid_phone'] = df['phone'].apply(validate_phone)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=3)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = model.fit_predict(df[numeric_cols])

    return df

# Streamlit UI
st.title("📊 AI-Enhanced CRM Data Cleaner")

uploaded_file = st.file_uploader("Upload your CRM CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data")
    st.dataframe(df)

    if st.button("Run Data Cleaning"):
        cleaned_df = process_data(df)
        st.success("✅ Data cleaned successfully!")
        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(cleaned_df)

        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", csv, "crm_cleaned.csv", "text/csv")
