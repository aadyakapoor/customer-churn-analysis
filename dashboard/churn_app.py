import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

# -----------------------------
# Load model artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():

    with open(os.path.join(BASE_DIR, 'models', 'demo_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'models', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    feature_cols = pd.read_csv(
        os.path.join(BASE_DIR, 'models', 'feature_columns.csv')
    ).iloc[:, 0].tolist()

    return model, scaler, feature_cols


model, scaler, feature_cols = load_artifacts()

# -----------------------------
# App Title
# -----------------------------
st.title("📉 Customer Churn Risk Predictor")

st.markdown(
    """
Predict customer churn probability using a Random Forest model.

Model Highlights:
- Accuracy: 0.760
- ROC-AUC: 0.817
- Stable predictions for user-entered profiles
- Trained on Telco Customer Churn Dataset
"""
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider(
    "Tenure (months)",
    0,
    72,
    12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    18.0,
    120.0,
    70.0
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

# -----------------------------
# Build Feature Row
# -----------------------------
def build_input():

    row = pd.DataFrame(
        np.zeros((1, len(feature_cols))),
        columns=feature_cols
    )

    total_charges = monthly_charges * max(tenure, 1)

    if 'tenure' in row.columns:
        row['tenure'] = tenure

    if 'MonthlyCharges' in row.columns:
        row['MonthlyCharges'] = monthly_charges

    if 'TotalCharges' in row.columns:
        row['TotalCharges'] = total_charges

    if 'SeniorCitizen' in row.columns:
        row['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0

    mappings = {
        f'Contract_{contract}': 1,
        f'InternetService_{internet_service}': 1,
        f'TechSupport_{tech_support}': 1,
        f'OnlineSecurity_{online_security}': 1,
        f'StreamingTV_{streaming_tv}': 1,
        f'PaymentMethod_{payment_method}': 1
    }

    for col, value in mappings.items():
        if col in row.columns:
            row[col] = value

    num_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]

    row[num_cols] = scaler.transform(
        row[num_cols]
    )

    return row


# -----------------------------
# Predict Button
# -----------------------------
if st.sidebar.button("Predict Churn Risk"):

    input_row = build_input()

    probability = model.predict_proba(input_row)[0, 1]

    if probability >= 0.55:
        risk = "🔴 High Risk"
    elif probability >= 0.25:
        risk = "🟡 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )

    col2.metric(
        "Risk Level",
        risk
    )

    col3.metric(
        "Estimated LTV",
        f"${monthly_charges * 24:,.0f}"
    )

    st.markdown("---")

    st.subheader("Suggested Action")

    if contract == "Month-to-month":
        st.warning(
            "Offer a discounted upgrade to a longer contract."
        )

    elif internet_service == "Fiber optic":
        st.warning(
            "Consider a bundle discount for fiber customers."
        )

    elif tech_support == "No":
        st.warning(
            "Offer free tech support trial."
        )

    elif probability < 0.30:
        st.success(
            "Customer appears low risk. No intervention needed."
        )

    else:
        st.info(
            "Recommend proactive retention outreach."
        )

else:

    st.info(
        "Configure a customer profile on the left and click Predict Churn Risk."
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "Customer Churn Analysis Project | Random Forest Dashboard Deployment"
)