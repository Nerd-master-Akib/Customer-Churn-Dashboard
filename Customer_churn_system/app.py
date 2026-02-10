import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load model, features, and label encoders
# -------------------------------
model = joblib.load("model/churn_model.pkl")
features = joblib.load("model/features.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Intelligence Dashboard")
st.write(
    "Predict the probability of a customer churning. Fill in the customer details below."
)

st.divider()

# -------------------------------
# Input Section
# -------------------------------
st.subheader("🔍 Customer Information")

user_input = {}

# Example: dropdowns for categorical columns
categorical_cols = [col for col in features if col in label_encoders]

for col in categorical_cols:
    options = list(label_encoders[col].classes_)
    value = st.selectbox(f"{col}", options)
    # encode value
    user_input[col] = label_encoders[col].transform([value])[0]

# Number inputs for the rest
numeric_cols = [col for col in features if col not in categorical_cols]

for col in numeric_cols:
    value = st.number_input(f"{col}", value=0.0, step=1.0)
    user_input[col] = value

# Convert to model input array in the correct order
inputs = np.array([user_input[col] for col in features]).reshape(1, -1)

st.divider()

# -------------------------------
# Prediction Section
# -------------------------------
if st.button("Predict Churn Risk"):
    probability = model.predict_proba(inputs)[0][1]

    st.metric(
        label="Churn Probability",
        value=f"{probability*100:.2f}%"
    )

    # Business interpretation
    if probability >= 0.7:
        st.error("⚠️ High-risk customer — Immediate retention action recommended.")
    elif probability >= 0.4:
        st.warning("⚠️ Medium-risk customer — Monitor closely.")
    else:
        st.success("✅ Low churn risk — Customer likely to stay.")
