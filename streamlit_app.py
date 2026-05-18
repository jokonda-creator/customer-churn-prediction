# Import streamlit for building the web application
import streamlit as st

# Import pandas for working with tabular data
import pandas as pd

# Import joblib for loading the trained model
import joblib


# Define the path to the saved XGBoost model
model_file_path = "models/xgboost_model.pkl"

# Define the selected tuned threshold
selected_threshold = 0.35


# Load the trained model pipeline
model = joblib.load(model_file_path)


# Set the page title and layout
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide"
)


# Display the dashboard title
st.title("Customer Churn Prediction Dashboard")

# Display a short introduction
st.write(
    "This prototype uses a trained XGBoost machine learning model to estimate "
    "the likelihood that a customer may churn. The selected decision threshold "
    "is 0.35, based on threshold tuning results."
)


# Create two columns for input layout
left_column, right_column = st.columns(2)


# -----------------------------
# Customer demographic inputs
# -----------------------------

with left_column:
    st.subheader("Customer Profile")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure_months = st.slider(
        "Tenure Months",
        min_value=0,
        max_value=72,
        value=12
    )


# -----------------------------
# Service inputs
# -----------------------------

with right_column:
    st.subheader("Services and Contract")

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


# Create another two-column layout
left_column_2, right_column_2 = st.columns(2)


# -----------------------------
# Billing inputs
# -----------------------------

with left_column_2:
    st.subheader("Billing Information")

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )

    cltv = st.number_input(
    "Customer Lifetime Value (CLTV)",
    min_value=0.0,
    max_value=10000.0,
    value=3000.0,
    step=50.0
)


# -----------------------------
# Engineered feature preview
# -----------------------------

with right_column_2:
    st.subheader("Engineered Feature Preview")

    # Create tenure group from tenure months
    if tenure_months <= 12:
        tenure_group = "New Customer"
    elif tenure_months <= 24:
        tenure_group = "Developing Customer"
    elif tenure_months <= 48:
        tenure_group = "Established Customer"
    else:
        tenure_group = "Loyal Customer"

    # Create a list of service selections
    service_values = [
        phone_service,
        multiple_lines,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    # Count active services
    service_count = service_values.count("Yes")

    # Create support services feature
    support_values = [
        online_security,
        online_backup,
        device_protection,
        tech_support
    ]

    # Check whether the customer has at least one support service
    if "Yes" in support_values:
        has_support_services = "Yes"
    else:
        has_support_services = "No"

    # Calculate monthly charge per service
    monthly_charge_per_service = monthly_charges / (service_count + 1)

    # Create month-to-month contract feature
    if contract == "Month-to-month":
        month_to_month_contract = "Yes"
    else:
        month_to_month_contract = "No"

    # Display engineered feature values
    st.write("Tenure Group:", tenure_group)
    st.write("Service Count:", service_count)
    st.write("Has Support Services:", has_support_services)
    st.write("Monthly Charge per Service:", round(monthly_charge_per_service, 2))
    st.write("Month-to-Month Contract:", month_to_month_contract)


# -----------------------------
# Prediction section
# -----------------------------

st.subheader("Prediction Result")


# Create prediction button
if st.button("Predict Churn Risk"):

    # Create a single-row DataFrame using the same feature names used in training
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Senior Citizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure Months": [tenure_months],
        "Phone Service": [phone_service],
        "Multiple Lines": [multiple_lines],
        "Internet Service": [internet_service],
        "Online Security": [online_security],
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Paperless Billing": [paperless_billing],
        "Payment Method": [payment_method],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges],
        "CLTV": [cltv],
        "Tenure Group": [tenure_group],
        "Service Count": [service_count],
        "Has Support Services": [has_support_services],
        "Monthly Charge per Service": [monthly_charge_per_service],
        "Month-to-Month Contract": [month_to_month_contract]
    })

    # Generate churn probability
    churn_probability = model.predict_proba(input_data)[:, 1][0]

    # Apply tuned threshold
    churn_prediction = 1 if churn_probability >= selected_threshold else 0

    # Display probability
    st.metric(
        label="Churn Probability",
        value=f"{churn_probability:.2%}"
    )

    # Display prediction result
    if churn_prediction == 1:
        st.error("Prediction: High churn risk")
    else:
        st.success("Prediction: Low churn risk")

    # Create risk category
    if churn_probability >= 0.70:
        risk_category = "Very High Risk"
    elif churn_probability >= 0.50:
        risk_category = "High Risk"
    elif churn_probability >= selected_threshold:
        risk_category = "Medium Risk"
    else:
        risk_category = "Low Risk"

    # Display risk category
    st.write("Risk Category:", risk_category)

    # Display business recommendation
    st.subheader("Suggested Retention Action")

    if risk_category == "Very High Risk":
        st.write(
            "Priority retention action recommended. Consider personalised offers, "
            "contract review, and direct customer support follow-up."
        )
    elif risk_category == "High Risk":
        st.write(
            "Retention action recommended. Review contract type, monthly charges, "
            "and support service usage."
        )
    elif risk_category == "Medium Risk":
        st.write(
            "Monitor customer behaviour and consider a light-touch retention offer "
            "or customer satisfaction check."
        )
    else:
        st.write(
            "No immediate retention action required. Continue normal customer engagement."
        )

    # Display model decision details
    st.subheader("Model Decision Details")

    st.write("Selected model: XGBoost")
    st.write("Decision threshold:", selected_threshold)

    # Display input data used by the model
    st.subheader("Customer Data Used for Prediction")
    st.dataframe(input_data)
    

