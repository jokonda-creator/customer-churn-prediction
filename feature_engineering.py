# Import pandas for working with tabular data
import pandas as pd


# Define the input path for the cleaned dataset
input_file_path = "data/processed/cleaned_telco_churn.csv"

# Define the output path for the feature engineered dataset
output_file_path = "data/processed/featured_telco_churn.csv"


# Load the cleaned dataset into a DataFrame
df = pd.read_csv(input_file_path)


# Print the dataset shape before feature engineering
print("Dataset shape before feature engineering:", df.shape)


# Define a function to group customers by tenure
def create_tenure_group(tenure):
    # Check if the customer has been with the company for 12 months or less
    if tenure <= 12:
        # Return the new customer group
        return "New Customer"

    # Check if the customer has been with the company for 13 to 24 months
    elif tenure <= 24:
        # Return the developing customer group
        return "Developing Customer"

    # Check if the customer has been with the company for 25 to 48 months
    elif tenure <= 48:
        # Return the established customer group
        return "Established Customer"

    # Otherwise, the customer has been with the company for more than 48 months
    else:
        # Return the loyal customer group
        return "Loyal Customer"


# Create a tenure group feature from Tenure Months
df["Tenure Group"] = df["Tenure Months"].apply(create_tenure_group)


# Define service columns used to calculate service count
service_columns = [
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies"
]


# Create a service count feature by counting active services
df["Service Count"] = df[service_columns].apply(
    lambda row: (row == "Yes").sum(),
    axis=1
)


# Define support-related service columns
support_columns = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support"
]


# Create a feature showing whether the customer has any support service
df["Has Support Services"] = df[support_columns].apply(
    lambda row: "Yes" if (row == "Yes").sum() > 0 else "No",
    axis=1
)


# Create a monthly charge per service feature
df["Monthly Charge per Service"] = df["Monthly Charges"] / (df["Service Count"] + 1)


# Create a feature showing whether the customer has a month-to-month contract
df["Month-to-Month Contract"] = df["Contract"].apply(
    lambda contract: "Yes" if contract == "Month-to-month" else "No"
)


# Print the new features that were created
print("\nNew engineered features created:")
print([
    "Tenure Group",
    "Service Count",
    "Has Support Services",
    "Monthly Charge per Service",
    "Month-to-Month Contract"
])


# Print the distribution of tenure groups
print("\nTenure Group distribution:")
print(df["Tenure Group"].value_counts())


# Print summary statistics for service count
print("\nService Count summary:")
print(df["Service Count"].describe())


# Print churn rate by tenure group
print("\nChurn rate by Tenure Group:")
print(df.groupby("Tenure Group")["Churn Value"].mean())


# Print churn rate by support services
print("\nChurn rate by Has Support Services:")
print(df.groupby("Has Support Services")["Churn Value"].mean())


# Print the dataset shape after feature engineering
print("\nDataset shape after feature engineering:", df.shape)


# Save the feature engineered dataset as a CSV file
df.to_csv(output_file_path, index=False)


# Print confirmation message
print("\nFeature engineered dataset saved successfully.")
print("Saved to:", output_file_path)

