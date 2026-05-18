# Import pandas for working with tabular data
import pandas as pd


# Define the input path for the CSV dataset
input_file_path = "data/processed/telco_customer_churn.csv"

# Define the output path for the cleaned dataset
output_file_path = "data/processed/cleaned_telco_churn.csv"


# Load the CSV dataset into a DataFrame
df = pd.read_csv(input_file_path)


# Print the original dataset shape
print("Original dataset shape:", df.shape)


# Print the first five rows of the dataset
print("\nFirst five rows:")
print(df.head())


# Print information about columns and data types
print("\nDataset information:")
print(df.info())


# Print missing values for each column
print("\nMissing values before cleaning:")
print(df.isnull().sum())


# Print churn distribution as counts
print("\nChurn distribution:")
print(df["Churn Value"].value_counts())


# Print churn distribution as percentages
print("\nChurn distribution percentage:")
print(df["Churn Value"].value_counts(normalize=True) * 100)


# Convert Total Charges into a numeric column
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")


# Count missing values in Total Charges after conversion
print("\nMissing Total Charges after conversion:")
print(df["Total Charges"].isnull().sum())


# Fill missing Total Charges values with 0
df["Total Charges"] = df["Total Charges"].fillna(0)


# Define columns to remove from the modelling dataset
columns_to_drop = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "Churn Reason"
]


# Drop unsuitable and leakage-prone columns
df_cleaned = df.drop(columns=columns_to_drop)


# Print the cleaned dataset shape
print("\nCleaned dataset shape:", df_cleaned.shape)


# Print the remaining columns
print("\nRemaining columns:")
print(df_cleaned.columns.tolist())


# Print missing values after cleaning
print("\nMissing values after cleaning:")
print(df_cleaned.isnull().sum())


# Save the cleaned dataset as a CSV file
df_cleaned.to_csv(output_file_path, index=False)


# Print confirmation message
print("\nCleaned dataset saved successfully.")
print("Saved to:", output_file_path)
