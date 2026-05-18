# Import pandas for reading and saving data
import pandas as pd


# Define the path to the original Excel file
excel_file_path = "data/raw/Telco_customer_churn.xlsx"

# Define the path where the CSV file will be saved
csv_file_path = "data/processed/telco_customer_churn.csv"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Save the DataFrame as a CSV file
df.to_csv(csv_file_path, index=False)

# Print a success message
print("Excel file converted to CSV successfully.")

# Print the dataset shape
print("Dataset shape:", df.shape)

# Print the CSV output path
print("CSV saved to:", csv_file_path)