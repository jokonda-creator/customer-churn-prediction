# Import pandas for working with tabular data
import pandas as pd

# Import joblib for loading saved machine learning models
import joblib

# Import train_test_split for recreating the same test set
from sklearn.model_selection import train_test_split

# Import evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# Define the path to the feature-engineered dataset
data_file_path = "data/processed/featured_telco_churn.csv"

# Define the target column
target_column = "Churn Value"

# Define the output path for the threshold tuning results
threshold_results_output_path = "outputs/tables/threshold_tuning_results.csv"


# Define the saved models to tune
model_paths = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}


# Define the thresholds to test
thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


# Load the feature-engineered dataset
df = pd.read_csv(data_file_path)

# Print the dataset shape
print("Dataset shape:", df.shape)


# Separate features from the target variable
X = df.drop(columns=[target_column])

# Store the target variable separately
y = df[target_column]


# Recreate the same train/test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create an empty list to store threshold tuning results
threshold_results = []


# Loop through each saved model
for model_name, model_path in model_paths.items():

    # Print the model currently being tuned
    print(f"\nTuning threshold for: {model_name}")

    # Load the saved model pipeline
    model = joblib.load(model_path)

    # Generate churn probabilities for the test set
    y_proba = model.predict_proba(X_test)[:, 1]

    # Loop through each threshold value
    for threshold in thresholds:

        # Convert probabilities into class predictions using the threshold
        y_pred_threshold = (y_proba >= threshold).astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred_threshold)

        # Calculate precision
        precision = precision_score(y_test, y_pred_threshold)

        # Calculate recall
        recall = recall_score(y_test, y_pred_threshold)

        # Calculate F1-score
        f1 = f1_score(y_test, y_pred_threshold)

        # Create the confusion matrix
        confusion = confusion_matrix(y_test, y_pred_threshold)

        # Extract true negatives from the confusion matrix
        true_negatives = confusion[0][0]

        # Extract false positives from the confusion matrix
        false_positives = confusion[0][1]

        # Extract false negatives from the confusion matrix
        false_negatives = confusion[1][0]

        # Extract true positives from the confusion matrix
        true_positives = confusion[1][1]

        # Add the threshold results to the list
        threshold_results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "True Negatives": true_negatives,
            "False Positives": false_positives,
            "False Negatives": false_negatives,
            "True Positives": true_positives
        })

        # Print a short summary for this threshold
        print(
            f"Threshold: {threshold:.2f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1-score: {f1:.4f}"
        )


# Convert the threshold tuning results into a DataFrame
threshold_results_df = pd.DataFrame(threshold_results)

# Save the threshold tuning results as a CSV file
threshold_results_df.to_csv(threshold_results_output_path, index=False)

# Print confirmation message
print("\nThreshold tuning results saved successfully.")
print("Saved to:", threshold_results_output_path)


# Print the full threshold tuning table
print("\nThreshold Tuning Results:")
print(threshold_results_df)
