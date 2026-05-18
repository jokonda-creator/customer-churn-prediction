# Import pandas for working with tabular data
import pandas as pd

# Import joblib for loading the saved model
import joblib

# Import matplotlib for saving plots
import matplotlib.pyplot as plt

# Import train_test_split for recreating the same test set
from sklearn.model_selection import train_test_split

# Import evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Define the path to the feature-engineered dataset
data_file_path = "data/processed/featured_telco_churn.csv"

# Define the path to the saved XGBoost model
model_file_path = "models/xgboost_model.pkl"

# Define the selected decision threshold
selected_threshold = 0.35

# Define the target column
target_column = "Churn Value"

# Define output path for final selected model results
final_results_output_path = "outputs/tables/final_selected_model_results.csv"

# Define output path for tuned confusion matrix
confusion_matrix_output_path = "outputs/figures/confusion_matrix_xgboost_threshold_035.png"


# Load the feature-engineered dataset
df = pd.read_csv(data_file_path)

# Print dataset shape
print("Dataset shape:", df.shape)


# Separate features from target
X = df.drop(columns=[target_column])

# Store target variable separately
y = df[target_column]


# Recreate the same train/test split used during previous evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Load the saved XGBoost model pipeline
model = joblib.load(model_file_path)

# Print confirmation message
print("XGBoost model loaded successfully.")


# Generate churn probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Convert probabilities into predictions using the selected threshold
y_pred_tuned = (y_proba >= selected_threshold).astype(int)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_tuned)

# Calculate precision
precision = precision_score(y_test, y_pred_tuned)

# Calculate recall
recall = recall_score(y_test, y_pred_tuned)

# Calculate F1-score
f1 = f1_score(y_test, y_pred_tuned)

# Create confusion matrix
confusion = confusion_matrix(y_test, y_pred_tuned)

# Extract true negatives
true_negatives = confusion[0][0]

# Extract false positives
false_positives = confusion[0][1]

# Extract false negatives
false_negatives = confusion[1][0]

# Extract true positives
true_positives = confusion[1][1]


# Print final selected model results
print("\nFinal Selected Model: XGBoost")
print("Selected threshold:", selected_threshold)
print("------------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion)


# Store final selected model results
final_results = {
    "Model": "XGBoost",
    "Threshold": selected_threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "True Negatives": true_negatives,
    "False Positives": false_positives,
    "False Negatives": false_negatives,
    "True Positives": true_positives
}


# Convert final results into a DataFrame
final_results_df = pd.DataFrame([final_results])

# Save final results as a CSV file
final_results_df.to_csv(final_results_output_path, index=False)

# Print confirmation message
print("\nFinal selected model results saved successfully.")
print("Saved to:", final_results_output_path)


# Create confusion matrix plot
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_tuned,
    display_labels=["No Churn", "Churn"],
    cmap="Blues"
)

# Add title
plt.title("Confusion Matrix - XGBoost Tuned Threshold 0.35")

# Save the figure
plt.savefig(confusion_matrix_output_path, dpi=300, bbox_inches="tight")

# Close the figure
plt.close()

# Print confirmation message
print("\nTuned confusion matrix saved successfully.")
print("Saved to:", confusion_matrix_output_path)
