# Import pandas for working with tabular data
import pandas as pd

# Import joblib for loading saved models
import joblib

# Import matplotlib for creating and saving plots
import matplotlib.pyplot as plt

# Import train_test_split for recreating the same test set
from sklearn.model_selection import train_test_split

# Import confusion matrix display
from sklearn.metrics import ConfusionMatrixDisplay

# Import ROC curve display
from sklearn.metrics import RocCurveDisplay

# Import classification report
from sklearn.metrics import classification_report

# Import confusion matrix
from sklearn.metrics import confusion_matrix

# Import AUC-ROC score
from sklearn.metrics import roc_auc_score


# Define the path to the feature-engineered dataset
data_file_path = "data/processed/featured_telco_churn.csv"

# Define the target column
target_column = "Churn Value"


# Define model paths in a dictionary
model_paths = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}


# Define output paths for confusion matrices
confusion_matrix_paths = {
    "Logistic Regression": "outputs/figures/confusion_matrix_logistic_regression.png",
    "Random Forest": "outputs/figures/confusion_matrix_random_forest.png",
    "XGBoost": "outputs/figures/confusion_matrix_xgboost.png"
}


# Define output paths for ROC curves
roc_curve_paths = {
    "Logistic Regression": "outputs/figures/roc_curve_logistic_regression.png",
    "Random Forest": "outputs/figures/roc_curve_random_forest.png",
    "XGBoost": "outputs/figures/roc_curve_xgboost.png"
}


# Define output paths for classification reports
classification_report_paths = {
    "Logistic Regression": "outputs/tables/logistic_regression_classification_report.csv",
    "Random Forest": "outputs/tables/random_forest_classification_report.csv",
    "XGBoost": "outputs/tables/xgboost_classification_report.csv"
}


# Load the feature-engineered dataset
df = pd.read_csv(data_file_path)

# Print the dataset shape
print("Dataset shape:", df.shape)


# Separate features from the target variable
X = df.drop(columns=[target_column])

# Store the target variable separately
y = df[target_column]


# Recreate the same train/test split used during model training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Define a reusable function to evaluate and save model outputs
def evaluate_saved_model(model_name, model_path):
    # Print which model is being evaluated
    print(f"\nEvaluating model: {model_name}")

    # Load the saved model pipeline
    model = joblib.load(model_path)

    # Print confirmation that the model loaded
    print("Model loaded successfully.")

    # Generate class predictions on the test set
    y_pred = model.predict(X_test)

    # Generate churn probabilities on the test set
    y_proba = model.predict_proba(X_test)[:, 1]

    # Create the confusion matrix values
    confusion = confusion_matrix(y_test, y_pred)

    # Print the confusion matrix values
    print("\nConfusion Matrix:")
    print(confusion)

    # Calculate the AUC-ROC score
    auc_score = roc_auc_score(y_test, y_proba)

    # Print the AUC-ROC score
    print("AUC-ROC Score:", round(auc_score, 4))

    # Create the classification report as a dictionary
    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True
    )

    # Convert the classification report into a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Save the classification report as a CSV file
    report_df.to_csv(classification_report_paths[model_name])

    # Print confirmation that the report was saved
    print("Classification report saved successfully.")
    print("Saved to:", classification_report_paths[model_name])

    # Create the confusion matrix plot
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues"
    )

    # Add a title to the confusion matrix plot
    plt.title(f"Confusion Matrix - {model_name}")

    # Save the confusion matrix figure
    plt.savefig(confusion_matrix_paths[model_name], dpi=300, bbox_inches="tight")

    # Close the plot to avoid pausing the script
    plt.close()

    # Print confirmation that the confusion matrix was saved
    print("Confusion matrix saved successfully.")
    print("Saved to:", confusion_matrix_paths[model_name])

    # Create the ROC curve plot
    RocCurveDisplay.from_predictions(
        y_test,
        y_proba
    )

    # Add a title to the ROC curve plot
    plt.title(f"ROC Curve - {model_name}")

    # Save the ROC curve figure
    plt.savefig(roc_curve_paths[model_name], dpi=300, bbox_inches="tight")

    # Close the plot to avoid pausing the script
    plt.close()

    # Print confirmation that the ROC curve was saved
    print("ROC curve saved successfully.")
    print("Saved to:", roc_curve_paths[model_name])


# Loop through each saved model and evaluate it
for model_name, model_path in model_paths.items():
    # Evaluate the current saved model
    evaluate_saved_model(model_name, model_path)


# Print final confirmation message
print("\nAll model evaluation outputs were created successfully.")
