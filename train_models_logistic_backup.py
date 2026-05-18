# Import pandas for working with tabular data
import pandas as pd

# Import joblib for saving trained models
import joblib

# Import train_test_split for splitting the dataset
from sklearn.model_selection import train_test_split

# Import preprocessing tools
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import ColumnTransformer for applying different preprocessing to different columns
from sklearn.compose import ColumnTransformer

# Import Pipeline for combining preprocessing and modelling
from sklearn.pipeline import Pipeline

# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Import evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Define the input path for the feature-engineered dataset
input_file_path = "data/processed/featured_telco_churn.csv"

# Define the output path for the model results table
results_output_path = "outputs/tables/model_results.csv"

# Define the output path for the saved Logistic Regression model
model_output_path = "models/logistic_regression_model.pkl"


# Load the feature-engineered dataset
df = pd.read_csv(input_file_path)


# Print the dataset shape
print("Feature-engineered dataset shape:", df.shape)


# Define the target variable
target_column = "Churn Value"


# Separate the input features from the target variable
X = df.drop(columns=[target_column])


# Store the target variable separately
y = df[target_column]

# Print unique target values
print("\nUnique target values:")
print(y.unique())

# Print target class counts
print("\nTarget value counts:")
print(y.value_counts(dropna=False))

# Stop the script if only one class exists
if y.nunique() < 2:
    raise ValueError("Target variable contains only one class. Check the Churn Value column.")

# Print the feature shape
print("Feature matrix shape:", X.shape)


# Print the target shape
print("Target shape:", y.shape)


# Print the target class distribution
print("\nTarget class distribution:")
print(y.value_counts())


# Print the target class distribution as percentages
print("\nTarget class distribution percentage:")
print(y.value_counts(normalize=True) * 100)


# Identify categorical columns automatically
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()


# Identify numerical columns automatically
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()


# Print categorical columns
print("\nCategorical columns:")
print(categorical_columns)


# Print numerical columns
print("\nNumerical columns:")
print(numerical_columns)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Print the training set size
print("\nTraining feature shape:", X_train.shape)


# Print the testing set size
print("Testing feature shape:", X_test.shape)


# Print training target distribution
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)


# Print testing target distribution
print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True) * 100)


# Create the numerical preprocessing pipeline
numeric_transformer = Pipeline(
    steps=[
        # Standardise numerical features
        ("scaler", StandardScaler())
    ]
)


# Create the categorical preprocessing pipeline
categorical_transformer = Pipeline(
    steps=[
        # One-hot encode categorical features
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        # Apply scaling to numerical columns
        ("num", numeric_transformer, numerical_columns),

        # Apply one-hot encoding to categorical columns
        ("cat", categorical_transformer, categorical_columns)
    ]
)


# Create the Logistic Regression pipeline
logistic_regression_pipeline = Pipeline(
    steps=[
        # Apply preprocessing first
        ("preprocessor", preprocessor),

        # Train the Logistic Regression classifier
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]
)


# Train the Logistic Regression model
logistic_regression_pipeline.fit(X_train, y_train)


# Generate class predictions on the test set
y_pred = logistic_regression_pipeline.predict(X_test)


# Generate churn probabilities on the test set
y_proba = logistic_regression_pipeline.predict_proba(X_test)[:, 1]


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# Calculate precision
precision = precision_score(y_test, y_pred)


# Calculate recall
recall = recall_score(y_test, y_pred)


# Calculate F1-score
f1 = f1_score(y_test, y_pred)


# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test, y_proba)


# Print the model results
print("\nLogistic Regression Baseline Results")
print("------------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")


# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Create the confusion matrix
confusion = confusion_matrix(y_test, y_pred)


# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion)


# Store model results in a list of dictionaries
model_results = [
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC-ROC": auc_roc
    }
]


# Convert model results into a DataFrame
results_df = pd.DataFrame(model_results)


# Save the model results table as a CSV file
results_df.to_csv(results_output_path, index=False)


# Print confirmation that results were saved
print("\nModel results saved successfully.")
print("Saved to:", results_output_path)


# Save the trained Logistic Regression pipeline
joblib.dump(logistic_regression_pipeline, model_output_path)


# Print confirmation that the model was saved
print("\nLogistic Regression model saved successfully.")
print("Saved to:", model_output_path)
