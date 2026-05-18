# Import pandas for working with tabular data
import pandas as pd

# Import joblib for loading the saved model
import joblib

# Import matplotlib for creating and saving the feature importance chart
import matplotlib.pyplot as plt


# Define the path to the saved XGBoost model
model_file_path = "models/xgboost_model.pkl"

# Define the output path for the feature importance table
feature_importance_table_path = "outputs/tables/xgboost_feature_importance.csv"

# Define the output path for the feature importance figure
feature_importance_figure_path = "outputs/figures/xgboost_feature_importance_top20.png"


# Load the saved XGBoost pipeline
model_pipeline = joblib.load(model_file_path)

# Print confirmation message
print("XGBoost model pipeline loaded successfully.")


# Extract the preprocessing part of the pipeline
preprocessor = model_pipeline.named_steps["preprocessor"]

# Extract the XGBoost classifier part of the pipeline
xgboost_model = model_pipeline.named_steps["classifier"]


# Get the numerical feature names
numerical_features = preprocessor.transformers_[0][2]

# Get the categorical feature names before one-hot encoding
categorical_features = preprocessor.transformers_[1][2]


# Get the one-hot encoder from the categorical preprocessing pipeline
one_hot_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

# Get the new categorical feature names after one-hot encoding
encoded_categorical_features = one_hot_encoder.get_feature_names_out(categorical_features)


# Combine numerical and encoded categorical feature names
all_feature_names = list(numerical_features) + list(encoded_categorical_features)


# Get feature importance scores from the XGBoost model
importance_scores = xgboost_model.feature_importances_


# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importance_scores
})


# Sort the feature importance table from highest to lowest
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)


# Save the full feature importance table as a CSV file
feature_importance_df.to_csv(feature_importance_table_path, index=False)

# Print confirmation message
print("\nFeature importance table saved successfully.")
print("Saved to:", feature_importance_table_path)


# Print the top 20 most important features
print("\nTop 20 Feature Importance:")
print(feature_importance_df.head(20))


# Select the top 20 features for plotting
top_20_features = feature_importance_df.head(20)


# Create the feature importance bar chart
plt.figure(figsize=(10, 8))

# Plot the feature importance values
plt.barh(
    top_20_features["Feature"],
    top_20_features["Importance"]
)

# Reverse the order so the most important feature appears at the top
plt.gca().invert_yaxis()

# Add chart title
plt.title("Top 20 XGBoost Feature Importances")

# Add x-axis label
plt.xlabel("Importance Score")

# Add y-axis label
plt.ylabel("Feature")

# Adjust layout so labels fit properly
plt.tight_layout()

# Save the feature importance chart
plt.savefig(feature_importance_figure_path, dpi=300, bbox_inches="tight")

# Close the figure
plt.close()

# Print confirmation message
print("\nFeature importance figure saved successfully.")
print("Saved to:", feature_importance_figure_path)
