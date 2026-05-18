# Customer Churn Prediction System Using Machine Learning

## Project Overview

This project develops a customer churn prediction system using machine learning. The aim is to predict whether a telecom customer is likely to churn and to provide decision-support information through a lightweight Streamlit dashboard.

The project uses the Telco Customer Churn dataset and applies a full machine learning pipeline, including data cleaning, feature engineering, model training, model evaluation, threshold tuning, feature importance analysis, and prototype dashboard development.

## Project Objectives

The main objectives of this project are:

- To clean and prepare the Telco Customer Churn dataset for machine learning.
- To engineer additional features that may improve churn prediction.
- To train and compare multiple classification models.
- To evaluate models using accuracy, precision, recall, F1-score and AUC-ROC.
- To tune the decision threshold to improve churn detection.
- To identify important features influencing churn predictions.
- To build a Streamlit prototype dashboard for churn risk prediction.

## Dataset

The dataset used in this project is the Telco Customer Churn dataset.

The original dataset is stored in:
data/raw/Telco_customer_churn.xlsx

Processed versions of the dataset are stored in:
data/processed/

The target variable is:
Churn Value

Project structure:

telco_customer_churn_project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── Telco_customer_churn.xlsx
│   └── processed/
│       ├── telco_customer_churn.csv
│       ├── cleaned_telco_churn.csv
│       └── featured_telco_churn.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── convert_excel_to_csv.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── threshold_tuning.py
│   ├── final_model_evaluation.py
│   └── feature_importance.py
│
├── main.py
├── requirements.txt
└── README.md

Tools and Libraries

This project was developed using:

Python
pandas
NumPy
scikit-learn
XGBoost
matplotlib
joblib
Streamlit
openpyxl

Installation

Create and activate a virtual environment, then install the required libraries:
</> PowerShell
pip install -r requirements.txt

How to Run the Project
1. Convert Excel Dataset to CSV
</> PowerShell
python src/convert_excel_to_csv.py

2. Clean the Dataset
</> PowerShell
python src/data_preprocessing.py

3. Create Engineered Features
</> PowerShell
python src/feature_engineering.py

4. Train Machine Learning Models
</> PowerShell
python src/train_models.py

This trains:

Logistic Regression
Random Forest
XGBoost

The trained models are saved in the models/ folder.

5. Evaluate the Models
</> PowerShell
python src/evaluate_models.py

6. Run Threshold Tuning
</> PowerShell
python src/threshold_tuning.py

7. Evaluate the Final Selected Model
</> PowerShell
python src/final_model_evaluation.py

8. Generate Feature Importance Outputs
</> PowerShell
python src/feature_importance.py

This creates a feature importance table and chart for the XGBoost model.

How to Run the Streamlit Dashboard

To launch the dashboard, run:
</> PowerShell
streamlit run app/streamlit_app.py

The dashboard will open in the browser, usually at:
http://localhost:8501

The dashboard allows the user to enter customer details and returns:

Churn probability
Risk category
Final churn prediction
Suggested retention action
Customer data used for prediction

Final Model

The final selected model is:
XGBoost

The selected decision threshold is:
0.35

This threshold was selected because it improved recall and reduced false negatives compared with the default threshold of 0.50.

Model Performance Summary
Model	          Accuracy	  Precision	   Recall	  F1-score	   AUC-ROC
Logistic Regression	80.06%	  64.95%	   54.01%	  58.98%	   84.91%
Random Forest	    78.99%	  63.36%	   49.47%	  55.56%	   83.77%
XGBoost	            79.06%	  62.31%	   53.48%	  57.55%	   85.46%

After threshold tuning, XGBoost at threshold 0.35 achieved:

Model	Threshold Accuracy	Precision	Recall	  F1-score
XGBoost	    0.35	77.86%	56.49%	    72.19%	  63.38%

Notes

This project is an academic prototype. The Streamlit dashboard runs locally and is not deployed to a live production server. The results should be interpreted as predictive decision-support outputs rather than guaranteed customer behaviour.

Feature importance results should be interpreted as associations with model predictions, not as proof of causation.



