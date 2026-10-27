# Customer Churn Prediction

An educational data science project that explores customer churn in a Telco dataset and compares two beginner-friendly classification models in one Jupyter notebook.

## Business problem

Churn occurs when a customer stops using a service. Understanding which customer groups are more often marked as churned can help a business consider appropriate retention support. The patterns and model scores here describe this dataset; they do not prove why a customer churns or guarantee what will happen to an individual.

## Dataset

The project uses the local file `data/raw/Telco-Customer-Churn.csv`, which contains 7,043 customer records and 21 columns, including contract, service, tenure, charges, and the `Churn` outcome.

The [IBM Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d) is a reference for this commonly used dataset. The current local Git history does not establish the file's exact upstream provenance, so this project does not claim that the CSV has been independently verified byte-for-byte against IBM's copy.

## Analysis

The notebook:

1. Loads and inspects the CSV, including column types, summary statistics, missing values, and duplicate rows.
2. Converts blank `TotalCharges` values to missing numeric values. Eleven such values are found; the rows are retained and imputed within the training pipeline.
3. Explores churn counts and rates, including rates by contract, internet service, payment method, and senior-citizen status, plus tenure and monthly-charge distributions.
4. Splits data into stratified training and test sets (80/20, `random_state=42`) and excludes `customerID` from model features.
5. Imputes numeric and categorical values, scales numeric features, and one-hot encodes categories inside scikit-learn pipelines. Preprocessing is fitted on training data only.
6. Trains and evaluates Logistic Regression and Random Forest classifiers.

## Observed findings

In this dataset, 1,869 customers (26.5%) are marked as churned. The observed churn rates are 42.7% for month-to-month contracts, 11.3% for one-year contracts, and 2.8% for two-year contracts. Fiber-optic customers have a 41.9% churn rate, and customers using electronic check have a 45.3% churn rate. Average tenure is 18.0 months for churned customers and 37.6 months for customers who did not churn; average monthly charges are 74.4 and 61.3, respectively. These are associations in the sample, not causal findings.

## Test-set model results

The following metrics were calculated on the same held-out test set using each model's default classification threshold:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.738 | 0.504 | 0.783 | 0.614 | 0.841 |
| Random Forest | 0.782 | 0.618 | 0.468 | 0.533 | 0.819 |

Using F1-score as a balance between precision and recall, with recall as the tie-breaker, Logistic Regression ranks first in this run. It identifies more of the churned customers (higher recall), while Random Forest has higher accuracy and precision. This is a trade-off, not evidence that one model is best for every business; outreach costs and capacity should inform any real decision.

## Project structure

```text
Customer-Churn-Prediction/
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
├── notebooks/
│   └── customer_churn_prediction.ipynb
├── .gitignore
└── README.md
```

## Run the notebook

Use Python with Jupyter Notebook or JupyterLab. From the project folder in PowerShell:

```powershell
cd C:\Users\PC\Documents\Customer-Churn-Prediction
jupyter notebook notebooks\customer_churn_prediction.ipynb
```

You can also open `notebooks/customer_churn_prediction.ipynb` in VS Code and select a Python kernel. Run the cells from top to bottom. The notebook looks for the CSV relative to the project folder or its `notebooks` subfolder and raises a clear error if the expected file is missing.

## Libraries

- Python
- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn
- Jupyter/IPython

These libraries were already available in the selected Anaconda environment used to execute the notebook.

## Limitations and future improvements

This is a historical sample and may not represent a current customer base. The test results come from one stratified split, use the default classification threshold, and are not an external or production validation. Accuracy alone can be misleading when churn classes are imbalanced; precision and recall also matter. The models and observed group differences do not establish causality. Future work could validate on newer data, choose a threshold based on intervention costs and outreach capacity, and monitor model performance and fairness over time.
