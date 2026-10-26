# Customer Churn Prediction

This project explores the IBM Telco Customer Churn sample and prepares its customer records for a later classification task. The notebook contains descriptive exploratory data analysis (EDA) and a reproducible preprocessing workflow. **No predictive model has been trained yet.**

## Dataset

The CSV is the public sample from IBM's [telco-customer-churn-on-icp4d repository](https://github.com/IBM/telco-customer-churn-on-icp4d), stored locally at `data/raw/Telco-Customer-Churn.csv`.

## Project structure

```text
Customer-Churn-Prediction/
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
├── notebooks/
│   └── 01_churn_exploration.ipynb
├── .gitignore
└── README.md
```

## Exploratory findings

The dataset has 7,043 customer records and 21 columns. Eleven whitespace-only `TotalCharges` entries are treated as missing and converted to numeric values during the notebook workflow. Churn is recorded for 1,869 customers (26.5%); 5,174 customers (73.5%) did not churn.

The notebook includes charts comparing customer counts by churn outcome, descriptive churn rates across contract types, and tenure and monthly charges by churn group. In this sample, the observed churn rates are 42.7% for month-to-month contracts, 11.3% for one-year contracts, and 2.8% for two-year contracts. Mean tenure is 18.0 months for customers marked as churned and 37.6 months for those not marked as churned; mean monthly charges are 74.4 and 61.3, respectively. These are descriptive associations in this dataset, not evidence of causation or predictions.

## Preprocessing

The notebook maps `Churn` from `No`/`Yes` to 0/1, excludes the customer ID from model features, and makes a reproducible 80/20 stratified train/test split with `random_state=42`. A scikit-learn `ColumnTransformer` imputes numeric features with the median and categorical features with the most frequent value, then one-hot encodes categorical features with `handle_unknown="ignore"`. The transformer is fit only on the training split and used to transform the test split. This prepares data for a later modeling task; it does not train a model.

## Run

Use the existing Anaconda Python environment with Jupyter, pandas, matplotlib, and scikit-learn installed. From the project root, open and run all cells in order:

```powershell
cd C:\Users\PC\Documents\Customer-Churn-Prediction
jupyter notebook notebooks\01_churn_exploration.ipynb
```

The notebook resolves the CSV through a path relative to the project root, whether it is run with the project root or the `notebooks` folder as its working directory.
