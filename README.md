# Titanic Analysis

This project holds the analysis and modeling for the famous Kaggle Titanic dataset. This repository serves as the centralized version control for our notebooks, source code, and configurations.

## Methodology & Feature Engineering

To achieve the best possible performance, we implemented a robust feature engineering and preprocessing pipeline:

1. **Title Extraction**: Passenger titles (Mr, Mrs, Miss, etc.) were extracted from names, and rare titles were grouped.
2. **Family Features**: A `FamilySize` feature was created (`SibSp` + `Parch` + 1), and an `IsAlone` binary feature was derived.
3. **Deck Extraction**: We extracted the `Deck` (the first letter) from the `Cabin` column, filling missing values with `'U'` (Unknown).
4. **Data Imputation & Preprocessing**: Instead of dropping rows with missing values (which discarded ~20% of the dataset), we preserved all data using a `ColumnTransformer` with a pipeline:
   - **Numeric Features** (`Fare`, `Age`, `FamilySize`, `IsAlone`): Imputed with median and scaled using `StandardScaler`.
   - **Categorical Features** (`Pclass`, `Sex`, `Embarked`, `Title`, `Deck`): Imputed with the most frequent value and encoded using `OneHotEncoder`.

## Model Results

We evaluated several classification algorithms using 5-fold cross-validation with `GridSearchCV` hyperparameter tuning. The results on the complete training set are as follows:

| Model | Accuracy | ROC AUC |
| :--- | :--- | :--- |
| **Support Vector Machine (SVC)** | **0.8268** | 0.8434 |
| **Gradient Boosting** | **0.8268** | 0.8434 |
| **Logistic Regression** | 0.8212 | 0.8617 |
| **K-Nearest Neighbours** | 0.8044 | **0.8653** |
| **Decision Tree** | 0.7988 | 0.8431 |
| **XGBoost** | 0.7988 | 0.8421 |
| **Random Forest** | 0.7932 | 0.8334 |
| **LightGBM** | 0.7765 | 0.8229 |

**Support Vector Machine (SVC)** and **Gradient Boosting** proved to be the most accurate models in strict classification, while **K-Nearest Neighbours (KNN)** and **Logistic Regression** showed the best overall probabilistic ordering (ROC AUC).

Our final submission cell in the notebook allows you to quickly swap between the best-performing models to generate `submission.csv` predictions on the test dataset.

## Directory Structure

- `data/`: Contains raw and processed data. (Note: Actual data files are git-ignored).
- `notebooks/`: Jupyter notebooks for exploratory data analysis and modeling.
- `src/`: Reusable Python source code (scripts, modules) imported by notebooks or used in production.
- `test/`: Extracted scripts and output artifacts.

## Getting Started

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`. (Note: you may need to run `pip install xgboost lightgbm` if they're not in the requirements)
3. Launch Jupyter: `jupyter notebook` or `jupyter lab`.

