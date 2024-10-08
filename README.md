# Advanced-Sales-Forecasting-with-Machine-Learning-RandomForest-XGBoost-and-ARIMA

## Project Overview
This project focuses on predicting sales for ABC, a leading grocery retailer in South America, using historical sales data from 2013 to 2017. The dataset spans across 54 stores and 33 product types, and the goal is to accurately forecast sales from July 31, 2017, to August 15, 2017. The analysis evaluates multiple machine learning models: Linear Regression, XGBoost, RandomForest Regressor, and ARIMA.

Our approach involves extensive data preprocessing, feature engineering, and the application of supervised learning algorithms. After comparing models, we optimize the best-performing algorithm using GridSearchCV to fine-tune its hyperparameters.

## Dataset Information
The dataset contains the following features:
- **Date**: Temporal context for the sales data.
- **Store_nbr**: Unique identifier for each store.
- **Product_type**: Classifies products sold across categories.
- **Sales**: Total sales achieved for each product type.
- **Special_offer**: Measures the intensity of promotional efforts (0-200 scale).

## Objectives
1. **Data Preprocessing**:
   - Data cleaning, conversion of date columns to `DateTime` format, and handling missing values.
   - Feature extraction (temporal and lag features), feature scaling using Box-Cox transformation, and data splitting.

2. **Feature Engineering**:
   - Creation of temporal features: `year`, `month`, `day_of_week`, and lag-based features.
   - Feature encoding using One-Hot Encoding for categorical features like `product_type`.
   - Feature selection using correlation heatmaps and `SelectKBest`.

3. **Modeling**:
   - Evaluate several machine learning models: Linear Regression, XGBoost, RandomForest, and ARIMA.
   - GridSearchCV for hyperparameter tuning of the best model.

## Key Features
- **Feature Creation**: Temporal and lag features such as `sales_lag_1`, `sales_lag_7`, and `rolling_avg_sales` to capture time-series trends.
- **Feature Scaling**: Applied Box-Cox transformation to handle skewed distributions.
- **Model Evaluation**: Comparative analysis of models based on Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

## Models Used
1. **Linear Regression**: 
   - Baseline model with an R² score of 0.2105 on testing data.
   - RMSE: 982.69 | MAE: 475.50

2. **XGBoost Regressor**:
   - Achieved 93.01% training accuracy and 90.41% testing accuracy.
   - RMSE: 342.471 | MAE: 69.155

3. **RandomForest Regressor**:
   - Tuned with GridSearchCV for optimal performance.
   - Final RMSE: 314.748 | MAE: 76.853 after optimization.

4. **ARIMA**:
   - Time-series analysis model to capture temporal dependencies.

## Results

### Model Performance Summary:

| **Model**                  | **Mean Absolute Error (MAE)** | **Root Mean Squared Error (RMSE)** |
|----------------------------|-------------------------------|------------------------------------|
| **RandomForest Regressor**  | 79.355                        | 323.684                           |
| **XGBoost Regressor**       | 78.305                        | 324.131                           |
| **Linear Regression**       | 554.014                       | 1054.399                          |
| **RandomForest (Optimized)**| 76.853                        | 314.748                           |

### Model Performance on Unseen Data:

| **Model**                             | **Mean Absolute Error (MAE)** | **Root Mean Squared Error (RMSE)** |
|---------------------------------------|-------------------------------|------------------------------------|
| **RandomForest Regressor (Optimized)**| 83.966                        | 290.730                           |

## Installation & Usage

### Requirements:
- Python 3.6 or above
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `statsmodels`

### Install Dependencies
```bash
pip install requirements.txt
