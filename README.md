# DS239 Project 3 – PCA and Regression

## Problem
The goal of this project is to predict the probability that a job will be automated by 2030. This is a regression problem where the target variable is Automation_Probability_2030.

## Dataset
AI Impact on Jobs 2030 Dataset

## Cleaning Process
- Filled missing numeric values with median
- Filled missing categorical values with mode
- One-hot encoded categorical variables
- Scaled features using StandardScaler
- Removed Risk_Category to avoid data leakage

## PCA Analysis
- PCA performed using SVD
- Selected k components to retain 90% variance
- Reduced dimensionality from original feature space

## Model Comparison
Models used:
- Linear Regression
- k-Nearest Neighbors
- Random Forest

Evaluation metrics:
- RMSE
- MAE
- R²

## Conclusion
Random Forest performed best overall. PCA reduced dimensionality but did not significantly improve model performance. The dataset shows some multicollinearity based on the condition number.
