# ===============================
# DS239 Project 3
# Submission 1 – Data Cleaning + PCA
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ===============================
# 1. Load Dataset
# ===============================

df = pd.read_csv("train.csv")

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())


# ===============================
# 2. Missing Value Analysis
# ===============================

missing = df.isnull().sum()
missing = missing[missing > 0]

print("\nColumns with missing values:")
print(missing.sort_values(ascending=False))


# ===============================
# 3. Define Target
# ===============================

target_col = "SalePrice"


# ===============================
# 4. Identify Numeric + Categorical Columns
# ===============================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if target_col in numeric_cols:
    numeric_cols.remove(target_col)


# ===============================
# 5. Handle Missing Values
# ===============================

# Fill numeric columns with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])


# ===============================
# 6. One-Hot Encode Categorical Variables
# ===============================

df = pd.get_dummies(df, drop_first=True)


# ===============================
# 7. Define Features and Target
# ===============================

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]


# ===============================
# 8. Feature Scaling
# ===============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# 9. PCA via SVD
# ===============================

U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

explained_variance = S**2 / np.sum(S**2)
cumulative_variance = np.cumsum(explained_variance)


# ===============================
# 10. Choose k (retain 90% variance)
# ===============================

k = np.argmax(cumulative_variance >= 0.90) + 1

print("\nNumber of PCA components chosen (k):", k)


# ===============================
# 11. Reduced Feature Representation
# ===============================

X_reduced = U[:, :k] @ np.diag(S[:k])

print("\nOriginal feature shape:", X_scaled.shape)
print("Reduced feature shape:", X_reduced.shape)


# ===============================
# 12. Explained Variance Plot
# ===============================

plt.figure(figsize=(8,5))
plt.plot(cumulative_variance)
plt.axhline(0.90, color="red", linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.show()
