# ==========================================
# DS239 Project 3 - Final Submission
# AI Impact on Jobs 2030
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================
# SETTINGS (RUN FROM ANYWHERE ON MAC)
# ==========================================

DATA_FILE = os.path.expanduser("~/Downloads/AI_Impact_on_Jobs_2030.csv")
TARGET_COL = "Automation_Probability_2030"


# ==========================================
# CHECK FILE EXISTS
# ==========================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found at: {DATA_FILE}")

print("Using dataset:", DATA_FILE)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst rows:")
print(df.head())


# ==========================================
# MISSING VALUES
# ==========================================

missing = df.isnull().sum()
missing = missing[missing > 0]

print("\nMissing values:")
print(missing.sort_values(ascending=False))


# ==========================================
# CLEANING
# ==========================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

# Fill numeric
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Remove leakage column if exists
if "Risk_Category" in df.columns:
    df = df.drop("Risk_Category", axis=1)


# ==========================================
# ENCODING
# ==========================================

df = pd.get_dummies(df, drop_first=True)


# ==========================================
# FEATURES / TARGET
# ==========================================

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]


# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================
# PCA (SVD)
# ==========================================

U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

explained_variance = S**2 / np.sum(S**2)
cumulative_variance = np.cumsum(explained_variance)

k = np.argmax(cumulative_variance >= 0.90) + 1
X_reduced = U[:, :k] @ np.diag(S[:k])

print("\nPCA components (k):", k)


# Plot PCA
plt.figure(figsize=(8,5))
plt.plot(cumulative_variance)
plt.axhline(0.90, linestyle="--")
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.title("PCA Explained Variance")
plt.show()


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    mae = mean_absolute_error(yte, pred)
    r2 = r2_score(yte, pred)
    return rmse, mae, r2


# ==========================================
# MODELS
# ==========================================

models = {
    "Linear": LinearRegression(),
    "kNN": KNeighborsRegressor(n_neighbors=5),
    "RandomForest": RandomForestRegressor(random_state=42)
}

results = []

for name, model in models.items():
    rmse, mae, r2 = evaluate(model, X_train, X_test, y_train, y_test)
    results.append([name, "No PCA", rmse, mae, r2])

    rmse, mae, r2 = evaluate(model, Xr_train, Xr_test, yr_train, yr_test)
    results.append([name, "With PCA", rmse, mae, r2])

results_df = pd.DataFrame(results, columns=["Model", "PCA", "RMSE", "MAE", "R2"])

print("\nMODEL RESULTS:")
print(results_df)


# ==========================================
# NORMAL EQUATION + CONDITION NUMBER
# ==========================================

X_design = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))

beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y

cond_number = np.linalg.cond(X_design)

print("\nCondition Number:", cond_number)

# Save results
results_df.to_csv("model_results.csv", index=False)

plt.savefig("pca_plot.png")

print("\nFiles saved: model_results.csv, pca_plot.png")