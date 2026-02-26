import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.feature_selection import VarianceThreshold  # type: ignore

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("../data/final_dataset_top10.csv")

X = df.drop(columns=["Label"])
y = df["Label"]

# -----------------------------
# 2. First split: train+val vs test (80/20)
# -----------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# -----------------------------
# 3. Second split: train vs val (70/10 total)
# 0.10 / 0.80 = 0.125
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125,
    stratify=y_temp,
    random_state=42
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)

# -----------------------------
# 4. Convert to numeric
# -----------------------------
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_val   = X_val.apply(pd.to_numeric, errors="coerce")
X_test  = X_test.apply(pd.to_numeric, errors="coerce")

# -----------------------------
# 5. Impute missing values (train mean)
# -----------------------------
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_val   = X_val.fillna(train_means)
X_test  = X_test.fillna(train_means)

# -----------------------------
# 6. Remove zero-variance columns
# -----------------------------
selector = VarianceThreshold(threshold=0.0)
X_train = selector.fit_transform(X_train)
X_val   = selector.transform(X_val)
X_test  = selector.transform(X_test)

# Keep column names for DataFrame
cols_kept = X.columns[selector.get_support()]

# -----------------------------
# 7. Standardize (Z-score)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# -----------------------------
# 8. Convert back to DataFrame
# -----------------------------
X_train = pd.DataFrame(X_train, columns=cols_kept)
X_val   = pd.DataFrame(X_val, columns=cols_kept)
X_test  = pd.DataFrame(X_test, columns=cols_kept)

# -----------------------------
# 9. Reattach labels
# -----------------------------
train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
val_df   = pd.concat([X_val,   y_val.reset_index(drop=True)], axis=1)
test_df  = pd.concat([X_test,  y_test.reset_index(drop=True)], axis=1)

# -----------------------------
# 10. Save to disk
# -----------------------------
train_df.to_csv("../data/train_scaled.csv", index=False)
val_df.to_csv("../data/val_scaled.csv", index=False)
test_df.to_csv("../data/test_scaled.csv", index=False)

print("\nFiles saved successfully.")