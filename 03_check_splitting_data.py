import pandas as pd # pyright: ignore[reportMissingModuleSource]

train = pd.read_csv("../data/train_scaled.csv")
val   = pd.read_csv("../data/val_scaled.csv")
test  = pd.read_csv("../data/test_scaled.csv")

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)

# Check first few rows
print(train.head())

# Check mean and std of first 5 genes
print("\nMeans of first 5 genes (Train):")
print(train.iloc[:, :5].mean())

print("\nStd of first 5 genes (Train):")
print(train.iloc[:, :5].std())