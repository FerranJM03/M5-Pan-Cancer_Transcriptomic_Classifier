import pandas as pd

train = pd.read_csv("../data/train_encoded.csv")
val   = pd.read_csv("../data/val_encoded.csv")

duplicates = pd.merge(train, val, how="inner")

print("Number of overlapping rows:", len(duplicates))