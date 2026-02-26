#!/usr/bin/env python

import pandas as pd # type: ignore

# -----------------------------
# 1. Load datasets
# -----------------------------
train_df = pd.read_csv("../data/train_scaled.csv")
val_df   = pd.read_csv("../data/val_scaled.csv")
test_df  = pd.read_csv("../data/test_scaled.csv")

# -----------------------------
# 2. Create label mapping (FROM TRAIN ONLY)
# -----------------------------
unique_labels = sorted(train_df["Label"].unique())

label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# -----------------------------
# 3. Save mapping to TXT file
# -----------------------------
with open("../data/label_mapping.txt", "w") as f:
    for label, idx in label_to_index.items():
        f.write(f"{label} {idx}\n")

print("Label mapping saved.")

# -----------------------------
# 4. Apply mapping to datasets
# -----------------------------
train_df["Label"] = train_df["Label"].map(label_to_index)
val_df["Label"]   = val_df["Label"].map(label_to_index)
test_df["Label"]  = test_df["Label"].map(label_to_index)

# -----------------------------
# 5. Save encoded datasets
# -----------------------------
train_df.to_csv("../data/train_encoded.csv", index=False)
val_df.to_csv("../data/val_encoded.csv", index=False)
test_df.to_csv("../data/test_encoded.csv", index=False)

print("Encoded datasets saved successfully.")