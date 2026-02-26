import pandas as pd # type: ignore

# -----------------------------
# 1. Load data
# -----------------------------
labels = pd.read_csv("../data/Pan-cancer_label_num.csv")
mrna = pd.read_csv("../data/Pan-cancer_mRNA.csv")

# -----------------------------
# 2. Prepare mRNA
# -----------------------------
mrna = mrna.set_index("sample")
mrna_t = mrna.T
mrna_t.index.name = "SampleID"
mrna_t.reset_index(inplace=True)

# -----------------------------
# 3. Attach labels (order-based)
# -----------------------------
print("mRNA samples:", len(mrna_t))
print("Label rows:", len(labels))

mrna_t["Label"] = labels["Label"].values

merged = mrna_t

# -----------------------------
# 4. Select top 10 cancer types
# -----------------------------
top10 = (
    merged["Label"]
    .value_counts()
    .nlargest(10)
    .index
)

filtered = merged[merged["Label"].isin(top10)]

# -----------------------------
# 5. Final output
# -----------------------------
print("Final shape:", filtered.shape)
print("\nSamples per cancer type:")
print(filtered["Label"].value_counts().sort_index())

filtered.to_csv("../data/final_dataset_top10.csv", index=False)