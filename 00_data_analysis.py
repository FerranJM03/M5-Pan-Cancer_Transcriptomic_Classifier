import pandas as pd # type: ignore
import numpy as np
import os

df = pd.read_csv("../data/Pan-cancer_mRNA.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

#print("\nFirst 5 rows:")
#print(df.head())

df = pd.read_csv("../data/Pan-cancer_label_num.csv")

# Count how many samples per cancer type
counts = df["Label"].value_counts()

print(counts)