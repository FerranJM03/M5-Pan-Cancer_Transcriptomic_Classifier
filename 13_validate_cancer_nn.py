#!/usr/bin/env python

import torch
import pandas as pd # type: ignore
from cancer_nn import CancerNet

# -----------------------------
# Load validation data
# -----------------------------
val_df = pd.read_csv("../../data/First_model/val_encoded.csv")

X_val = torch.tensor(val_df.drop(columns=["Label"]).values, dtype=torch.float32)
y_val = torch.tensor(val_df["Label"].values, dtype=torch.long)

input_size = X_val.shape[1]
num_classes = len(torch.unique(y_val))

# -----------------------------
# Load model
# -----------------------------
model = CancerNet(input_size, num_classes)
model.load_state_dict(torch.load("../../models/cancer_nn_trained_v1_1epoch.pth"))
model.eval()

# -----------------------------
# Evaluate
# -----------------------------
with torch.no_grad():
    outputs = model(X_val)
    preds = torch.argmax(outputs, dim=1)

accuracy = torch.sum(preds == y_val).item() / len(y_val)

print(f"Validation Accuracy: {accuracy:.4f}")