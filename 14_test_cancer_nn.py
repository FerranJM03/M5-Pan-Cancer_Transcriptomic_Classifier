#!/usr/bin/env python

import torch
import pandas as pd
from cancer_nn import CancerNet

test_df = pd.read_csv("../../data/First_model/test_encoded.csv")

X_test = torch.tensor(test_df.drop(columns=["Label"]).values, dtype=torch.float32)
y_test = torch.tensor(test_df["Label"].values, dtype=torch.long)

input_size = X_test.shape[1]
num_classes = len(torch.unique(y_test))

model = CancerNet(input_size, num_classes)
model.load_state_dict(torch.load("../../models/cancer_nn_trained_v1_1epoch.pth"))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)

accuracy = torch.sum(preds == y_test).item() / len(y_test)

print(f"Test Accuracy: {accuracy:.4f}")