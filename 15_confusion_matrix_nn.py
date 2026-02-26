#!/usr/bin/env python

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import cancer_nn  # your neural network module

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("../data/test_encoded.csv")

X = df.drop(columns=["Label"]).values
y = df["Label"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

input_size = X.shape[1]
num_classes = len(torch.unique(y))
# -----------------------------
# Load trained model
# -----------------------------
model = cancer_nn.CancerNet(input_size,num_classes)
model.load_state_dict(torch.load("../models/cancer_nn_trained_v1_1epoch.pth"))
model.eval()

# -----------------------------
# Predict
# -----------------------------
with torch.no_grad():
    outputs = model(X)
    preds = torch.argmax(outputs, dim=1)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y.numpy(), preds.numpy())

print("Confusion Matrix:\n")
print(cm)

# -----------------------------
# Plot using matplotlib only
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(cax)
print(num_classes)
num_classes = cm.shape[0]
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix - Test Set')

# Add numbers inside the squares
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='red' if cm[i, j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.savefig("../figures/confusion_matrix.png")
plt.show()