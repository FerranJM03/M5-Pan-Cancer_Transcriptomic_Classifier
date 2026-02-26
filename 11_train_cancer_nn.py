#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd # type: ignore
from cancer_nn import CancerNet

torch.manual_seed(42)

# -----------------------------
# 1. Load encoded training data
# -----------------------------
train_df = pd.read_csv("../../data/First_model/train_encoded.csv")

X = train_df.drop(columns=["Label"]).values
y = train_df["Label"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

input_size = X.shape[1]
num_classes = len(torch.unique(y))

# -----------------------------
# 2. Model
# -----------------------------
model = CancerNet(input_size, num_classes)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

batch_size = 128
epochs = 1
n_samples = X.shape[0]
n_batches = n_samples // batch_size + 1

# -----------------------------
# 3. Training Loop
# -----------------------------
for epoch in range(epochs):

    epoch_loss = 0
    correct = 0

    for i in range(n_batches):

        i1 = i * batch_size
        i2 = min((i+1) * batch_size, n_samples)

        xb = X[i1:i2]
        yb = y[i1:i2]

        optimizer.zero_grad()
        outputs = model(xb)

        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += torch.sum(preds == yb).item()

    accuracy = correct / n_samples
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {accuracy:.4f}")

# -----------------------------
# 4. Save model
# -----------------------------
torch.save(model.state_dict(), "../../models/cancer_nn_trained_v1_1epoch.pth")
print("Model saved successfully.")