import pandas as pd # type: ignore
from sklearn.decomposition import PCA # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Adapted from scripts seen in class

# -----------------------------
# 1. Load preprocessed training data
# -----------------------------
train_df = pd.read_csv("../data/train_scaled.csv")  # update path if needed

X = train_df.drop(columns=["Label"]) # Everything except the class column
y = train_df["Label"] 

# -----------------------------
# 2. Run PCA (2 components)
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# -----------------------------
# 3. Prepare colors for plotting
# -----------------------------
# Map each cancer type to a color
unique_labels = sorted(y.unique())
color_map = plt.cm.get_cmap('tab10', len(unique_labels))  # up to 10 distinct colors
colors = [color_map(unique_labels.index(lbl)) for lbl in y]

# -----------------------------
# 4. Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=colors, alpha=0.7)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA of Training Dataset")

# Add legend
handles = [plt.Line2D([], [], marker="o", color=color_map(i), linestyle="", label=str(lbl))
           for i, lbl in enumerate(unique_labels)]
ax.legend(handles=handles, title="Cancer Type", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("../figures/pca_train.png")  # save figure
plt.show()