import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv("../data/train_scaled.csv")

X = train_df.drop(columns=["Label"])
y = train_df["Label"]

# -----------------------------
# PCA (3 components)
# -----------------------------
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

explained_var = pca.explained_variance_ratio_
print(f"Variance explained by PC1: {explained_var[0]:.2%}")
print(f"Variance explained by PC2: {explained_var[1]:.2%}")
print(f"Variance explained by PC3: {explained_var[2]:.2%}")

# -----------------------------
# Colors for plotting
# -----------------------------
unique_labels = sorted(y.unique())
color_map = plt.cm.get_cmap('tab10', len(unique_labels))
colors = [color_map(unique_labels.index(lbl)) for lbl in y]

# -----------------------------
# 3D Plot
# -----------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=colors, alpha=0.7)

ax.set_xlabel(f"PC1 ({explained_var[0]:.2%})")
ax.set_ylabel(f"PC2 ({explained_var[1]:.2%})")
ax.set_zlabel(f"PC3 ({explained_var[2]:.2%})")
ax.set_title("3D PCA of Training Data")

# Legend
handles = [plt.Line2D([], [], marker="o", color=color_map(i), linestyle="", label=str(lbl))
           for i, lbl in enumerate(unique_labels)]
ax.legend(handles=handles, title="Cancer Type", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("../figures/pca_train_3d.png")
plt.show()