import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load preprocessed training data
# -----------------------------
train_df = pd.read_csv("../data/train_scaled.csv")  # update path if needed

X = train_df.drop(columns=["Label"])
y = train_df["Label"]  # original labels, for reference only

# -----------------------------
# 2. Run KMeans
# -----------------------------
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

print("KMeans inertia:", kmeans.inertia_)
print("Cluster centers shape:", kmeans.cluster_centers_.shape)

# -----------------------------
# 3. Reduce to 2D for plotting
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# -----------------------------
# 4. Plot clusters
# -----------------------------
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_, cmap='tab10', alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"KMeans Clustering ({n_clusters} clusters) on PCA-reduced data")

# Add legend for clusters
handles = [plt.Line2D([], [], marker="o", color=plt.cm.tab10(i), linestyle="", label=f"Cluster {i}")
           for i in range(n_clusters)]
ax.legend(handles=handles, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("../figures/kmeans_train.png")
plt.show()

# -----------------------------
# 5. Optional: compare clusters with true labels
# -----------------------------
from collections import Counter
for cluster in range(n_clusters):
    cluster_labels = y[kmeans.labels_ == cluster]
    counts = Counter(cluster_labels)
    print(f"\nCluster {cluster} composition:")
    for label, count in counts.items():
        print(f"  Label {label}: {count} samples")