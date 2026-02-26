import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# 1. Load preprocessed training data
# -----------------------------
train_df = pd.read_csv("../data/train_scaled.csv")  # update path if needed

X = train_df.drop(columns=["Label"])
y = train_df["Label"]

# -----------------------------
# 2. Define PCA components to test
# -----------------------------
pca_components_list = [10, 50, 100, 200]
n_clusters = 10

# -----------------------------
# 3. Loop through PCA components
# -----------------------------
for n_comp in pca_components_list:
    print(f"\n=== PCA with {n_comp} components ===")
    
    # PCA reduction
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"Explained variance by first 2 PCs: {pca.explained_variance_ratio_[:2].sum():.2%}")
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_pca)
    print(f"KMeans inertia: {kmeans.inertia_:.2f}")
    
    # Optional: t-SNE for 2D visualization
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='pca')
    X_tsne = tsne.fit_transform(X_pca)
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=kmeans.labels_, cmap='tab10', alpha=0.7)
    plt.title(f"KMeans clusters (n_clusters={n_clusters}) after PCA ({n_comp} components)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.colorbar(label="Cluster Label")
    plt.tight_layout()
    plt.savefig(f"../figures/kmeans_pca{n_comp}_tsne.png")
    plt.show()
    
    # Optional: show composition of each cluster
    for cluster in range(n_clusters):
        cluster_labels = y[kmeans.labels_ == cluster]
        counts = Counter(cluster_labels)
        print(f"\nCluster {cluster} composition:")
        for label, count in counts.items():
            print(f"  Label {label}: {count} samples")