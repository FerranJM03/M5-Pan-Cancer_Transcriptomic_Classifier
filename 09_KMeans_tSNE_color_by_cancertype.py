import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# 1. Load preprocessed training data
# -----------------------------
train_df = pd.read_csv("../data/train_scaled.csv")  # path to your scaled train data

X = train_df.drop(columns=["Label"])
y = train_df["Label"]

# -----------------------------
# 2. Define PCA components to test
# -----------------------------
pca_components_list = [10, 50, 100, 200]
n_clusters = 10  # for KMeans

# -----------------------------
# 3. Loop through PCA components
# -----------------------------
for n_comp in pca_components_list:
    print(f"\n=== PCA with {n_comp} components ===")
    
    # PCA reduction
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"Explained variance by first 2 PCs: {pca.explained_variance_ratio_[:2].sum():.2%}")
    
    # KMeans clustering (optional, for analysis)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_pca)
    print(f"KMeans inertia: {kmeans.inertia_:.2f}")
    
    # t-SNE for 2D visualization
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='pca')
    X_tsne = tsne.fit_transform(X_pca)
    
    # -----------------------------
    # 4. Plot colored by true labels
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10,8))
    
    # Map each cancer type to a color
    unique_labels = sorted(y.unique())
    color_map = plt.cm.get_cmap('tab10', len(unique_labels))  # up to 10 cancer types
    colors = [color_map(unique_labels.index(lbl)) for lbl in y]
    
    scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=colors, alpha=0.7)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title(f"t-SNE visualization (PCA {n_comp} components) colored by cancer type")
    
    # Add legend with cancer types
    handles = [plt.Line2D([], [], marker="o", color=color_map(i), linestyle="", label=str(lbl))
               for i, lbl in enumerate(unique_labels)]
    ax.legend(handles=handles, title="Cancer Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"../figures/tsne_by_cancer_pca{n_comp}.png")
    plt.show()
    
    # -----------------------------
    # 5. Optional: show cluster composition vs cancer type
    # -----------------------------
    for cluster in range(n_clusters):
        cluster_labels = y[kmeans.labels_ == cluster]
        counts = Counter(cluster_labels)
        print(f"\nCluster {cluster} composition:")
        for label, count in counts.items():
            print(f"  Label {label}: {count} samples")