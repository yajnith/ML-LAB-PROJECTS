import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# -------------------------------------------------
# 1. Create sample data
# -------------------------------------------------
# We generate a synthetic dataset with 3 clusters
X, y_true = make_blobs(
    n_samples=300,      # total points
    centers=3,          # actual clusters in data
    cluster_std=0.60,   # how spread out the clusters are
    random_state=42
)

# -------------------------------------------------
# 2. Apply K-Means
# -------------------------------------------------
k = 3  # number of clusters we want

kmeans = KMeans(
    n_clusters=k,
    init="k-means++",   # smart initialization
    n_init=10,          # run k-means 10 times with different centroid seeds
    max_iter=300,       # maximum iterations for a single run
    random_state=42
)

kmeans.fit(X)          # run the algorithm

# Cluster centers (centroids)
centers = kmeans.cluster_centers_

# Cluster labels for each point (0, 1, 2, ...)
labels = kmeans.labels_

print("Cluster centers (centroids):")
print(centers)
print("\nFirst 10 points and their assigned cluster labels:")
for i in range(10):
    print(f"Point {X[i]} --> Cluster {labels[i]}")

# -------------------------------------------------
# 3. Visualize the clusters
# -------------------------------------------------
plt.figure(figsize=(8, 6))

# Plot each cluster with a different colo
