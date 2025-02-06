import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

print("Step 1: Testing imports... OK")

# Generate sample data
print("\nStep 2: Generating sample data...")
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
print("Data generation successful! OK")

# Standardize the features
print("\nStep 3: Testing StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("StandardScaler working! OK")

# Perform K-means clustering
print("\nStep 4: Testing KMeans clustering...")
kmeans = KMeans(n_clusters=4, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)
print("KMeans clustering successful! OK")

print("\nAll tests passed! scikit-learn is working correctly!")
