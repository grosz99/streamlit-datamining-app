import streamlit as st
from sklearn.cluster import KMeans
import numpy as np

st.title('Testing scikit-learn')

# Generate some random data
X = np.random.rand(100, 2)

# Create and fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plot the results
st.write("K-Means clustering test successful!")
st.scatter_chart(X, y=kmeans.labels_)
