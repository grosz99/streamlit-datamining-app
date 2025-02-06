import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import plotly.express as px

st.title('Scikit-learn Test in Streamlit')

# Sidebar controls
st.sidebar.header('Clustering Parameters')
n_clusters = st.sidebar.slider('Number of Clusters', 2, 8, 4)
n_samples = st.sidebar.slider('Number of Samples', 100, 1000, 300)
noise = st.sidebar.slider('Noise Level', 0.1, 2.0, 0.6)

# Generate sample data
X, y = make_blobs(
    n_samples=n_samples,
    centers=n_clusters,
    cluster_std=noise,
    random_state=42
)

# Create DataFrame for plotting
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Create interactive plot
fig = px.scatter(
    df,
    x='Feature 1',
    y='Feature 2',
    color='Cluster',
    title=f'K-Means Clustering (k={n_clusters})',
    template='plotly_white'
)

# Add cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
fig.add_scatter(
    x=centers[:, 0],
    y=centers[:, 1],
    mode='markers',
    marker=dict(
        color='black',
        size=15,
        symbol='x'
    ),
    name='Cluster Centers'
)

st.plotly_chart(fig, use_container_width=True)

# Show cluster information
st.subheader('Cluster Information')
cluster_sizes = df['Cluster'].value_counts().sort_index()
cluster_info = pd.DataFrame({
    'Cluster': cluster_sizes.index,
    'Size': cluster_sizes.values,
    'Percentage': (cluster_sizes.values / len(df) * 100).round(2)
})
st.dataframe(cluster_info)

# Verify sklearn version
import sklearn
st.sidebar.info(f'scikit-learn version: {sklearn.__version__}')
