try:
    from sklearn.cluster import KMeans
    print("scikit-learn imported successfully!")
except ImportError as e:
    print(f"Error importing scikit-learn: {e}")
