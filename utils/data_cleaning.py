import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df, column, method='median', custom_value=None, correlated_columns=None):
    """Handle missing values using specified method"""
    result_df = df.copy()
    
    if method == 'mean':
        mean_value = df[column].mean()
        result_df[column] = df[column].fillna(mean_value)
    
    elif method == 'median':
        median_value = df[column].median()
        result_df[column] = df[column].fillna(median_value)
    
    elif method == 'remove':
        result_df = df.dropna(subset=[column])
    
    elif method == 'kmeans':
        try:
            # Prepare data for clustering
            if correlated_columns is None or len(correlated_columns) == 0:
                # If no correlated columns provided, use single column K-means
                data_for_clustering = df[[column]].copy()
                missing_mask = data_for_clustering[column].isna()
                clean_data = data_for_clustering[~missing_mask]
            else:
                # Use correlated columns for better K-means clustering
                columns_to_use = [column] + correlated_columns
                data_for_clustering = df[columns_to_use].copy()
                missing_mask = data_for_clustering[column].isna()
                clean_data = data_for_clustering[~missing_mask]
            
            if len(clean_data) == 0:
                raise ValueError("No complete data points available for K-means clustering")
            
            # Standardize the data
            scaler = StandardScaler()
            if len(clean_data.columns) == 1:
                scaled_data = scaler.fit_transform(clean_data.values.reshape(-1, 1))
            else:
                scaled_data = scaler.fit_transform(clean_data)
            
            # Perform K-means clustering
            n_clusters = min(3, len(clean_data) // 2)  # Adjust number of clusters based on data size
            n_clusters = max(2, n_clusters)  # Ensure at least 2 clusters
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(scaled_data)
            
            # For each missing value, find nearest cluster and use its center
            missing_rows = df[missing_mask].index
            if len(missing_rows) > 0:
                for idx in missing_rows:
                    if correlated_columns is None or len(correlated_columns) == 0:
                        # For single column, use mean of cluster centers
                        cluster_centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
                        imputed_value = np.mean(cluster_centers_unscaled)
                        result_df.at[idx, column] = imputed_value
                    else:
                        # Get values of correlated columns for this row
                        row_data = df.loc[idx, correlated_columns].values.reshape(1, -1)
                        # If any correlated values are missing, use the mean of all cluster centers
                        if np.isnan(row_data).any():
                            cluster_centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
                            imputed_value = np.mean(cluster_centers_unscaled[:, 0])  # Use first column (target column)
                        else:
                            # Scale the row data
                            row_scaled = scaler.transform(row_data)
                            # Find nearest cluster
                            cluster = kmeans.predict(row_scaled)[0]
                            # Get the value from cluster center
                            cluster_center_scaled = kmeans.cluster_centers_[cluster].reshape(1, -1)
                            cluster_center_unscaled = scaler.inverse_transform(cluster_center_scaled)
                            imputed_value = cluster_center_unscaled[0, 0]  # Use first column (target column)
                        result_df.at[idx, column] = imputed_value
                        
        except Exception as e:
            # Fallback to median if K-means fails
            print(f"K-means clustering failed: {str(e)}. Falling back to median imputation.")
            median_value = df[column].median()
            result_df[column] = df[column].fillna(median_value)
    
    return result_df

def handle_outliers(df, column, std_multiplier=3.0, method='remove'):
    """
    Detect and handle outliers in the specified column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name to handle outliers
    std_multiplier : float
        Number of standard deviations to use for outlier detection
    method : str
        Method to handle outliers: 'remove' or 'cap'
    
    Returns:
    --------
    tuple(pandas.DataFrame, pandas.Series)
        Processed dataframe and boolean mask of outlier positions
    """
    result_df = df.copy()
    
    # Calculate statistics
    mean = df[column].mean()
    std = df[column].std()
    threshold = std_multiplier * std
    
    # Identify outliers
    outliers = np.abs(df[column] - mean) > threshold
    
    if method == 'remove':
        result_df = result_df[~outliers]
    elif method == 'cap':
        result_df.loc[df[column] > mean + threshold, column] = mean + threshold
        result_df.loc[df[column] < mean - threshold, column] = mean - threshold
    
    return result_df, outliers

def get_correlation_candidates(df, target_column):
    """
    Find columns that might be good candidates for correlation-based imputation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Column name to find correlations for
    
    Returns:
    --------
    list
        List of tuples (column_name, correlation) sorted by absolute correlation
    """
    # Only consider numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove target column from candidates
    candidates = [col for col in numeric_cols if col != target_column]
    
    # Calculate correlations
    correlations = []
    for col in candidates:
        correlation = df[[target_column, col]].corr().iloc[0, 1]
        if not pd.isna(correlation):  # Only include if correlation can be calculated
            correlations.append((col, abs(correlation)))
    
    # Sort by absolute correlation value
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    return correlations
