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
    dict
        Dictionary containing:
        - 'df': Processed dataframe
        - 'outliers': Boolean mask of outlier positions
        - 'summary': Dictionary with outlier statistics
        - 'code': Python code used for outlier handling
    """
    result_df = df.copy()
    
    # Convert to numeric if not already
    if not pd.api.types.is_numeric_dtype(df[column]):
        try:
            result_df[column] = pd.to_numeric(result_df[column])
        except:
            raise ValueError(f"Column {column} cannot be converted to numeric type")
    
    # Remove NaN values for statistics calculation
    clean_data = result_df[column].dropna()
    
    if len(clean_data) == 0:
        raise ValueError(f"No valid numeric data in column {column}")
    
    # Calculate statistics
    mean = clean_data.mean()
    std = clean_data.std()
    threshold = std_multiplier * std
    
    # Identify outliers
    outliers = np.abs(result_df[column] - mean) > threshold
    n_outliers = outliers.sum()
    
    # Generate summary statistics
    summary = {
        'total_rows': len(df),
        'n_outliers': int(n_outliers),
        'outlier_percentage': float((n_outliers / len(df)) * 100),
        'threshold_upper': float(mean + threshold),
        'threshold_lower': float(mean - threshold)
    }
    
    # Generate detailed implementation code
    code = f"""import pandas as pd
import numpy as np

def detect_and_handle_outliers(df, column='{column}', std_multiplier={std_multiplier}, method='{method}'):
    '''
    Detect and handle outliers in a DataFrame column using the standard deviation method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column : str
        Name of the column to process
    std_multiplier : float
        Number of standard deviations to use as threshold
    method : str
        'remove' or 'cap' outliers
        
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with outliers handled
    '''
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure numeric type
    if not pd.api.types.is_numeric_dtype(result_df[column]):
        result_df[column] = pd.to_numeric(result_df[column], errors='coerce')
    
    # Calculate statistics using non-NaN values
    clean_data = result_df[column].dropna()
    mean = clean_data.mean()
    std = clean_data.std()
    threshold = std_multiplier * std
    
    # Identify outliers
    outliers = np.abs(result_df[column] - mean) > threshold
    
    if method == 'remove':
        # Remove rows with outliers
        result_df = result_df[~outliers]
    elif method == 'cap':
        # Cap outliers at threshold values
        upper_bound = mean + threshold
        lower_bound = mean - threshold
        result_df.loc[result_df[column] > upper_bound, column] = upper_bound
        result_df.loc[result_df[column] < lower_bound, column] = lower_bound
    
    return result_df

# Example usage:
# Load your data
df = pd.read_csv('your_data.csv')  # Replace with your data source

# Process outliers
processed_df = detect_and_handle_outliers(
    df=df,
    column='{column}',
    std_multiplier={std_multiplier},
    method='{method}'
)

# Summary statistics for verification
original_stats = df['{column}'].describe()
processed_stats = processed_df['{column}'].describe()

print("Original Statistics:")
print(original_stats)
print("\\nProcessed Statistics:")
print(processed_stats)"""

    # Handle outliers based on method
    if method == 'remove':
        result_df = result_df[~outliers]
    elif method == 'cap':
        result_df.loc[result_df[column] > mean + threshold, column] = mean + threshold
        result_df.loc[result_df[column] < mean - threshold, column] = mean - threshold
    
    return {
        'df': result_df,
        'outliers': outliers,
        'summary': summary,
        'code': code
    }

def get_correlation_candidates(df, target_column):
    """
    Find columns that are correlated with the target column for better imputation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Column to find correlations for
    
    Returns:
    --------
    dict
        Dictionary containing correlation information and recommendations
    """
    # Handle datetime columns by converting to numeric (timestamp)
    df_numeric = df.copy()
    for col in df.select_dtypes(include=['datetime64']).columns:
        df_numeric[col] = df_numeric[col].astype(np.int64) // 10**9
    
    # Convert categorical columns to numeric using label encoding
    for col in df.select_dtypes(include=['category']).columns:
        df_numeric[col] = pd.Categorical(df_numeric[col]).codes
    
    # Get numeric columns only
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
    if target_column not in numeric_cols:
        return {
            'error': 'Target column must be numeric for correlation analysis',
            'correlations': [],
            'recommendations': [],
            'best_candidates': []
        }
    
    # Calculate correlations
    correlations = []
    for col in numeric_cols:
        if col != target_column:
            correlation = df_numeric[[target_column, col]].corr().iloc[0, 1]
            if not np.isnan(correlation):
                correlations.append((col, abs(correlation)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Generate recommendations
    strong_correlations = [c for c in correlations if c[1] >= 0.7]
    moderate_correlations = [c for c in correlations if 0.5 <= c[1] < 0.7]
    
    recommendations = []
    if strong_correlations:
        recommendations.append(f"Strong correlations found with: {', '.join(c[0] for c in strong_correlations)}")
        recommendations.append("These variables are excellent candidates for K-means clustering")
    elif moderate_correlations:
        recommendations.append(f"Moderate correlations found with: {', '.join(c[0] for c in moderate_correlations)}")
        recommendations.append("These variables might be useful for K-means clustering")
    else:
        recommendations.append("No strong correlations found. Consider using simple imputation methods instead")
    
    # Add Netflix-specific recommendations
    if 'Time Watched' in df.columns:
        recommendations.append("\nNetflix-specific recommendations:")
        if 'Completed' in df.columns:
            recommendations.append("- Consider using 'Completed' status to help predict viewing times")
        if 'Time of Day' in df.columns:
            recommendations.append("- 'Time of Day' might indicate viewing patterns")
        if 'Season' in df.columns and 'Episode' in df.columns:
            recommendations.append("- 'Season' and 'Episode' numbers might show binge-watching patterns")
    
    return {
        'correlations': correlations,
        'recommendations': recommendations,
        'best_candidates': [c[0] for c in strong_correlations] or [c[0] for c in moderate_correlations[:2]]
    }

def get_imputation_code(column, method='median', correlated_columns=None):
    """
    Generate Python code for the imputation method used
    
    Parameters:
    -----------
    column : str
        Column name to impute
    method : str
        Imputation method ('mean', 'median', 'mode', 'remove', 'kmeans')
    correlated_columns : list, optional
        List of correlated columns for K-means imputation
    
    Returns:
    --------
    str
        Generated Python code for the imputation method
    """
    if method == 'mean':
        code = f"""# Mean imputation for column '{column}'
import pandas as pd
import numpy as np

def impute_with_mean(df, column='{column}'):
    '''
    Impute missing values with column mean
    '''
    result = df.copy()
    column_mean = result[column].mean()
    result[column] = result[column].fillna(column_mean)
    return result

# Example usage:
# data = pd.read_csv('your_data.csv')
# result = impute_with_mean(data, '{column}')"""

    elif method == 'median':
        code = f"""# Median imputation for column '{column}'
import pandas as pd
import numpy as np

def impute_with_median(df, column='{column}'):
    '''
    Impute missing values with column median
    '''
    result = df.copy()
    column_median = result[column].median()
    result[column] = result[column].fillna(column_median)
    return result

# Example usage:
# data = pd.read_csv('your_data.csv')
# result = impute_with_median(data, '{column}')"""

    elif method == 'mode':
        code = f"""# Mode imputation for column '{column}'
import pandas as pd
import numpy as np

def impute_with_mode(df, column='{column}'):
    '''
    Impute missing values with column mode (most frequent value)
    '''
    result = df.copy()
    column_mode = result[column].mode().iloc[0]  # Get first mode if multiple exist
    result[column] = result[column].fillna(column_mode)
    return result

# Example usage:
# data = pd.read_csv('your_data.csv')
# result = impute_with_mode(data, '{column}')"""

    elif method == 'remove':
        code = f"""# Remove rows with missing values in column '{column}'
import pandas as pd
import numpy as np

def remove_missing_values(df, column='{column}'):
    '''
    Remove rows with missing values in specified column
    '''
    result = df.copy()
    original_rows = len(result)
    result = result.dropna(subset=[column])
    return result

# Example usage:
# data = pd.read_csv('your_data.csv')
# result = remove_missing_values(data, '{column}')"""

    elif method == 'kmeans':
        col_list = f"['{column}']"
        if correlated_columns:
            col_list = f"['{column}', '" + "', '".join(correlated_columns) + "']"
        
        code = f"""# K-means imputation for column '{column}'
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def impute_with_kmeans(df, target_column='{column}', columns_to_use={col_list}, n_clusters=3):
    '''
    Impute missing values using K-means clustering
    '''
    result = df.copy()
    data_for_clustering = result[columns_to_use].copy()
    missing_mask = data_for_clustering[target_column].isna()
    
    if missing_mask.all():
        raise ValueError("All values are missing in target column")
    
    complete_cases = data_for_clustering[~missing_mask]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(complete_cases)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    
    def get_cluster_mean(row):
        if row[target_column] is not pd.NA:
            return row[target_column]
        
        row_data = row[columns_to_use[1:]].values.reshape(1, -1)
        if not np.isnan(row_data).any():
            row_scaled = scaler.transform(row_data)
            cluster = kmeans.predict(row_scaled)[0]
            cluster_mask = kmeans.labels_ == cluster
            return complete_cases[target_column][cluster_mask].mean()
        return None
    
    result.loc[missing_mask, target_column] = result[missing_mask].apply(
        get_cluster_mean, axis=1
    )
    return result

# Example usage:
# data = pd.read_csv('your_data.csv')
# result = impute_with_kmeans(
#     data,
#     target_column='{column}',
#     columns_to_use={col_list},
#     n_clusters=3
# )"""

    return code

def get_data_profile(df, original_df=None):
    """
    Generate a comprehensive profile of the dataset, comparing with original if provided.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Current dataset to profile
    original_df : pandas.DataFrame, optional
        Original dataset to compare with
        
    Returns:
    --------
    dict : Profile information including data types, missing values, and cleaning status
    """
    profile = {}
    
    for column in df.columns:
        col_info = {
            'data_type': str(df[column].dtype),
            'unique_values': df[column].nunique(),
            'missing_values': df[column].isnull().sum(),
            'missing_percentage': (df[column].isnull().sum() / len(df)) * 100,
            'needs_cleaning': df[column].isnull().any(),
        }
        
        # Add basic statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            col_info.update({
                'mean': df[column].mean() if not df[column].isnull().all() else None,
                'std': df[column].std() if not df[column].isnull().all() else None,
                'min': df[column].min() if not df[column].isnull().all() else None,
                'max': df[column].max() if not df[column].isnull().all() else None
            })
        
        # Compare with original dataset if provided
        if original_df is not None and column in original_df.columns:
            orig_missing = original_df[column].isnull().sum()
            col_info['cleaning_status'] = {
                'original_missing': orig_missing,
                'current_missing': col_info['missing_values'],
                'values_cleaned': orig_missing - col_info['missing_values'],
                'is_fully_cleaned': col_info['missing_values'] == 0,
                'cleaning_percentage': ((orig_missing - col_info['missing_values']) / orig_missing * 100) if orig_missing > 0 else 100
            }
        
        profile[column] = col_info
    
    return profile

def format_profile_for_display(profile):
    """
    Format the profile data for display in Streamlit
    """
    display_data = []
    
    for column, info in profile.items():
        row = {
            'Column': column,
            'Data Type': info['data_type'],
            'Unique Values': info['unique_values'],
            'Missing Values': info['missing_values'],
            'Missing %': f"{info['missing_percentage']:.2f}%",
            'Needs Cleaning': '✗' if info['needs_cleaning'] else '✓'
        }
        
        # Add cleaning status if available
        if 'cleaning_status' in info:
            status = info['cleaning_status']
            row.update({
                'Values Cleaned': status['values_cleaned'],
                'Cleaning Progress': f"{status['cleaning_percentage']:.2f}%",
                'Status': '✓ Fully Cleaned' if status['is_fully_cleaned'] else '⚠ Needs Cleaning'
            })
        
        display_data.append(row)
    
    return pd.DataFrame(display_data)
