import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df, column, method='median', custom_value=None, correlated_columns=None, n_clusters=3):
    """Handle missing values using specified method"""
    result_df = df.copy()
    
    try:
        if method == 'mean':
            result_df[column].fillna(result_df[column].mean(), inplace=True)
        
        elif method == 'median':
            result_df[column].fillna(result_df[column].median(), inplace=True)
        
        elif method == 'mode':
            result_df[column].fillna(result_df[column].mode()[0], inplace=True)
        
        elif method == 'remove':
            result_df.dropna(subset=[column], inplace=True)
        
        elif method == 'kmeans':
            if not correlated_columns:
                raise ValueError("No correlated columns provided for k-means imputation")
            
            # Get complete cases for clustering
            data_for_clustering = result_df[[column] + correlated_columns].copy()
            missing_mask = data_for_clustering[column].isna()
            complete_cases = data_for_clustering[~missing_mask]
            
            if len(complete_cases) == 0:
                raise ValueError("No complete cases available for k-means imputation")
            
            # Standardize the data
            scaler = StandardScaler()
            if len(complete_cases) > 0:
                scaled_data = scaler.fit_transform(complete_cases)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(scaled_data)
            
            # For each row with missing target value
            for idx in result_df[missing_mask].index:
                # Get the row's features
                row_data = result_df.loc[idx, correlated_columns].values.reshape(1, -1)
                if not np.isnan(row_data).any():  # Only if we have all feature values
                    # Scale the features
                    row_scaled = scaler.transform(row_data)
                    # Predict cluster
                    cluster = kmeans.predict(row_scaled)[0]
                    # Find mean of target variable in this cluster
                    cluster_mask = kmeans.labels_ == cluster
                    cluster_mean = complete_cases[column][cluster_mask].mean()
                    # Impute the missing value
                    result_df.loc[idx, column] = cluster_mean
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
    except Exception as e:
        raise Exception(f"Error in handle_missing_values: {str(e)}")
    
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

def get_imputation_code(column, method, features=None, n_clusters=None):
    """Generate code for the selected imputation method"""
    
    if method == "kmeans":
        if not features:
            return "Error: No features selected for k-means imputation"
        
        features_str = ", ".join([f"'{f}'" for f in features])
        code = f"""# K-means imputation for {column}
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
features = [{features_str}]
X = df[features].copy()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-means
kmeans = KMeans(n_clusters={n_clusters}, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate mean of target variable per cluster
cluster_means = df.groupby('Cluster')['{column}'].transform('mean')

# Fill missing values with cluster means
df['{column}'].fillna(cluster_means, inplace=True)

# Drop temporary cluster column
df.drop('Cluster', axis=1, inplace=True)"""
        
    elif method == "mean":
        code = f"df['{column}'].fillna(df['{column}'].mean(), inplace=True)"
    elif method == "median":
        code = f"df['{column}'].fillna(df['{column}'].median(), inplace=True)"
    elif method == "mode":
        code = f"df['{column}'].fillna(df['{column}'].mode()[0], inplace=True)"
    elif method == "remove":
        code = f"df.dropna(subset=['{column}'], inplace=True)"
    else:
        code = "# Method not recognized"
    
    return code

def get_data_profile(df, original_df=None):
    """
    Generate comprehensive data profile with comparisons
    """
    profile = {
        'basic': {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isna().sum().sum(),
            'duplicates': df.duplicated().sum()
        },
        'dtypes': df.dtypes.to_dict(),
        'stats': {},
        'changes': {}
    }
    
    if original_df is not None:
        profile['changes'] = {
            'rows_removed': len(original_df) - len(df),
            'columns_removed': len(original_df.columns) - len(df.columns),
            'missing_values_removed': original_df.isna().sum().sum() - df.isna().sum().sum()
        }
    
    # Numeric columns stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        profile['stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skew': df[col].skew()
        }
    
    # Categorical columns stats
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    for col in cat_cols:
        profile['stats'][col] = {
            'unique': df[col].nunique(),
            'top_value': df[col].mode()[0] if len(df[col]) > 0 else None,
            'top_freq': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
        }
    
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

def handle_missing_values(df, column, method='median', custom_value=None, correlated_columns=None, n_clusters=3):
    """
    Handle missing values in the specified column using the selected method
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name to handle missing values
    method : str
        Method to use ('mean', 'median', 'mode', 'remove', 'k-means')
    custom_value : float, optional
        Custom value to use for imputation
    correlated_columns : list, optional
        List of correlated columns to use for k-means clustering
    n_clusters : int
        Number of clusters for k-means method
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing values handled
    """
    # Create a copy of the dataframe
    result = df.copy()
    
    if method == 'k-means':
        if not correlated_columns:
            raise ValueError("Features must be specified for k-means imputation")
        
        # Select features for clustering
        X = result[correlated_columns].copy()
        
        # Standardize the features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        result['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate mean of target variable per cluster
        cluster_means = result.groupby('Cluster')[column].transform('mean')
        
        # Fill missing values with cluster means
        result[column].fillna(cluster_means, inplace=True)
        
        # Drop temporary cluster column
        result.drop('Cluster', axis=1, inplace=True)
        
    elif method == 'mean':
        result[column].fillna(result[column].mean(), inplace=True)
    elif method == 'median':
        result[column].fillna(result[column].median(), inplace=True)
    elif method == 'mode':
        result[column].fillna(result[column].mode()[0], inplace=True)
    elif method == 'remove':
        result.dropna(subset=[column], inplace=True)
    
    return result
