import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
import plotly.express as px
import plotly.graph_objects as go

def perform_pca_analysis(data, n_components):
    """
    Perform PCA analysis on the data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with numeric columns only
    n_components : int
        Number of PCA components to compute
        
    Returns:
    --------
    dict containing:
        - explained_variance: array of explained variance ratios
        - components_df: DataFrame of PCA components
        - transformed_data: PCA-transformed data
        - feature_importance: DataFrame of feature importance in each component
    """
    # Data preprocessing
    numeric_data = data.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(scaled_data)
    
    # Create components DataFrame
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=numeric_data.columns
    )
    
    # Calculate feature importance
    feature_importance = pd.DataFrame(
        np.abs(pca.components_).T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=numeric_data.columns
    )
    
    # Sort features by importance in PC1
    feature_importance['Mean_Importance'] = feature_importance.mean(axis=1)
    feature_importance = feature_importance.sort_values('Mean_Importance', ascending=False)
    
    return {
        'explained_variance': pca.explained_variance_ratio_,
        'components_df': components_df,
        'transformed_data': transformed_data,
        'feature_importance': feature_importance,
        'scaler': scaler,
        'pca_model': pca
    }

def get_feature_engineering_recommendations(data):
    """
    Generate automated feature engineering recommendations with detailed skewness analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
        
    Returns:
    --------
    dict containing:
        - recommendations: list of recommendation dictionaries
        - skewness_summary: DataFrame with skewness statistics
    """
    recommendations = []
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Calculate skewness for all numeric columns
    skewness_stats = []
    for col in numeric_cols:
        try:
            skew_val = data[col].skew()
            skewness_stats.append({
                'feature': col,
                'skewness': skew_val,
                'abs_skewness': abs(skew_val),
                'data_type': str(data[col].dtype)
            })
        except Exception as e:
            continue  # Skip columns that can't be analyzed
    
    skewness_df = pd.DataFrame(skewness_stats).sort_values('abs_skewness', ascending=False)
    
    # Check for high correlation between numeric features
    if len(numeric_cols) >= 2:
        corr_matrix = data[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((numeric_cols[i], numeric_cols[j]))
        
        if high_corr_pairs:
            recommendations.append({
                'type': 'interaction',
                'features': high_corr_pairs,
                'rationale': 'High correlation detected between these features. Consider creating interaction terms.'
            })
    
    # Check for skewed distributions
    for _, row in skewness_df.iterrows():
        if abs(row['skewness']) > 1:
            feature_stats = {
                'mean': data[row['feature']].mean(),
                'median': data[row['feature']].median(),
                'min': data[row['feature']].min(),
                'max': data[row['feature']].max()
            }
            
            recommendations.append({
                'type': 'transformation',
                'feature': row['feature'],
                'skewness': row['skewness'],
                'data_type': row['data_type'],
                'stats': feature_stats,
                'rationale': f"Feature '{row['feature']}' has high skewness ({row['skewness']:.2f}). "
                           f"[Current stats: mean={feature_stats['mean']:.2f}, "
                           f"median={feature_stats['median']:.2f}, "
                           f"range=({feature_stats['min']:.2f}, {feature_stats['max']:.2f})]"
            })
    
    # Suggest polynomial features only for numeric columns with potential non-linear relationships
    if len(numeric_cols) > 0:
        # Filter out highly skewed features and categorical-like numerics
        suitable_numeric_cols = []
        for col in numeric_cols:
            unique_ratio = data[col].nunique() / len(data)
            if abs(data[col].skew()) < 5 and unique_ratio > 0.01:  # Less than 5 skewness and more than 1% unique values
                suitable_numeric_cols.append(col)
        
        if suitable_numeric_cols:
            recommendations.append({
                'type': 'polynomial',
                'features': suitable_numeric_cols,
                'rationale': 'Consider polynomial features to capture non-linear relationships for these numeric columns.'
            })
    
    return {
        'recommendations': recommendations,
        'skewness_summary': skewness_df
    }

def apply_feature_transformation(data, transformation_type, features, **kwargs):
    """
    Apply specified feature transformation
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    transformation_type : str
        Type of transformation ('polynomial', 'log', 'interaction')
    features : list
        List of features to transform
    **kwargs : additional arguments for specific transformations
    
    Returns:
    --------
    pd.DataFrame with transformed features added
    """
    result = data.copy()
    
    if transformation_type == 'polynomial':
        degree = kwargs.get('degree', 2)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        feature_names = [f"{feat}_poly{i}" for feat in features for i in range(2, degree + 1)]
        
        # Apply polynomial transformation
        poly_features = poly.fit_transform(data[features])[:, len(features):]  # Exclude original features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
        
        result = pd.concat([result, poly_df], axis=1)
    
    elif transformation_type == 'log':
        for feat in features:
            # Add small constant to handle zeros
            min_val = data[feat].min()
            offset = 1 if min_val >= 0 else abs(min_val) + 1
            result[f"{feat}_log"] = np.log(data[feat] + offset)
    
    elif transformation_type == 'interaction':
        for feat1, feat2 in features:  # features should be list of tuples for interaction
            result[f"{feat1}_{feat2}_interact"] = data[feat1] * data[feat2]
    
    return result

def evaluate_feature_importance(data, target_col):
    """
    Evaluate feature importance using mutual information
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_col : str
        Name of the target column
        
    Returns:
    --------
    pd.DataFrame with feature importance scores
    """
    features = data.drop(columns=[target_col])
    numeric_features = features.select_dtypes(include=[np.number])
    
    # Calculate mutual information scores
    mi_scores = mutual_info_regression(numeric_features, data[target_col])
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': numeric_features.columns,
        'Importance': mi_scores
    }).sort_values('Importance', ascending=False)
    
    return importance_df
