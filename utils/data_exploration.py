import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_correlations(df, method='pearson'):
    """
    Calculate correlation matrix for numeric columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    method : str
        Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr(method=method)

def calculate_vif(df):
    """
    Calculate Variance Inflation Factor for numeric columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with VIF scores for each numeric column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create a dataframe with scaled features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols
    )
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_cols
    vif_data["VIF"] = [variance_inflation_factor(scaled_features.values, i)
                       for i in range(scaled_features.shape[1])]
    
    return vif_data.sort_values('VIF', ascending=False)

def get_visualization_options(df):
    """Get available visualization options based on column types"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    viz_options = {
        'Scatter Plot': {
            'type': 'numeric-numeric',
            'description': 'Show relationship between numeric variables. Can plot multiple y-axes.',
            'required_cols': (2, 3),  # min, max columns
            'col_types': ['numeric', 'numeric', 'numeric'],
            'supports_sort': True,
            'sort_options': ['x', 'y1', 'y2']
        },
        'Box Plot': {
            'type': 'categorical-numeric',
            'description': 'Show distribution of numeric variable across categories',
            'required_cols': (2, 2),
            'col_types': ['categorical', 'numeric'],
            'supports_sort': True,
            'sort_options': ['category', 'median', 'mean']
        },
        'Bar Plot': {
            'type': 'any',
            'description': 'Show counts or values. Can plot multiple y-axes for comparison.',
            'required_cols': (1, 3),
            'col_types': ['any', 'numeric', 'numeric'],
            'supports_sort': True,
            'sort_options': ['x', 'y1', 'y2', 'value']
        },
        'Histogram': {
            'type': 'numeric',
            'description': 'Show distribution of a numeric variable',
            'required_cols': (1, 1),
            'col_types': ['numeric'],
            'supports_sort': False,
            'sort_options': []
        },
        'Line Plot': {
            'type': 'numeric-numeric',
            'description': 'Show trends between variables. Can plot multiple y-axes.',
            'required_cols': (2, 3),
            'col_types': ['numeric', 'numeric', 'numeric'],
            'supports_sort': True,
            'sort_options': ['x', 'y1', 'y2']
        },
        'Heatmap': {
            'type': 'numeric-matrix',
            'description': 'Show correlation matrix as a heatmap',
            'required_cols': 'all_numeric',
            'col_types': ['numeric'],
            'supports_sort': True,
            'sort_options': ['correlation']
        }
    }
    
    return viz_options

def create_visualization(df, viz_type, columns, **kwargs):
    """
    Create a plotly figure based on visualization type and columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    viz_type : str
        Type of visualization
    columns : list
        List of columns to use
    **kwargs : dict
        Additional parameters for the visualization
        
    Returns:
    --------
    plotly.graph_objs._figure.Figure
        Plotly figure object
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Get color scheme
    NU_RED = '#D41B2C'
    NU_BLACK = '#000000'
    NU_GRAY = '#6A6A6A'
    
    # Get sorting options
    sort_by = kwargs.get('sort_by', None)
    sort_ascending = kwargs.get('sort_ascending', True)
    
    if viz_type == 'Scatter Plot':
        if len(columns) == 2:  # Single y-axis
            df_plot = df.sort_values(sort_by) if sort_by else df
            fig = px.scatter(df_plot, x=columns[0], y=columns[1],
                           title=f'{columns[1]} vs {columns[0]}')
            fig.update_traces(marker_color=NU_RED)
        else:  # Multiple y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            df_plot = df.sort_values(sort_by) if sort_by else df
            
            # Primary y-axis
            fig.add_trace(
                go.Scatter(x=df_plot[columns[0]], y=df_plot[columns[1]], 
                          name=columns[1], mode='markers',
                          marker_color=NU_RED),
                secondary_y=False
            )
            
            # Secondary y-axis
            fig.add_trace(
                go.Scatter(x=df_plot[columns[0]], y=df_plot[columns[2]], 
                          name=columns[2], mode='markers',
                          marker_color=NU_GRAY),
                secondary_y=True
            )
            
            fig.update_layout(
                title=f'Multiple Variables vs {columns[0]}',
                xaxis_title=columns[0],
                yaxis_title=columns[1],
                yaxis2_title=columns[2]
            )
    
    elif viz_type == 'Box Plot':
        df_plot = df.sort_values(sort_by) if sort_by else df
        fig = px.box(df_plot, x=columns[0], y=columns[1],
                    title=f'Distribution of {columns[1]} by {columns[0]}')
        fig.update_traces(marker_color=NU_RED)
    
    elif viz_type == 'Bar Plot':
        if len(columns) == 1:
            value_counts = df[columns[0]].value_counts()
            if sort_by == columns[0]:
                value_counts = value_counts.sort_index(ascending=sort_ascending)
            elif sort_by == 'value':
                value_counts = value_counts.sort_values(ascending=sort_ascending)
            
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f'Counts of {columns[0]}')
            fig.update_traces(marker_color=NU_RED)
        else:
            df_plot = df.sort_values(sort_by if sort_by else columns[0], 
                                   ascending=sort_ascending)
            if len(columns) == 2:  # Single y-axis
                fig = px.bar(df_plot, x=columns[0], y=columns[1],
                           title=f'{columns[1]} by {columns[0]}')
                fig.update_traces(marker_color=NU_RED)
            else:  # Multiple y-axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Primary y-axis
                fig.add_trace(
                    go.Bar(x=df_plot[columns[0]], y=df_plot[columns[1]], 
                          name=columns[1], marker_color=NU_RED),
                    secondary_y=False
                )
                
                # Secondary y-axis
                fig.add_trace(
                    go.Bar(x=df_plot[columns[0]], y=df_plot[columns[2]], 
                          name=columns[2], marker_color=NU_GRAY),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title=f'Multiple Variables by {columns[0]}',
                    xaxis_title=columns[0],
                    yaxis_title=columns[1],
                    yaxis2_title=columns[2],
                    barmode='group'
                )
    
    elif viz_type == 'Histogram':
        fig = px.histogram(df, x=columns[0],
                          title=f'Distribution of {columns[0]}')
        fig.update_traces(marker_color=NU_RED)
    
    elif viz_type == 'Line Plot':
        if len(columns) == 2:  # Single y-axis
            df_plot = df.sort_values(sort_by if sort_by else columns[0], 
                                   ascending=sort_ascending)
            fig = px.line(df_plot, x=columns[0], y=columns[1],
                         title=f'{columns[1]} vs {columns[0]}')
            fig.update_traces(line_color=NU_RED)
        else:  # Multiple y-axes
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            df_plot = df.sort_values(sort_by if sort_by else columns[0], 
                                   ascending=sort_ascending)
            
            # Primary y-axis
            fig.add_trace(
                go.Scatter(x=df_plot[columns[0]], y=df_plot[columns[1]], 
                          name=columns[1], mode='lines',
                          line_color=NU_RED),
                secondary_y=False
            )
            
            # Secondary y-axis
            fig.add_trace(
                go.Scatter(x=df_plot[columns[0]], y=df_plot[columns[2]], 
                          name=columns[2], mode='lines',
                          line_color=NU_GRAY),
                secondary_y=True
            )
            
            fig.update_layout(
                title=f'Multiple Variables vs {columns[0]}',
                xaxis_title=columns[0],
                yaxis_title=columns[1],
                yaxis2_title=columns[2]
            )
    
    elif viz_type == 'Heatmap':
        corr_matrix = calculate_correlations(df[columns])
        if sort_by == 'correlation':
            # Sort by average absolute correlation
            avg_corr = abs(corr_matrix).mean()
            sorted_cols = avg_corr.sort_values(ascending=sort_ascending).index
            corr_matrix = corr_matrix.loc[sorted_cols, sorted_cols]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title='Correlation Heatmap')
    
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")
    
    return fig
