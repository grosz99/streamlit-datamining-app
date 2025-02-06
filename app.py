import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_cleaning import handle_missing_values, handle_outliers, get_correlation_candidates
from utils.data_exploration import (
    calculate_correlations,
    calculate_vif,
    get_visualization_options,
    create_visualization
)
import os

# Northeastern University colors
NU_RED = '#D41B2C'
NU_BLACK = '#000000'
NU_GRAY = '#6A6A6A'

# Custom plotly template
custom_template = {
    'layout': {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'color': NU_BLACK},
        'title': {'font': {'color': NU_BLACK}},
        'xaxis': {'gridcolor': '#EEEEEE', 'linecolor': NU_GRAY},
        'yaxis': {'gridcolor': '#EEEEEE', 'linecolor': NU_GRAY}
    }
}

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

# Sidebar
st.sidebar.title('Data Mining App')

# Dataset selection
st.sidebar.subheader("Choose Dataset")
dataset_option = st.sidebar.radio(
    "Select data source",
    ["Use Sample Dataset", "Upload Your Own Dataset"]
)

if dataset_option == "Use Sample Dataset":
    if st.session_state.data is None:
        sample_data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
        st.session_state.data = pd.read_csv(sample_data_path)
        st.session_state.cleaned_data = st.session_state.data.copy()
else:
    # File upload
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.cleaned_data = st.session_state.data.copy()

# Page selection
page = st.sidebar.radio('Select Lab', ['1: Data Cleaning', '2: Data Exploration'])

if page == '1: Data Cleaning':
    st.title('Data Cleaning Lab')
    if st.session_state.data is not None:
        # Column selection
        selected_col = st.selectbox('Choose column', st.session_state.data.columns)
        
        # Show column statistics
        st.write(f"Column: {selected_col}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Missing Values", st.session_state.data[selected_col].isna().sum())
        with col2:
            st.metric("Unique Values", st.session_state.data[selected_col].nunique())
        with col3:
            dtype = st.session_state.data[selected_col].dtype
            st.metric("Data Type", str(dtype))
        with col4:
            st.metric("Total Rows", len(st.session_state.data))
        
        # Show distribution using histogram for numeric columns
        if pd.api.types.is_numeric_dtype(st.session_state.data[selected_col]):
            st.subheader("Distribution Plot")
            fig = px.histogram(
                st.session_state.data,
                x=selected_col,
                nbins=20,
                title=f"Distribution of {selected_col}"
            )
            fig.update_traces(marker_color=NU_RED)
            fig.update_layout(
                template=custom_template,
                showlegend=False,
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Create tabs for different operations
        tab1, tab2 = st.tabs(["Missing Values", "Outliers"])
        
        with tab1:
            st.subheader("Handle Missing Values")
            
            # Determine column type and available methods
            is_numeric = pd.api.types.is_numeric_dtype(st.session_state.data[selected_col])
            
            # Method selection with descriptions
            method_descriptions = {
                'mean': """
                **Best for**: Continuous numeric data with normal distribution
                
                **How it works**: Replaces missing values with the arithmetic mean of the column.
                This method is particularly useful when:
                - The data is normally distributed
                - Outliers don't significantly impact the mean
                - You want to preserve the exact average of the data
                """,
                'median': """
                **Best for**: Continuous numeric data with potential outliers
                
                **How it works**: Replaces missing values with the median value of the column.
                This method is particularly useful when:
                - The data has outliers
                - The distribution is skewed
                - You want a more robust central tendency measure
                """,
                'kmeans': """
                **Best for**: Continuous numeric data with patterns and correlations
                
                **How it works**: Uses K-means clustering to group similar data points and imputes missing values based on cluster centers.
                This method is particularly effective when:
                - The data has natural groupings or patterns
                - There are correlations with other variables
                - Simple statistical methods might not capture the data structure
                """,
                'remove': """
                **Best for**: Cases where data quality is critical
                
                **How it works**: Removes rows with missing values in the selected column.
                This method is useful when:
                - You need completely clean data
                - You have enough data to afford removing rows
                - Missing values might introduce bias
                """
            }
            
            # Method selection
            available_methods = ['mean', 'median', 'remove']
            if is_numeric:
                available_methods.append('kmeans')
            
            method = st.selectbox(
                'Select method',
                available_methods
            )
            
            # Show method description
            st.markdown(method_descriptions[method])
            
            # Additional inputs based on method
            correlated_columns = None
            if method == 'kmeans' and is_numeric:
                st.subheader("Correlation Analysis")
                correlations = get_correlation_candidates(st.session_state.data, selected_col)
                
                if correlations:
                    # Create correlation plot
                    corr_data = pd.DataFrame(correlations, columns=['Column', 'Correlation'])
                    fig = px.bar(
                        corr_data,
                        x='Column',
                        y='Correlation',
                        title='Correlation with Selected Column',
                        template=custom_template
                    )
                    fig.update_traces(marker_color=NU_RED)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Allow user to select correlated columns for K-means
                    selected_correlations = st.multiselect(
                        'Select correlated columns to use for K-means clustering',
                        [col for col, corr in correlations if corr > 0.1],  # Only show columns with correlation > 0.1
                        help='Select columns that have meaningful correlations to improve K-means clustering'
                    )
                    correlated_columns = selected_correlations if selected_correlations else None
                else:
                    st.warning("No significant correlations found with other numeric columns.")
            
            # Process button
            if st.button('Process Missing Values'):
                try:
                    # Apply the selected method
                    st.session_state.cleaned_data = handle_missing_values(
                        st.session_state.cleaned_data,
                        selected_col,
                        method=method,
                        correlated_columns=correlated_columns
                    )
                    st.success('Missing values processed successfully!')
                    
                    # Show sample code
                    st.subheader("Sample Code")
                    code = f"""# Import required libraries
import pandas as pd
import numpy as np

# Assuming your dataframe is called 'df'
if '{method}' == 'mean':
    mean_value = df['{selected_col}'].mean()
    df['{selected_col}'] = df['{selected_col}'].fillna(mean_value)
elif '{method}' == 'median':
    median_value = df['{selected_col}'].median()
    df['{selected_col}'] = df['{selected_col}'].fillna(median_value)
elif '{method}' == 'kmeans':
    from sklearn.cluster import KMeans
    clean_data = df.dropna(subset=['{selected_col}'])
    kmeans = KMeans(n_clusters=2).fit(clean_data[['{selected_col}']])
    # Process missing values using cluster centers
    # ... additional code for K-means imputation
elif '{method}' == 'remove':
    df = df.dropna(subset=['{selected_col}'])
"""
                    
                    st.code(code, language='python')
                    
                    # Show before/after comparison
                    st.subheader("Before and After Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Before:")
                        st.write(st.session_state.data[selected_col].isna().sum(), "missing values")
                        if is_numeric:
                            fig = px.histogram(
                                st.session_state.data,
                                x=selected_col,
                                nbins=20,
                                title="Before"
                            )
                            fig.update_traces(marker_color=NU_RED)
                            fig.update_layout(template=custom_template, showlegend=False, bargap=0.1)
                            st.plotly_chart(fig)
                    with col2:
                        st.write("After:")
                        st.write(st.session_state.cleaned_data[selected_col].isna().sum(), "missing values")
                        if is_numeric:
                            fig = px.histogram(
                                st.session_state.cleaned_data,
                                x=selected_col,
                                nbins=20,
                                title="After"
                            )
                            fig.update_traces(marker_color=NU_RED)
                            fig.update_layout(template=custom_template, showlegend=False, bargap=0.1)
                            st.plotly_chart(fig)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
        with tab2:
            if pd.api.types.is_numeric_dtype(st.session_state.data[selected_col]):
                st.subheader("Handle Outliers")
                
                # Outlier detection settings
                std_multiplier = st.slider('Standard Deviation Multiplier', 1.0, 5.0, 3.0, 0.1,
                                        help='Number of standard deviations from mean to consider as outlier')
                
                # Method selection
                method = st.selectbox(
                    'Select method',
                    ['remove', 'cap']
                )
                
                # Process button
                if st.button('Process Outliers'):
                    try:
                        # Apply outlier detection and handling
                        st.session_state.cleaned_data, outlier_mask = handle_outliers(
                            st.session_state.cleaned_data,
                            selected_col,
                            std_multiplier=std_multiplier,
                            method=method
                        )
                        
                        st.success('Outliers processed successfully!')
                        
                        # Show sample code
                        st.subheader("Sample Code")
                        code = f"""# Import required libraries
import pandas as pd
import numpy as np

# Assuming your dataframe is called 'df'
# Calculate statistics
mean = df['{selected_col}'].mean()
std = df['{selected_col}'].std()
threshold = {std_multiplier} * std

# Identify outliers
outliers = np.abs(df['{selected_col}'] - mean) > threshold

if '{method}' == 'remove':
    df = df[~outliers]
elif '{method}' == 'cap':
    df.loc[df['{selected_col}'] > mean + threshold, '{selected_col}'] = mean + threshold
    df.loc[df['{selected_col}'] < mean - threshold, '{selected_col}'] = mean - threshold
"""
                        st.code(code, language='python')
                        
                        # Show before/after comparison
                        st.subheader("Before and After Comparison")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Before:")
                            fig = px.histogram(
                                st.session_state.data,
                                x=selected_col,
                                nbins=20,
                                title="Before"
                            )
                            fig.update_traces(marker_color=NU_RED)
                            fig.update_layout(template=custom_template, showlegend=False, bargap=0.1)
                            st.plotly_chart(fig)
                        with col2:
                            st.write("After:")
                            fig = px.histogram(
                                st.session_state.cleaned_data,
                                x=selected_col,
                                nbins=20,
                                title="After"
                            )
                            fig.update_traces(marker_color=NU_RED)
                            fig.update_layout(template=custom_template, showlegend=False, bargap=0.1)
                            st.plotly_chart(fig)
                        
                        # Show outlier statistics
                        st.write(f"Number of outliers detected: {outlier_mask.sum()}")
                        st.write(f"Percentage of data marked as outliers: {(outlier_mask.sum() / len(outlier_mask) * 100):.2f}%")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.info("Outlier detection is only available for numeric columns")
    else:
        st.info('Please select a data source to begin')

elif page == '2: Data Exploration':
    st.title('Data Exploration Lab')
    if st.session_state.data is not None:
        # Add option to use original or cleaned data
        data_version = st.radio(
            "Select data version",
            ["Original Data", "Cleaned Data"],
            help="Use cleaned data to see the effects of your data cleaning operations"
        )
        
        # Use appropriate dataset
        current_data = st.session_state.cleaned_data if data_version == "Cleaned Data" else st.session_state.data
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs([
            "Correlation Analysis",
            "Multicollinearity Analysis",
            "Custom Visualizations"
        ])
        
        with tab1:
            st.subheader("Correlation Analysis")
            
            # Correlation method selection
            corr_method = st.selectbox(
                "Select correlation method",
                ["pearson", "spearman", "kendall"],
                help="""
                - Pearson: Linear correlation, assumes normal distribution
                - Spearman: Rank correlation, good for non-linear monotonic relationships
                - Kendall: Rank correlation, more robust to outliers than Spearman
                """
            )
            
            # Calculate and display correlation matrix
            corr_matrix = calculate_correlations(current_data, method=corr_method)
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu',
                aspect='auto',
                title=f'{corr_method.capitalize()} Correlation Matrix'
            )
            fig.update_layout(template=custom_template)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation table
            st.subheader("Correlation Values")
            st.dataframe(
                corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1),
                height=400
            )
        
        with tab2:
            st.subheader("Multicollinearity Analysis")
            st.markdown("""
            Variance Inflation Factor (VIF) helps detect multicollinearity in your data.
            
            Guidelines:
            - VIF = 1: No correlation
            - 1 < VIF < 5: Moderate correlation
            - VIF >= 5: High correlation (potential problem)
            - VIF >= 10: Severe correlation (definite problem)
            """)
            
            # Calculate and display VIF
            try:
                vif_data = calculate_vif(current_data)
                
                # Create bar plot of VIF scores
                fig = px.bar(
                    vif_data,
                    x='Feature',
                    y='VIF',
                    title='Variance Inflation Factors',
                    template=custom_template
                )
                fig.update_traces(marker_color=NU_RED)
                fig.add_hline(y=5, line_dash="dash", line_color="orange",
                            annotation_text="High Correlation Threshold")
                fig.add_hline(y=10, line_dash="dash", line_color="red",
                            annotation_text="Severe Correlation Threshold")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show VIF table
                st.subheader("VIF Values")
                st.dataframe(vif_data)
                
            except Exception as e:
                st.error(f"Could not calculate VIF scores: {str(e)}")
        
        with tab3:
            st.subheader("Custom Visualizations")
            
            # Get visualization options
            viz_options = get_visualization_options(current_data)
            
            # Visualization type selection
            viz_type = st.selectbox(
                "Select visualization type",
                list(viz_options.keys())
            )
            
            # Show visualization description
            st.markdown(viz_options[viz_type]['description'])
            
            # Column selection based on visualization type
            selected_columns = []
            viz_config = viz_options[viz_type]
            min_cols, max_cols = viz_config['required_cols'] if isinstance(viz_config['required_cols'], tuple) else (viz_config['required_cols'], viz_config['required_cols'])
            
            if viz_config['required_cols'] == 'all_numeric':
                numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                selected_columns = st.multiselect(
                    "Select numeric columns",
                    numeric_cols,
                    default=list(numeric_cols)[:min(5, len(numeric_cols))]
                )
            else:
                for i in range(max_cols):
                    if i >= min_cols and len(selected_columns) >= min_cols:
                        if not st.checkbox(f"Add another variable (y-axis {i})", key=f"add_var_{i}"):
                            break
                    
                    col_type = viz_config['col_types'][i]
                    if col_type == 'numeric':
                        cols = current_data.select_dtypes(include=[np.number]).columns
                    elif col_type == 'categorical':
                        cols = current_data.select_dtypes(include=['object', 'category']).columns
                    else:  # 'any'
                        cols = current_data.columns
                    
                    selected_columns.append(
                        st.selectbox(f"Select column {i+1}", cols, key=f"col_{i}")
                    )
            
            # Sorting options
            sort_options = {}
            if viz_config['supports_sort']:
                col1, col2 = st.columns(2)
                with col1:
                    sort_by = st.selectbox(
                        "Sort by",
                        ['None'] + viz_config['sort_options'],
                        help="Select column or metric to sort by"
                    )
                with col2:
                    sort_ascending = st.checkbox("Sort ascending", value=True)
                
                if sort_by != 'None':
                    sort_options = {
                        'sort_by': sort_by,
                        'sort_ascending': sort_ascending
                    }
            
            # Create visualization
            if len(selected_columns) >= min_cols:
                try:
                    fig = create_visualization(
                        current_data,
                        viz_type,
                        selected_columns,
                        **sort_options
                    )
                    fig.update_layout(template=custom_template)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show code for the visualization
                    st.subheader("Code for this Visualization")
                    
                    # Generate code based on visualization type
                    code = "import plotly.express as px\nimport plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\n\n"
                    
                    if viz_type == 'Scatter Plot':
                        if len(selected_columns) == 2:
                            code += f"""# Create scatter plot
fig = px.scatter(
    data_frame=df,
    x='{selected_columns[0]}',
    y='{selected_columns[1]}',
    title='{selected_columns[1]} vs {selected_columns[0]}'
)
fig.update_traces(marker_color='{NU_RED}')\n"""
                        else:
                            code += f"""# Create scatter plot with dual y-axes
fig = make_subplots(specs=[[{{"secondary_y": True}}]])

# Primary y-axis
fig.add_trace(
    go.Scatter(
        x=df['{selected_columns[0]}'],
        y=df['{selected_columns[1]}'],
        name='{selected_columns[1]}',
        mode='markers',
        marker_color='{NU_RED}'
    ),
    secondary_y=False
)

# Secondary y-axis
fig.add_trace(
    go.Scatter(
        x=df['{selected_columns[0]}'],
        y=df['{selected_columns[2]}'],
        name='{selected_columns[2]}',
        mode='markers',
        marker_color='{NU_GRAY}'
    ),
    secondary_y=True
)

fig.update_layout(
    title='Multiple Variables vs {selected_columns[0]}',
    xaxis_title='{selected_columns[0]}',
    yaxis_title='{selected_columns[1]}',
    yaxis2_title='{selected_columns[2]}'
)\n"""

                    elif viz_type == 'Box Plot':
                        code += f"""# Create box plot
fig = px.box(
    data_frame=df,
    x='{selected_columns[0]}',
    y='{selected_columns[1]}',
    title='Distribution of {selected_columns[1]} by {selected_columns[0]}'
)
fig.update_traces(marker_color='{NU_RED}')\n"""

                    elif viz_type == 'Bar Plot':
                        if len(selected_columns) == 1:
                            code += f"""# Create bar plot for single column
value_counts = df['{selected_columns[0]}'].value_counts()
fig = px.bar(
    x=value_counts.index,
    y=value_counts.values,
    title='Counts of {selected_columns[0]}'
)
fig.update_traces(marker_color='{NU_RED}')\n"""
                        elif len(selected_columns) == 2:
                            code += f"""# Create bar plot
fig = px.bar(
    data_frame=df,
    x='{selected_columns[0]}',
    y='{selected_columns[1]}',
    title='{selected_columns[1]} by {selected_columns[0]}'
)
fig.update_traces(marker_color='{NU_RED}')\n"""
                        else:
                            code += f"""# Create bar plot with dual y-axes
fig = make_subplots(specs=[[{{"secondary_y": True}}]])

# Primary y-axis
fig.add_trace(
    go.Bar(
        x=df['{selected_columns[0]}'],
        y=df['{selected_columns[1]}'],
        name='{selected_columns[1]}',
        marker_color='{NU_RED}'
    ),
    secondary_y=False
)

# Secondary y-axis
fig.add_trace(
    go.Bar(
        x=df['{selected_columns[0]}'],
        y=df['{selected_columns[2]}'],
        name='{selected_columns[2]}',
        marker_color='{NU_GRAY}'
    ),
    secondary_y=True
)

fig.update_layout(
    title='Multiple Variables by {selected_columns[0]}',
    xaxis_title='{selected_columns[0]}',
    yaxis_title='{selected_columns[1]}',
    yaxis2_title='{selected_columns[2]}',
    barmode='group'
)\n"""

                    elif viz_type == 'Histogram':
                        code += f"""# Create histogram
fig = px.histogram(
    data_frame=df,
    x='{selected_columns[0]}',
    title='Distribution of {selected_columns[0]}'
)
fig.update_traces(marker_color='{NU_RED}')\n"""

                    elif viz_type == 'Line Plot':
                        if len(selected_columns) == 2:
                            code += f"""# Create line plot
fig = px.line(
    data_frame=df,
    x='{selected_columns[0]}',
    y='{selected_columns[1]}',
    title='{selected_columns[1]} vs {selected_columns[0]}'
)
fig.update_traces(line_color='{NU_RED}')\n"""
                        else:
                            code += f"""# Create line plot with dual y-axes
fig = make_subplots(specs=[[{{"secondary_y": True}}]])

# Primary y-axis
fig.add_trace(
    go.Scatter(
        x=df['{selected_columns[0]}'],
        y=df['{selected_columns[1]}'],
        name='{selected_columns[1]}',
        mode='lines',
        line_color='{NU_RED}'
    ),
    secondary_y=False
)

# Secondary y-axis
fig.add_trace(
    go.Scatter(
        x=df['{selected_columns[0]}'],
        y=df['{selected_columns[2]}'],
        name='{selected_columns[2]}',
        mode='lines',
        line_color='{NU_GRAY}'
    ),
    secondary_y=True
)

fig.update_layout(
    title='Multiple Variables vs {selected_columns[0]}',
    xaxis_title='{selected_columns[0]}',
    yaxis_title='{selected_columns[1]}',
    yaxis2_title='{selected_columns[2]}'
)\n"""

                    elif viz_type == 'Heatmap':
                        cols_str = "', '".join(selected_columns)
                        code += f"""# Calculate correlation matrix
corr_matrix = df[['{cols_str}']].corr()

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmid=0
))

fig.update_layout(title='Correlation Heatmap')\n"""

                    # Add sorting code if applicable
                    if sort_options:
                        sort_col = sort_options['sort_by']
                        ascending = sort_options['sort_ascending']
                        code = f"""# Sort the data
df = df.sort_values('{sort_col}', ascending={str(ascending)})\n\n""" + code

                    # Add template code
                    code += """\n# Apply custom template
custom_template = {
    'layout': {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'color': '#000000'},
        'title': {'font': {'color': '#000000'}},
        'xaxis': {'gridcolor': '#EEEEEE', 'linecolor': '#6A6A6A'},
        'yaxis': {'gridcolor': '#EEEEEE', 'linecolor': '#6A6A6A'}
    }
}
fig.update_layout(template=custom_template)\n"""

                    # Display code
                    st.code(code, language='python')
                    
                    # Add download button for the code
                    st.download_button(
                        label="Download Code",
                        data=code,
                        file_name=f"plotly_{viz_type.lower().replace(' ', '_')}.py",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Could not create visualization: {str(e)}")
            elif len(selected_columns) > 0:
                st.info(f"Please select at least {min_cols} columns for this visualization type")
    else:
        st.info("Please upload a dataset or use the sample dataset to begin exploration.")
