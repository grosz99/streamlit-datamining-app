import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime
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

# Set page configuration
st.set_page_config(
    page_title="Data Mining App",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'pinned_visualizations' not in st.session_state:
    st.session_state.pinned_visualizations = []
if 'component_order' not in st.session_state:
    st.session_state.component_order = []
if 'story_components' not in st.session_state:
    st.session_state.story_components = {}
if 'component_counter' not in st.session_state:
    st.session_state.component_counter = 0

# Title and description
st.title('Data Mining Application')
st.markdown("""
This application helps you explore, clean, and analyze your data using various data mining techniques.
Choose your operation from the sidebar and follow the instructions.
""")

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Select a page:', [
    'Data Cleaning Lab',
    'Data Exploration Lab',
    'Prediction Models',
    'Data Optimization Lab',
    'Story Dashboard'
])

# Data Input section in sidebar
st.sidebar.header('Data Input')
dataset_option = st.sidebar.radio(
    "Select data source",
    ["Use Sample Dataset", "Upload Your Own Dataset"]
)

if dataset_option == "Use Sample Dataset":
    try:
        sample_data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
        if os.path.exists(sample_data_path):
            st.session_state.data = pd.read_csv(sample_data_path)
            st.session_state.cleaned_data = st.session_state.data.copy()
            st.sidebar.success('Sample dataset loaded successfully!')
        else:
            st.sidebar.error('Sample dataset not found. Please upload your own dataset.')
            dataset_option = "Upload Your Own Dataset"
    except Exception as e:
        st.sidebar.error(f'Error loading sample dataset: {str(e)}')
        dataset_option = "Upload Your Own Dataset"

if dataset_option == "Upload Your Own Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.cleaned_data = st.session_state.data.copy()
            st.sidebar.success('File uploaded successfully!')
        except Exception as e:
            st.sidebar.error(f'Error: {str(e)}')

if page == 'Data Cleaning Lab':
    st.title('Data Cleaning Lab')
    
    if st.session_state.data is None:
        st.error("Please upload a dataset first!")
    else:
        # Get current data
        current_data = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        # Tabs for different cleaning operations
        clean_tab1, clean_tab2, clean_tab3, clean_tab4 = st.tabs([
            "Missing Values", "Outliers", "Duplicates", "Data Type Conversion"
        ])
        
        # Tab 1: Missing Values
        with clean_tab1:
            st.subheader("Handle Missing Values")
            
            # Display missing value information
            missing_info = current_data.isnull().sum()
            if missing_info.sum() == 0:
                st.success("No missing values found in the dataset!")
            else:
                st.write("Missing value counts by column:")
                missing_df = pd.DataFrame({
                    'Column': missing_info.index,
                    'Missing Values': missing_info.values,
                    'Percentage': (missing_info.values / len(current_data) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Values'] > 0]
                st.dataframe(missing_df)
                
                # Column selection
                cols_with_missing = missing_df['Column'].tolist()
                if cols_with_missing:
                    selected_col = st.selectbox(
                        "Select column to handle missing values",
                        cols_with_missing
                    )
                    
                    # Show distribution of non-missing values
                    if pd.api.types.is_numeric_dtype(current_data[selected_col]):
                        st.write("### Distribution of Non-Missing Values")
                        fig = px.histogram(current_data[~current_data[selected_col].isnull()], 
                                          x=selected_col, 
                                          title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig)
                    else:
                        st.write("### Value Counts of Non-Missing Values")
                        st.write(current_data[selected_col].value_counts())

                    # Method selection with explanations
                    st.write("### Select Imputation Method")
                    method_descriptions = {
                        "mean": "**Mean Imputation**: Replaces missing values with the average of non-missing values. Best for normally distributed numerical data with no significant outliers.",
                        "median": "**Median Imputation**: Replaces missing values with the median. Better than mean when data is skewed or contains outliers.",
                        "mode": "**Mode Imputation**: Replaces missing values with the most frequent value. Suitable for categorical data or discrete numerical values.",
                        "constant": "**Constant Value**: Replaces missing values with a specified constant. Useful when missing values have a specific meaning or when you want to flag them.",
                        "kmeans": "**K-Means Imputation**: Uses K-means clustering to estimate missing values. Good for data with strong relationships between features but computationally intensive.",
                        "drop": "**Drop Records**: Removes rows with missing values. Only use if missing data is random and you can afford to lose observations."
                    }
                    
                    method = st.selectbox("Select imputation method", list(method_descriptions.keys()))
                    
                    # Display method description
                    st.markdown(method_descriptions[method])
                    
                    # Method-specific parameters and processing
                    new_data = current_data.copy()
                    
                    if method == "constant":
                        constant_value = st.text_input("Enter constant value:")
                        if constant_value and st.button("Process Missing Values"):
                            try:
                                if pd.api.types.is_numeric_dtype(new_data[selected_col]):
                                    constant_value = float(constant_value)
                                new_data[selected_col].fillna(constant_value, inplace=True)
                                st.session_state.cleaned_data = new_data
                                st.success(f"Missing values in {selected_col} have been replaced with {constant_value}")
                            except ValueError:
                                st.error("Please enter a valid value")
                    
                    elif method == "kmeans":
                        n_clusters = st.slider("Select number of clusters:", 2, 10, 3)
                        if st.button("Process Missing Values"):
                            try:
                                # Get numeric columns for clustering
                                numeric_cols = new_data.select_dtypes(include=[np.number]).columns
                                X = new_data[numeric_cols].copy()
                                
                                # Temporarily fill missing values with mean for scaling
                                X = X.fillna(X.mean())
                                
                                # Scale the data
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Apply K-means
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                                clusters = kmeans.fit_predict(X_scaled)
                                
                                # For each cluster, fill missing values with cluster mean
                                for cluster in range(n_clusters):
                                    cluster_mean = new_data[selected_col][clusters == cluster].mean()
                                    mask = (clusters == cluster) & (new_data[selected_col].isna())
                                    new_data.loc[mask, selected_col] = cluster_mean
                                
                                st.session_state.cleaned_data = new_data
                                st.success(f"Missing values in {selected_col} have been imputed using K-means clustering")
                            except Exception as e:
                                st.error(f"Error during K-means imputation: {str(e)}")
                    
                    elif method == "drop":
                        if st.button("Process Missing Values"):
                            new_data = new_data.dropna(subset=[selected_col])
                            st.session_state.cleaned_data = new_data
                            st.success(f"Rows with missing values in {selected_col} have been dropped")
                    
                    else:  # mean, median, mode
                        if st.button("Process Missing Values"):
                            if method == "mean" and pd.api.types.is_numeric_dtype(new_data[selected_col]):
                                new_data[selected_col].fillna(new_data[selected_col].mean(), inplace=True)
                            elif method == "median" and pd.api.types.is_numeric_dtype(new_data[selected_col]):
                                new_data[selected_col].fillna(new_data[selected_col].median(), inplace=True)
                            elif method == "mode":
                                new_data[selected_col].fillna(new_data[selected_col].mode()[0], inplace=True)
                            else:
                                st.error(f"{method} imputation is only applicable to numeric columns")
                                st.stop()
                            
                            st.session_state.cleaned_data = new_data
                            st.success(f"Missing values in {selected_col} have been imputed using {method}")
                    
                    # Show before/after comparison
                    if 'cleaned_data' in st.session_state:
                        st.write("### Before vs After Imputation")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Before:")
                            if pd.api.types.is_numeric_dtype(st.session_state.data[selected_col]):
                                fig1 = px.histogram(st.session_state.data[selected_col], title="Original Distribution")
                                st.plotly_chart(fig1)
                            else:
                                st.write(st.session_state.data[selected_col].value_counts())
                                
                        with col2:
                            st.write("After:")
                            if pd.api.types.is_numeric_dtype(st.session_state.cleaned_data[selected_col]):
                                fig2 = px.histogram(st.session_state.cleaned_data[selected_col], title="Imputed Distribution")
                                st.plotly_chart(fig2)
                            else:
                                st.write(st.session_state.cleaned_data[selected_col].value_counts())
        
        with clean_tab2:
            st.header("Outlier Detection and Treatment")
            
            # Select column for outlier detection
            numeric_cols = st.session_state.cleaned_data.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox(
                "Select column for outlier detection:",
                numeric_cols,
                key='outlier_col'
            )
            
            if selected_col:
                # Calculate statistics
                data = st.session_state.cleaned_data[selected_col]
                mean = data.mean()
                std = data.std()
                
                # Outlier threshold control
                st.subheader("Set Outlier Threshold")
                threshold = st.slider(
                    "Standard deviation threshold:",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                    help="Data points beyond this many standard deviations from the mean will be considered outliers"
                )
                
                # Calculate bounds
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                # Identify outliers
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                # Display statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Statistics")
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Std Dev', 'Lower Bound', 'Upper Bound'],
                        'Value': [mean, std, lower_bound, upper_bound]
                    })
                    st.dataframe(stats_df.style.format({'Value': '{:.2f}'}))
                
                with col2:
                    st.subheader("Outlier Summary")
                    st.write(f"Total outliers found: {len(outliers)}")
                    st.write(f"Percentage of data: {(len(outliers)/len(data)*100):.2f}%")
                
                # Visualization
                fig = px.histogram(
                    data,
                    title=f"Distribution of {selected_col} with Outlier Bounds",
                    labels={'value': selected_col, 'count': 'Frequency'}
                )
                fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", name="Lower Bound")
                fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", name="Upper Bound")
                st.plotly_chart(fig)
                
                # Outlier treatment options
                if len(outliers) > 0:
                    st.subheader("Outlier Treatment")
                    treatment = st.radio(
                        "Select treatment method:",
                        ["None", "Remove", "Cap", "Mean", "Median"],
                        help="""
                        - Remove: Delete rows with outliers
                        - Cap: Cap values at the bounds
                        - Mean: Replace outliers with mean
                        - Median: Replace outliers with median
                        """
                    )
                    
                    if treatment != "None" and st.button("Apply Treatment"):
                        temp_data = st.session_state.cleaned_data.copy()
                        
                        if treatment == "Remove":
                            temp_data = temp_data[~temp_data[selected_col].isin(outliers)]
                        elif treatment == "Cap":
                            temp_data.loc[temp_data[selected_col] < lower_bound, selected_col] = lower_bound
                            temp_data.loc[temp_data[selected_col] > upper_bound, selected_col] = upper_bound
                        elif treatment == "Mean":
                            temp_data.loc[temp_data[selected_col].isin(outliers), selected_col] = mean
                        elif treatment == "Median":
                            temp_data.loc[temp_data[selected_col].isin(outliers), selected_col] = data.median()
                        
                        st.session_state.cleaned_data = temp_data
                        st.success(f"Applied {treatment} treatment to outliers in {selected_col}")
                        st.experimental_rerun()
        
        with clean_tab3:
            st.subheader("Handle Duplicates")
            
            # Duplicate detection settings
            duplicate_subset = st.multiselect(
                "Select columns to check for duplicates",
                st.session_state.data.columns.tolist(),  
                default=st.session_state.data.columns.tolist()  
            )
            
            # Duplicate detection
            duplicates = st.session_state.data.duplicated(subset=duplicate_subset, keep=False)
            
            # Display duplicate statistics
            st.write(f"Total duplicates found: {duplicates.sum()}")
            st.write(f"Percentage of data: {(duplicates.sum()/len(duplicates)*100):.2f}%")
            
            # Duplicate treatment options
            if duplicates.sum() > 0:
                st.subheader("Duplicate Treatment")
                treatment = st.radio(
                    "Select treatment method:",
                    ["None", "Remove"],
                    help="""
                    - Remove: Delete duplicate rows
                    """
                )
                
                if treatment != "None" and st.button("Apply Treatment"):
                    if treatment == "Remove":
                        st.session_state.cleaned_data = st.session_state.cleaned_data.drop_duplicates(subset=duplicate_subset)
                    st.success(f"Applied {treatment} treatment to duplicates")
                    st.experimental_rerun()
        
        with clean_tab4:
            st.subheader("Data Type Conversion")
            
            # Column selection
            selected_col = st.selectbox('Choose column', st.session_state.data.columns)
            
            # Data type conversion options
            conversion_options = {
                'int64': 'Integer',
                'float64': 'Float',
                'object': 'String',
                'category': 'Category'
            }
            current_type = st.session_state.data[selected_col].dtype
            available_types = [t for t in conversion_options if t != current_type.name]
            
            if available_types:
                new_type = st.selectbox(
                    "Select new data type",
                    [conversion_options[t] for t in available_types]
                )
                
                # Apply conversion
                if st.button("Apply Conversion"):
                    try:
                        st.session_state.cleaned_data[selected_col] = st.session_state.cleaned_data[selected_col].astype(available_types[[conversion_options[t] for t in available_types].index(new_type)])
                        st.success(f"Converted {selected_col} to {new_type}")
                    except ValueError as e:
                        st.error(f"Conversion failed: {str(e)}")
            else:
                st.info("No other data types available for conversion")

elif page == 'Data Exploration Lab':
    st.title('Data Exploration Lab')
    
    if st.session_state.data is None:
        st.error("Please upload a dataset first!")
    else:
        # Get current data
        current_data = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Correlation Analysis",
            "Multicollinearity Analysis",
            "Visualization Builder"
        ])
        
        # Tab 1: Correlation Analysis
        with tab1:
            st.header("Correlation Analysis")
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
            else:
                # Calculate correlation matrix
                corr_matrix = current_data[numeric_cols].corr()
                
                # Display correlation heatmap
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Heatmap",
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Find high correlations
                high_corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr = abs(corr_matrix.iloc[i, j])
                        if corr > 0.7:  # Threshold for high correlation
                            high_corr_pairs.append({
                                'Feature 1': numeric_cols[i],
                                'Feature 2': numeric_cols[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                
                if high_corr_pairs:
                    st.subheader("High Correlations (|r| > 0.7)")
                    high_corr_df = pd.DataFrame(high_corr_pairs)
                    st.dataframe(
                        high_corr_df.style.format({'Correlation': '{:.3f}'})
                        .background_gradient(subset=['Correlation'], cmap='RdBu')
                    )
                else:
                    st.success("No high correlations found between features (|r| ≤ 0.7)")
        
        # Tab 2: Multicollinearity Analysis
        with tab2:
            st.header("Multicollinearity Analysis")
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for multicollinearity analysis.")
            else:
                st.subheader("Variance Inflation Factor (VIF) Analysis")
                st.write("""
                VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity.
                - VIF = 1: No correlation
                - 1 < VIF < 5: Moderate correlation
                - VIF ≥ 5: High correlation (potential problem)
                - VIF ≥ 10: Serious multicollinearity problem
                """)
                
                # Calculate VIF scores
                vif_data = pd.DataFrame()
                vif_data["Feature"] = numeric_cols
                
                # Clean data for VIF calculation
                X = current_data[numeric_cols].copy()
                
                # Check for missing values and infinities
                has_missing = X.isnull().any().any() or np.isinf(X.values).any()
                
                if has_missing:
                    st.warning("⚠️ Dataset contains missing values or infinite numbers. These will be handled automatically.")
                    # Replace infinities with NaN
                    X = X.replace([np.inf, -np.inf], np.nan)
                    # Fill missing values with median
                    X = X.fillna(X.median())
                
                try:
                    # Calculate VIF for each feature
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    vif_data = vif_data.sort_values('VIF', ascending=False)
                    
                    # Display VIF scores with color coding
                    def color_vif(val):
                        if val >= 10:
                            return 'background-color: #ff9999'  # Red
                        elif val >= 5:
                            return 'background-color: #ffeb99'  # Yellow
                        return ''
                    
                    st.dataframe(
                        vif_data.style.format({'VIF': '{:.2f}'})
                        .applymap(color_vif, subset=['VIF'])
                    )
                    
                    # Identify problematic features
                    problem_features = vif_data[vif_data['VIF'] >= 5]['Feature'].tolist()
                    if problem_features:
                        st.warning("⚠️ The following features show signs of multicollinearity:")
                        for feat in problem_features:
                            st.write(f"- {feat} (VIF: {vif_data[vif_data['Feature'] == feat]['VIF'].values[0]:.2f})")
                        
                        st.subheader("Correlation Analysis for Problematic Features")
                        problem_corr = current_data[problem_features].corr()
                        
                        fig = px.imshow(
                            problem_corr,
                            title="Correlation Heatmap for High VIF Features",
                            labels=dict(color="Correlation"),
                            color_continuous_scale="RdBu"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Suggestions for Handling Multicollinearity")
                        st.markdown("""
                        1. **Feature Selection**:
                           - Remove one of each highly correlated pair
                           - Keep features that have stronger correlation with target variable
                        
                        2. **Feature Engineering**:
                           - Create interaction terms or ratios
                           - Combine correlated features into a single feature
                        
                        3. **Dimensionality Reduction**:
                           ```python
                           from sklearn.decomposition import PCA
                           
                           # Apply PCA to correlated features
                           pca = PCA(n_components=0.95)  # Keep 95% of variance
                           transformed_features = pca.fit_transform(df[problem_features])
                           ```
                        
                        4. **Regularization**:
                           - Use Ridge (L2) or Lasso (L1) regression
                           - These methods can handle multicollinearity
                           ```python
                           from sklearn.linear_model import Ridge, Lasso
                           
                           # Ridge Regression
                           ridge = Ridge(alpha=1.0)
                           
                           # Lasso Regression
                           lasso = Lasso(alpha=1.0)
                           ```
                        """)
                    else:
                        st.success("No significant multicollinearity detected (all VIF scores < 5)")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
        # Tab 3: Visualization Builder
        with tab3:
            st.header("Create Custom Visualization")
            
            # Visualization type selection
            viz_type = st.selectbox(
                "Select visualization type",
                ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Violin Plot"]
            )
            
            # Column selection based on visualization type
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis", current_data.columns.tolist(), key="x_axis")
                if viz_type in ["Scatter Plot", "Line Chart", "Box Plot", "Violin Plot"]:
                    y_col = st.selectbox("Y-axis", current_data.columns.tolist(), key="y_axis")
                else:
                    y_col = None
            
            with col2:
                color_col = st.selectbox(
                    "Color by",
                    ["None"] + current_data.columns.tolist(),
                    key="color"
                )
                
                if viz_type in ["Bar Chart", "Box Plot"]:
                    agg_func = st.selectbox(
                        "Aggregation function",
                        ["mean", "median", "sum", "count"],
                        key="agg_func"
                    )
                else:
                    agg_func = None
            
            try:
                fig = None
                
                if viz_type == "Scatter Plot":
                    fig = px.scatter(
                        current_data,
                        x=x_col,
                        y=y_col,
                        color=None if color_col == "None" else color_col,
                        title=f"{y_col} vs {x_col}"
                    )
                
                elif viz_type == "Line Chart":
                    fig = px.line(
                        current_data,
                        x=x_col,
                        y=y_col,
                        color=None if color_col == "None" else color_col,
                        title=f"{y_col} over {x_col}"
                    )
                
                elif viz_type == "Bar Chart":
                    if agg_func == "count":
                        fig = px.histogram(
                            current_data,
                            x=x_col,
                            color=None if color_col == "None" else color_col,
                            title=f"Count of {x_col}"
                        )
                    else:
                        grouped_data = current_data.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        fig = px.bar(
                            grouped_data,
                            x=x_col,
                            y=y_col,
                            title=f"{agg_func.capitalize()} of {y_col} by {x_col}"
                        )
                
                elif viz_type == "Histogram":
                    fig = px.histogram(
                        current_data,
                        x=x_col,
                        color=None if color_col == "None" else color_col,
                        title=f"Distribution of {x_col}"
                    )
                
                elif viz_type == "Box Plot":
                    fig = px.box(
                        current_data,
                        x=x_col,
                        y=y_col,
                        color=None if color_col == "None" else color_col,
                        title=f"Box Plot of {y_col} by {x_col}"
                    )
                
                elif viz_type == "Violin Plot":
                    fig = px.violin(
                        current_data,
                        x=x_col,
                        y=y_col,
                        color=None if color_col == "None" else color_col,
                        title=f"Violin Plot of {y_col} by {x_col}"
                    )
                
                if fig:
                    # Update layout
                    fig.update_layout(
                        template="plotly_white",
                        height=600
                    )
                    
                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Pin visualization option
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        viz_description = st.text_area(
                            "Add a description for this visualization",
                            placeholder="Enter a description to help remember why this visualization is important..."
                        )
                    
                    with col2:
                        if st.button("📌 Pin to Story Dashboard"):
                            # Generate unique component ID
                            component_id = f"viz_{st.session_state.component_counter}"
                            st.session_state.component_counter += 1
                            
                            # Create component data
                            component_data = {
                                'title': fig.layout.title.text,
                                'figure': fig,
                                'description': viz_description,
                                'type': 'visualization'
                            }
                            
                            # Add to story components
                            st.session_state.story_components[component_id] = component_data
                            st.session_state.component_order.append(component_id)
                            
                            st.success("Visualization pinned to Story Dashboard! 📌")
            
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

elif page == "Prediction Models":
    st.title("Prediction Models Lab")
    st.write("Build and evaluate Linear or Logistic Regression models.")
    
    if 'data' not in st.session_state:
        st.error("Please upload a dataset in the Data Cleaning Lab first.")
    else:
        # Get current data
        current_data = (st.session_state['cleaned_data'] 
                       if 'cleaned_data' in st.session_state 
                       else st.session_state['data'])
        
        # Add Data Profile section
        with st.expander("View Data Profile", expanded=True):
            profile = current_data.describe()
            st.dataframe(profile)
        
        st.divider()
        
        # Select target variable
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        target_col = st.selectbox(
            "Select the variable you want to predict (target variable):",
            numeric_cols
        )
        
        if target_col:
            # Get model recommendation
            recommendation = "Linear Regression"
            
            st.subheader("Model Recommendation")
            st.write(recommendation)
            
            if recommendation != 'none':
                # Select features
                feature_cols = st.multiselect(
                    "Select the variables to use as features:",
                    [col for col in current_data.columns if col != target_col],
                    help="Choose the variables that you think will help predict your target variable."
                )
                
                if feature_cols:
                    # Train model button
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            try:
                                # Prepare data
                                data = {
                                    'X': current_data[feature_cols],
                                    'y': current_data[target_col]
                                }
                                
                                # Train and evaluate model
                                model = LinearRegression()
                                model.fit(data['X'], data['y'])
                                y_pred = model.predict(data['X'])
                                
                                # Display results
                                st.subheader("Model Performance")
                                metrics = {
                                    'R-squared': r2_score(data['y'], y_pred),
                                    'Mean Squared Error': mean_squared_error(data['y'], y_pred)
                                }
                                st.dataframe(pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']))
                                
                                # Display coefficients in a formatted table
                                st.write("\nModel Coefficients:")
                                coef_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Coefficient': model.coef_
                                })
                                st.dataframe(coef_df)
                            
                            except Exception as e:
                                st.error(f"An error occurred while training the model: {str(e)}")

elif page == "Data Optimization Lab":
    st.title("Data Optimization Lab")
    
    if st.session_state.data is None:
        st.error("Please upload a dataset first!")
    else:
        # Get current data
        current_data = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        # Create tabs
        tab1, tab2 = st.tabs([
            "Dimensionality Reduction",
            "Feature Selection"
        ])
        
        # Tab 1: Dimensionality Reduction
        with tab1:
            st.header("Dimensionality Reduction with PCA")
            
            # Get numeric columns
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for PCA analysis.")
            else:
                # Select features for PCA
                selected_features = st.multiselect(
                    "Select features for PCA",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if len(selected_features) < 2:
                    st.warning("Please select at least 2 features for PCA analysis.")
                else:
                    try:
                        # Prepare data for PCA
                        X = current_data[selected_features].copy()
                        
                        # Handle missing values if any
                        if X.isnull().any().any():
                            st.warning("⚠️ Dataset contains missing values. They will be filled with median values.")
                            X = X.fillna(X.median())
                        
                        # Scale the data
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Perform PCA
                        n_components = min(len(selected_features), 10)
                        pca = PCA(n_components=n_components)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # Calculate explained variance ratio
                        explained_variance_ratio = pca.explained_variance_ratio_
                        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
                        
                        # Plot explained variance ratio
                        fig = make_subplots(rows=1, cols=2,
                                          subplot_titles=('Explained Variance Ratio',
                                                        'Cumulative Explained Variance'))
                        
                        # Individual explained variance
                        fig.add_trace(
                            go.Bar(x=[f'PC{i+1}' for i in range(n_components)],
                                  y=explained_variance_ratio,
                                  name='Individual'),
                            row=1, col=1
                        )
                        
                        # Cumulative explained variance
                        fig.add_trace(
                            go.Scatter(x=[f'PC{i+1}' for i in range(n_components)],
                                     y=cumulative_variance_ratio,
                                     name='Cumulative',
                                     mode='lines+markers'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=False,
                                        title_text="PCA Analysis Results")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show explained variance table
                        variance_df = pd.DataFrame({
                            'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                            'Explained Variance Ratio': explained_variance_ratio,
                            'Cumulative Variance Ratio': cumulative_variance_ratio
                        })
                        st.write("Explained Variance Details:")
                        st.dataframe(
                            variance_df.style.format({
                                'Explained Variance Ratio': '{:.3f}',
                                'Cumulative Variance Ratio': '{:.3f}'
                            })
                        )
                        
                        # Feature importance in principal components
                        loadings = pd.DataFrame(
                            pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=selected_features
                        )
                        
                        st.subheader("Feature Importance in Principal Components")
                        fig = px.imshow(loadings,
                                      labels=dict(x="Principal Component", y="Feature"),
                                      title="PCA Loadings Heatmap",
                                      color_continuous_scale="RdBu")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Option to save transformed data
                        n_components_to_keep = st.slider(
                            "Select number of components to keep",
                            min_value=2,
                            max_value=n_components,
                            value=min(3, n_components),
                            help="Choose how many principal components to keep in the transformed data"
                        )
                        
                        if st.button("Apply PCA Transformation"):
                            # Create new dataframe with PCA results
                            pca_cols = [f'PC{i+1}' for i in range(n_components_to_keep)]
                            pca_df = pd.DataFrame(
                                X_pca[:, :n_components_to_keep],
                                columns=pca_cols,
                                index=current_data.index
                            )
                            
                            # Add non-numeric columns back
                            non_numeric_cols = [col for col in current_data.columns if col not in selected_features]
                            if non_numeric_cols:
                                pca_df = pd.concat([pca_df, current_data[non_numeric_cols]], axis=1)
                            
                            # Update session state
                            st.session_state.cleaned_data = pca_df
                            st.success(f"Data transformed! Reduced {len(selected_features)} features to {n_components_to_keep} principal components.")
                            
                            # Show preview of transformed data
                            st.write("Preview of transformed data:")
                            st.dataframe(pca_df.head())
                            
                    except Exception as e:
                        st.error(f"An error occurred during PCA analysis: {str(e)}")
        
        # Tab 2: Feature Selection
        with tab2:
            st.header("Feature Selection")
            st.info("Feature selection tools will be available in the next update!")

elif page == "Story Dashboard":
    st.header("Data Storytelling Dashboard")
    
    # Initialize storytelling components if not exists
    if 'story_components' not in st.session_state:
        st.session_state.story_components = {}
    
    # Display components in order
    if st.session_state.component_order:
        for component_id in st.session_state.component_order:
            if component_id in st.session_state.story_components:
                component = st.session_state.story_components[component_id]
                
                # Create a container for the component
                with st.container():
                    st.markdown(f"### {component['title']}")
                    
                    # Display visualization if it exists
                    if 'figure' in component:
                        st.plotly_chart(component['figure'])
                    
                    # Display description if it exists
                    if 'description' in component:
                        st.markdown(component['description'])
                    
                    # Add remove button
                    if st.button(f"Remove from Dashboard 🗑️", key=f"remove_{component_id}"):
                        st.session_state.component_order.remove(component_id)
                        del st.session_state.story_components[component_id]
                        st.experimental_rerun()
                    
                    st.markdown("---")  # Add separator between components
    else:
        st.info("Your data story will appear here. Pin visualizations and insights from the Data Exploration Lab to build your story!")
