# Data Mining App

A Streamlit-based data mining application for data cleaning and exploration.

## Features

### 1. Data Cleaning Lab
- Handle missing values using multiple methods:
  - Mean imputation
  - Median imputation
  - K-means clustering with correlation analysis
  - Remove rows with missing values
- Visual comparison of data before and after cleaning
- Detailed descriptions of each cleaning method

### 2. Data Exploration Lab
- **Correlation Analysis**
  - Pearson, Spearman, and Kendall correlation methods
  - Interactive correlation heatmaps
  - Detailed correlation tables

- **Multicollinearity Analysis**
  - Variance Inflation Factor (VIF) calculation
  - Visual representation of collinearity
  - Guidelines for interpretation

- **Custom Visualizations**
  - Multiple plot types:
    - Scatter Plot (with dual y-axes support)
    - Box Plot
    - Bar Plot (with dual y-axes support)
    - Histogram
    - Line Plot (with dual y-axes support)
    - Heatmap
  - Sorting options for each visualization
  - Code display for reproducibility
  - Download option for visualization code

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/streamlit-datamining-app.git
cd streamlit-datamining-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Start by selecting either the sample dataset or uploading your own CSV file
2. Choose between the Data Cleaning Lab or Data Exploration Lab
3. Follow the intuitive interface to clean and analyze your data

### Data Cleaning
- Select a column to clean
- Choose a cleaning method
- View before/after comparisons
- Apply changes to your dataset

### Data Exploration
- Analyze correlations between variables
- Check for multicollinearity
- Create custom visualizations
- Download visualization code for reuse

## Requirements
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Statsmodels

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
