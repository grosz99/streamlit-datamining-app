# Data Mining App

A Streamlit-based data mining application for data cleaning, exploration, and visualization with an interactive storytelling dashboard.

## Features

### 1. Data Cleaning Lab
- **Handle Missing Values** with multiple methods:
  - Mean imputation (for normally distributed data)
  - Median imputation (for skewed data)
  - Mode imputation (for categorical data)
  - K-means clustering (for data with strong relationships)
  - Constant value (for specific business cases)
  - Drop records (for random missing data)
- **Visual Distribution Analysis**
  - View distribution of non-missing values
  - Compare before/after imputation results
  - Interactive histograms and value counts
- **Detailed Method Explanations**
  - Clear descriptions of each imputation method
  - Guidance on when to use each method
  - Impact analysis of chosen method

### 2. Data Exploration Lab
- **Univariate Analysis**
  - Distribution visualization
  - Summary statistics
  - Outlier detection
- **Bivariate Analysis**
  - Relationship exploration
  - Pattern identification
  - Correlation studies
- **Custom Visualization Builder**
  - Multiple plot types:
    - Scatter Plot
    - Line Plot
    - Bar Plot
    - Box Plot
    - Histogram
    - Pie Chart
  - Customization options:
    - Color coding
    - Axis selection
    - Binning controls
  - Interactive plots with Plotly

### 3. Data Storytelling Dashboard
- **Pin and Organize Visualizations**
  - Save important insights
  - Create narrative flow
  - Build comprehensive data stories
- **Interactive Components**
  - Add descriptions to visualizations
  - Remove or rearrange components
  - View full-screen visualizations
- **Collaborative Features**
  - Share insights with team
  - Export visualizations
  - Document findings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/grosz99/streamlit-datamining-app.git
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

1. **Data Input**
   - Use sample dataset or upload your own CSV file
   - Preview data and check basic statistics
   - Select relevant columns for analysis

2. **Data Cleaning**
   - Identify missing values and patterns
   - Choose appropriate imputation methods
   - Validate results with visual comparisons

3. **Data Exploration**
   - Create custom visualizations
   - Analyze distributions and relationships
   - Detect patterns and outliers

4. **Data Storytelling**
   - Pin important visualizations
   - Add context and descriptions
   - Build a narrative flow
   - Share insights with stakeholders

## Requirements
- Python 3.8+
- Streamlit >= 1.24.0
- Pandas >= 1.5.3
- NumPy >= 1.24.3
- Plotly >= 5.14.1
- Scikit-learn >= 1.2.2
- SciPy >= 1.10.1
- Statsmodels >= 0.14.0

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
