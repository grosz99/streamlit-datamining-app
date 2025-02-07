import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def determine_variable_type(series):
    """
    Determine if a variable is suitable for linear or logistic regression
    
    Returns:
    --------
    str: 'continuous', 'binary', or 'categorical'
    """
    if not pd.api.types.is_numeric_dtype(series):
        return 'categorical'
    
    # Check if binary (allowing for 0/1 or -1/1)
    unique_values = series.unique()
    if len(unique_values) == 2 and set(unique_values).issubset({0, 1, -1}):
        return 'binary'
    
    # If more than 10 unique values, consider it continuous
    if len(unique_values) > 10:
        return 'continuous'
    
    return 'categorical'

def recommend_model(target_series):
    """
    Recommend a model based on the target variable type
    """
    var_type = determine_variable_type(target_series)
    
    if var_type == 'continuous':
        return {
            'model_type': 'linear',
            'name': 'Linear Regression',
            'description': 'Recommended for predicting continuous values.',
            'metrics': ['R² Score', 'Mean Squared Error', 'Mean Absolute Error']
        }
    elif var_type == 'binary':
        return {
            'model_type': 'logistic',
            'name': 'Logistic Regression',
            'description': 'Recommended for predicting binary outcomes (0/1).',
            'metrics': ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        }
    else:
        return {
            'model_type': 'none',
            'name': 'No Recommendation',
            'description': 'Target variable must be continuous or binary.',
            'metrics': []
        }

def prepare_data(df, target_col, feature_cols, test_size=0.2, random_state=42):
    """
    Prepare data for modeling
    """
    # Split features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns,
        'scaler': scaler
    }

def train_linear_model(X_train, y_train):
    """
    Train a linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_logistic_model(X_train, y_train):
    """
    Train a logistic regression model
    """
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def calculate_linear_regression_pvalues(X, y, model):
    """Calculate p-values for linear regression coefficients"""
    n = len(y)
    p = len(model.coef_)  # number of parameters
    dof = n - p  # degrees of freedom
    mse = np.sum((y - model.predict(X)) ** 2) / dof
    
    # Calculate variance-covariance matrix
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_covar_matrix = mse * np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))
    
    # Standard errors
    se = np.sqrt(np.diag(var_covar_matrix))
    
    # t-statistics
    params = np.concatenate([[model.intercept_], model.coef_])
    t_stats = params / se
    
    # p-values
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    
    return p_values

def calculate_logistic_regression_pvalues(X, y, model):
    """Calculate p-values for logistic regression coefficients"""
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Get predictions
    y_pred = model.predict_proba(X)[:, 1]
    
    # Calculate variance-covariance matrix
    W = np.diag(y_pred * (1 - y_pred))
    var_covar_matrix = np.linalg.inv(X_with_intercept.T.dot(W).dot(X_with_intercept))
    
    # Standard errors
    se = np.sqrt(np.diag(var_covar_matrix))
    
    # z-statistics
    params = np.concatenate([[model.intercept_[0]], model.coef_[0]])
    z_stats = params / se
    
    # p-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
    
    return p_values

def evaluate_logistic_model(model, X_test, y_test, X_train, y_train, feature_names):
    """
    Evaluate a logistic regression model and return various metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC AUC for binary classification
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except:
        roc_auc = None
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Convert confusion matrix to list for JSON serialization
    cm_list = cm.tolist()
    
    # Calculate coefficients and p-values
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0].tolist(),  # Convert to list for serialization
        'Odds_Ratio': np.exp(model.coef_[0]).tolist()  # Convert to list for serialization
    })
    
    # Calculate p-values using statsmodels
    X_with_intercept = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_with_intercept)
    results = logit_model.fit(disp=0)
    
    # Add p-values to coefficient dataframe
    coef_df['P_Value'] = results.pvalues[1:].tolist()  # Skip intercept, convert to list
    
    return {
        'metrics': {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc if roc_auc is not None else 'N/A'
        },
        'confusion_matrix': cm_list,
        'coefficients': coef_df.to_dict('records')
    }

def evaluate_linear_model(model, X_test, y_test, X_train, y_train, feature_names):
    """
    Evaluate linear regression model
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate coefficients and p-values
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_.tolist()
    })
    
    # Calculate p-values
    p_values = calculate_linear_regression_pvalues(X_train, y_train, model)[1:]  # Skip intercept
    coef_df['P_Value'] = p_values.tolist()
    
    return {
        'metrics': {
            'R² Score': r2,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Error': mae
        },
        'coefficients': coef_df.to_dict('records')
    }

def get_feature_importance(coefficients, model_type='linear'):
    """
    Get feature importance interpretation
    """
    interpretations = []
    
    # Convert to DataFrame if it's a list
    if isinstance(coefficients, list):
        coefficients = pd.DataFrame(coefficients)
    
    for _, row in coefficients.iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        
        if feature == 'Intercept':
            if model_type == 'linear':
                interpretation = f"The model's baseline prediction (intercept) is {coef:.4f}"
            else:  # logistic
                odds = row.get('Odds_Ratio', np.exp(coef))
                interpretation = (
                    f"The baseline log-odds (intercept) is {coef:.4f}, "
                    f"corresponding to baseline odds of {odds:.4f}"
                )
        else:
            if model_type == 'linear':
                interpretation = (
                    f"For each one standard deviation increase in {feature}, "
                    f"the target variable is expected to "
                    f"{'increase' if coef > 0 else 'decrease'} by {abs(coef):.4f} "
                    f"standard deviations, holding other variables constant."
                )
            else:  # logistic
                odds = row.get('Odds_Ratio', np.exp(coef))
                interpretation = (
                    f"For each one standard deviation increase in {feature}, "
                    f"the odds of the target being 1 are multiplied by {odds:.4f} "
                    f"({'increases' if coef > 0 else 'decreases'} by "
                    f"{abs(1 - odds) * 100:.1f}%), holding other variables constant."
                )
        
        interpretations.append({
            'feature': feature,
            'coefficient': coef,
            'interpretation': interpretation
        })
    
    return interpretations
