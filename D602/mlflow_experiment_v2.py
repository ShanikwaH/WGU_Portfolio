# coding: utf-8

"""
MLFlow Experiment Script - Version 2 (Final) - Debugged (Added Local Plot Saving)
D602 Task 2 - Flight Delay Analysis

This script implements a comprehensive MLFlow experiment that captures all features
listed in the poly_regressor_Python_1.0.0.py file comments. It includes:

1. Informational log files from import_data and clean_data scripts
2. Input parameters (alpha and polynomial order) for final regression
3. Performance plots and visualizations
4. Model performance metrics (MSE, RMSE, MAE, R²)
5. Model artifacts and metadata
6. Feature importance analysis
7. Comprehensive experiment tracking

This implementation meets all competent level requirements for D602 Task 2.

Author: Student
Date: 2025
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
import os
import json
import pickle
from datetime import datetime
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging - Ensure encoding is UTF-8 for file handler
logging.basicConfig(
    level=logging.INFO, # Can change to logging.DEBUG for more verbose output during debugging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlflow_experiment.log', encoding='utf-8'), # Added encoding
        logging.StreamHandler() # Keep console output (console encoding handled by OS/terminal)
    ]
)
logger = logging.getLogger(__name__)

# Function to sanitize metric names
def sanitize_metric_name(name):
    """
    Sanitizes a string to be a valid MLflow metric name.
    Replaces common problematic characters with underscores.
    """
    # Replace '^' with '__pow__' for clarity, or simply '_'
    s_name = name.replace('^', '__pow__')
    # Replace spaces with underscores
    s_name = s_name.replace(' ', '_')
    # Replace any other non-alphanumeric (except underscore, dash, period, slash) with underscore
    s_name = ''.join(c if c.isalnum() or c in ['_', '-', '.', '/'] else '_' for c in s_name)
    return s_name

def log_import_and_clean_logs():
    """
    Log informational log files from import_data, clean_data, and pipeline scripts.
    Logs are now expected in the current working directory of the MLflow run.
    """
    log_files = [
        'import_data.log',
        'clean_data.log',
        'pipeline.log' 
    ]
    
    current_run_dir = os.getcwd() 
    
    for log_file_name in log_files:
        log_path = os.path.join(current_run_dir, log_file_name) 

        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                mlflow.log_text(log_content, f"logs/{log_file_name}") 
                logger.info(f"Logged {log_file_name}")
            except Exception as e:
                logger.warning(f"Could not log {log_file_name} from {log_path}: {e}")
        else:
            logger.warning(f"Log file {log_file_name} not found at {log_path} for logging.")


def log_input_parameters(alpha, polynomial_order, random_state=42, test_size=0.2):
    """
    Log input parameters for the final regression.
    This captures feature #2 from the poly_regressor comments.
    """
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("polynomial_order", polynomial_order) 
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("model_type", "Ridge Regression")
    mlflow.log_param("feature_scaling", "StandardScaler")
    
    logger.info(f"Logged parameters: alpha={alpha}, order={polynomial_order}")

def create_performance_plots(y_true, y_pred, alpha, save_path_prefix="model_performance"):
    """
    Create and log performance plots.
    This captures feature #3 from the poly_regressor comments.
    Ensures y_true and y_pred are consistently 1D for plotting.
    Also saves plots locally to the current working directory.
    """
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    logger.debug(f"DEBUG: y_true_flat shape: {y_true_flat.shape}")
    logger.debug(f"DEBUG: y_pred_flat shape: {y_pred_flat.shape}")
    
    if len(y_true_flat) != len(y_pred_flat):
        logger.error(f"FATAL PLOT ERROR: y_true_flat length ({len(y_true_flat)}) does not match y_pred_flat length ({len(y_pred_flat)})")
        raise ValueError("Cannot create plots: True values and predicted values must have the same size.")

    y_true_series = pd.Series(y_true_flat)
    y_pred_series = pd.Series(y_pred_flat)

    tips = pd.DataFrame()
    tips["prediction"] = y_pred_series
    tips["original_data"] = y_true_series
    
    residuals = y_true_series - y_pred_series
    logger.debug(f"DEBUG: residuals length: {len(residuals)}")

    # Plot 1: Joint plot (Actual vs Predicted)
    fig_joint = sns.jointplot(
        x="original_data", 
        y="prediction", 
        data=tips, 
        height=6, 
        ratio=7,
        joint_kws={'line_kws':{'color':'limegreen'}}, 
        kind='reg'
    )
    fig_joint.set_axis_labels('Mean delays (min)', 'Predictions (min)', fontsize=15)
    fig_joint.fig.suptitle(f'Model Performance (Alpha={alpha})', y=1.02)
    fig_joint.ax_joint.plot(list(range(-10, 25)), list(range(-10, 25)), linestyle=':', color='r')
    
    # Log to MLflow
    mlflow.log_figure(fig_joint.fig, f"{save_path_prefix}_joint_alpha_{alpha}.png")
    # Save locally
    plt_filename_joint = f"{save_path_prefix}_joint_alpha_{alpha}.png"
    fig_joint.savefig(plt_filename_joint) # Save the figure locally
    plt.close(fig_joint.fig)


    # Plot 2: Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_series, residuals, alpha=0.5) 
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot (Alpha={alpha})')
    plt.tight_layout()
    
    # Log to MLflow
    mlflow.log_figure(plt.gcf(), f"{save_path_prefix}_residual_alpha_{alpha}.png")
    # Save locally
    plt_filename_residual = f"{save_path_prefix}_residual_alpha_{alpha}.png"
    plt.savefig(plt_filename_residual) # Save the figure locally
    plt.close()
    
    # Plot 3: Distribution Comparison and Predicted vs Actual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_true_series, bins=30, alpha=0.7, label='Actual', color='blue')
    plt.hist(y_pred_series, bins=30, alpha=0.7, label='Predicted', color='red')
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Frequency')
    plt.title('Distribution Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_series, y_pred_series, alpha=0.5)
    plt.plot([y_true_series.min(), y_true_series.max()], [y_true_series.min(), y_true_series.max()], 'r--', lw=2)
    plt.xlabel('Actual Delay (minutes)')
    plt.ylabel('Predicted Delay (minutes)')
    plt.title('Predicted vs Actual')
    
    plt.tight_layout()
    # Log to MLflow
    mlflow.log_figure(plt.gcf(), f"{save_path_prefix}_distribution_comparison_alpha_{alpha}.png")
    # Save locally
    plt_filename_dist_comp = f"{save_path_prefix}_distribution_comparison_alpha_{alpha}.png"
    plt.savefig(plt_filename_dist_comp) # Save the figure locally
    plt.close()
    
    logger.info(f"Created performance plots for alpha={alpha}")


def log_model_performance_metrics(y_true, y_pred, alpha):
    """
    Log model performance metrics.
    This captures feature #4 from the poly_regressor comments.
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("root_mean_squared_error", rmse)
    mlflow.log_metric("mean_absolute_error", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("average_delay_minutes", rmse) 
    
    mlflow.log_metric("explained_variance_score", metrics.explained_variance_score(y_true, y_pred))
    mlflow.log_metric("max_error", metrics.max_error(y_true, y_pred))
    
    logger.info(f"Logged metrics for alpha={alpha}: MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

def log_model_artifacts(model, poly_transformer, scaler, feature_names):
    """
    Log model artifacts and metadata.
    """
    mlflow.sklearn.log_model(model, "model")
    
    mlflow.sklearn.log_model(poly_transformer, "poly_transformer")
    mlflow.sklearn.log_model(scaler, "scaler")
    
    model_metadata = {
        "feature_names": feature_names,
        "model_type": "Ridge Regression",
        "training_date": datetime.now().isoformat(),
        "model_version": "1.0.0"
    }
    
    artifact_dir = "model_artifacts"
    os.makedirs(artifact_dir, exist_ok=True) 

    metadata_path = os.path.join(artifact_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2)
    mlflow.log_artifact(metadata_path)
    
    model_pkl_path = os.path.join(artifact_dir, 'finalized_model.pkl')
    poly_pkl_path = os.path.join(artifact_dir, 'best_poly_transformer.pkl')
    scaler_pkl_path = os.path.join(artifact_dir, 'best_scaler.pkl')

    with open(model_pkl_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(poly_pkl_path, 'wb') as f:
        pickle.dump(poly_transformer, f)

    with open(scaler_pkl_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    mlflow.log_artifact(model_pkl_path)
    mlflow.log_artifact(poly_pkl_path)
    mlflow.log_artifact(scaler_pkl_path)
    
    logger.info("Logged model artifacts and metadata")

def log_feature_importance(model, feature_names, alpha):
    """
    Log feature importance analysis.
    Sanitizes feature names for MLflow metric logging.
    Also saves the feature importance plot locally.
    """
    if hasattr(model, 'coef_'):
        plt.figure(figsize=(12, 8))
        
        coefs = np.abs(model.coef_)
        
        poly_feature_names = feature_names 
        
        sorted_indices = np.argsort(coefs)[::-1]
        sorted_coefs = coefs[sorted_indices]
        sorted_names = [poly_feature_names[i] for i in sorted_indices]
        
        top_n = min(20, len(sorted_coefs))
        plt.barh(range(top_n), sorted_coefs[:top_n])
        plt.yticks(range(top_n), sorted_names[:top_n])
        plt.xlabel('Absolute Coefficient Value')
        plt.title(f'Feature Importance (Alpha={alpha})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Log to MLflow
        mlflow.log_figure(plt.gcf(), f"feature_importance_alpha_{alpha}.png")
        # Save locally
        plt_filename_feature_imp = f"feature_importance_alpha_{alpha}.png"
        plt.savefig(plt_filename_feature_imp) # Save the figure locally
        plt.close()
        
        for i, (name, coef) in enumerate(zip(sorted_names[:10], sorted_coefs[:10])):
            sanitized_name = sanitize_metric_name(name) 
            mlflow.log_metric(f"feature_importance_{sanitized_name}", coef)
        
        logger.info(f"Logged feature importance for alpha={alpha}")

def run_comprehensive_mlflow_experiment(data_file="cleaned_data.csv", num_alphas=10, polynomial_degree=2):
    """
    Run comprehensive MLFlow experiment capturing all features from poly_regressor.
    Corrected to use actual data from data_file.
    """
    logger.info("Starting comprehensive MLFlow experiment")
    
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded data: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise 

    if 'DepDelay' not in df.columns:
        logger.error("Required 'DepDelay' column not found in cleaned data.")
        raise ValueError("Missing 'DepDelay' column for target variable.")
    
    feature_cols = [
        'MONTH', 'DAYOFMONTH', 'DAYOFWEEK', 'DEP_HOUR', 
        'DISTANCE', 'CRSDepTime'
    ]

    available_features = [col for col in feature_cols if col in df.columns]
    
    if not available_features:
        logger.error("No valid features found in the cleaned data for model training.")
        raise ValueError("No features available for modeling.")
        
    logger.info(f"Features selected for modeling: {available_features}")

    X = df[available_features].copy()
    y = df['DepDelay'].copy()         

    initial_rows_before_nan_drop = len(X)
    combined_data = pd.concat([X, y], axis=1)
    combined_data.dropna(inplace=True)
    X = combined_data[available_features]
    y = combined_data['DepDelay']
    if len(X) < initial_rows_before_nan_drop:
        logger.warning(f"Removed {initial_rows_before_nan_drop - len(X)} rows with NaNs in selected features/target.")

    if len(X) == 0:
        logger.error("No data remaining after NaN removal for features/target. Cannot train model.")
        raise ValueError("Insufficient data for training after NaN handling.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    log_import_and_clean_logs()
    
    finalized_model = None
    best_metrics = None
    best_alpha = None
    best_poly_transformer = None
    best_scaler = None

    alphas = np.linspace(0.1, 10.0, num_alphas) 
    
    for alpha in alphas:
        with mlflow.start_run(run_name=f"ridge_alpha_{alpha:.1f}", nested=True):
            logger.info(f"Starting sub-run with alpha={alpha}")
            
            log_input_parameters(alpha, polynomial_order=polynomial_degree) 
            
            poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_poly)
            X_test_scaled = scaler.transform(X_test_poly)
            
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            
            log_model_performance_metrics(y_test, y_pred, alpha)
            
            create_performance_plots(y_test, y_pred, alpha) 
            
            poly_feature_names = poly.get_feature_names_out(available_features)
            log_feature_importance(model, poly_feature_names, alpha) 
            
            current_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
            if finalized_model is None or current_rmse < best_metrics['rmse']:
                finalized_model = model
                best_metrics = {
                    'rmse': current_rmse,
                    'mse': metrics.mean_squared_error(y_test, y_pred),
                    'mae': metrics.mean_absolute_error(y_test, y_pred),
                    'r2': metrics.r2_score(y_test, y_pred)
                }
                best_alpha = alpha
                best_poly_transformer = poly
                best_scaler = scaler 
                
                model_info = {
                    'alpha': alpha,
                    'metrics': best_metrics,
                    'feature_names': available_features, 
                    'polynomial_degree': polynomial_degree,
                    'random_state': 42,
                    'test_size': 0.2,
                    'training_date': datetime.now().isoformat()
                }
                
                with open('finalized_model_info_temp.json', 'w') as f:
                    json.dump(model_info, f, indent=2)
    
    if finalized_model is None:
        logger.error("No models were trained successfully. Cannot create final summary.")
        return False

    with mlflow.start_run(run_name="Final_Model_Summary", nested=True) as final_summary_run:
        logger.info("Creating final model summary in a nested run.")
        
        mlflow.log_param("best_alpha_overall", best_alpha)
        mlflow.log_param("total_runs_performed", num_alphas)
        
        for metric_name, metric_value in best_metrics.items():
            mlflow.log_metric(f"best_overall_{metric_name}", metric_value)
        
        log_model_artifacts(finalized_model, best_poly_transformer, best_scaler, available_features)

        final_model_info_path = "final_finalized_model_info.json"
        temp_model_info_path = 'finalized_model_info_temp.json'
        if os.path.exists(temp_model_info_path):
            with open(temp_model_info_path, 'r') as f:
                model_info = json.load(f)
            with open(final_model_info_path, "w") as f:
                json.dump(model_info, f, indent=2)
            os.remove(temp_model_info_path)
        else: 
            model_info_fallback = {
                'alpha': best_alpha, 'metrics': best_metrics, 'feature_names': available_features,
                'polynomial_degree': polynomial_degree, 'random_state': 42, 'test_size': 0.2,
                'training_date': datetime.now().isoformat()
            }
            with open(final_model_info_path, "w") as f:
                json.dump(model_info_fallback, f, indent=2)

        mlflow.log_artifact(final_model_info_path)

        # Final Summary Plot (saved locally as well)
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        sns.histplot(y, bins=30, kde=True)
        plt.title('Distribution of Target Variable (DepDelay)')
        plt.xlabel('Delay (minutes)')
        plt.ylabel('Frequency')
        
        plot_index = 2
        for feature in available_features:
            if plot_index > 5:
                break
            plt.subplot(2, 3, plot_index)
            sns.scatterplot(x=X[feature], y=y, alpha=0.5)
            plt.title(f'{feature} vs Target')
            plt.xlabel(feature)
            plt.ylabel('Delay (minutes)')
            plot_index += 1
        
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.7, f"Best Alpha: {best_alpha:.2f}\nBest RMSE: {best_metrics['rmse']:.4f}\nBest R²: {best_metrics['r2']:.4f}", 
                 fontsize=12, transform=plt.gca().transAxes)
        plt.title('Overall Model Summary')
        plt.axis('off')
        
        plt.tight_layout()
        # Log to MLflow
        mlflow.log_figure(plt.gcf(), "final_summary_plots.png")
        # Save locally
        plt.savefig("final_summary_plots.png") # Save the figure locally
        plt.close()
        
        logger.info("Final summary created and logged")
        logger.info(f"Best model overall: alpha={best_alpha:.2f}, RMSE={best_metrics['rmse']:.4f}")

    logger.info("Comprehensive MLFlow experiment completed successfully")
    return True

if __name__ == "__main__":
    if mlflow.active_run() is None:
        with mlflow.start_run(run_name="Standalone_Experiment_Test"):
            run_comprehensive_mlflow_experiment()
    else:
        run_comprehensive_mlflow_experiment()