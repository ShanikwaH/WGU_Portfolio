
# coding: utf-8

"""
MLFlow Experiment Script - Version 1
D602 Task 2 - Flight Delay Analysis

This script implements an MLFlow experiment for polynomial regression
to predict flight delays. This is the initial version with basic MLFlow tracking.
Based on poly_regressor_Python_1.0.0.py template.

Author: Student
Date: 2025
"""

import datetime
import pandas as pd
import argparse
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import mlflow
import mlflow.sklearn
import logging
import os
import pickle
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """
    Load the cleaned data and prepare it for modeling.
    Based on poly_regressor_Python_1.0.0.py data loading approach.
    """
    logger.info("Loading cleaned data...")
    df = pd.read_csv("cleaned_data.csv")
    
    # Select features for modeling based on poly_regressor template
    feature_columns = ['YEAR', 'MONTH', 'DAYOFMONTH', 'DAYOFWEEK', 
                      'DESTAIRPORTID', 'CRSDEPTIME', 'DISTANCE']
    
    # Ensure all required columns exist
    available_columns = [col for col in feature_columns if col in df.columns]
    logger.info(f"Using features: {available_columns}")
    
    X = df[available_columns].copy()
    y = df['DEPDELAY'].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    return X, y

def create_polynomial_features(X, degree=2):
    """
    Create polynomial features for the model.
    Based on poly_regressor_Python_1.0.0.py approach.
    """
    logger.info(f"Creating polynomial features with degree {degree}...")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly

def train_model(X, y, alpha=1.0):
    """
    Train a Ridge regression model with polynomial features.
    Based on poly_regressor_Python_1.0.0.py training approach.
    """
    logger.info(f"Training Ridge regression model with alpha={alpha}...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    
    return model, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def run_mlflow_experiment():
    """
    Run the MLFlow experiment with basic tracking.
    Based on poly_regressor_Python_1.0.0.py MLFlow implementation.
    """
    logger.info("Starting MLFlow experiment...")
    
    # Set up MLFlow
    mlflow.set_experiment("Flight_Delay_Prediction_v1")
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Create polynomial features
    X_poly, poly_transformer = create_polynomial_features(X, degree=2)
    
    # Test different alpha values based on poly_regressor template
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    best_score = float('inf')
    best_alpha = None
    best_model = None
    
    for alpha in alpha_values:
        logger.info(f"Testing alpha = {alpha}")
        
        with mlflow.start_run():
            # Log parameters based on poly_regressor template
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("polynomial_degree", 2)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("model_type", "Ridge")
            
            # Train model
            model, metrics_dict = train_model(X_poly, y, alpha)
            
            # Log metrics
            mlflow.log_metric("mse", metrics_dict['mse'])
            mlflow.log_metric("rmse", metrics_dict['rmse'])
            mlflow.log_metric("mae", metrics_dict['mae'])
            mlflow.log_metric("r2", metrics_dict['r2'])
            
            # Log model
            mlflow.sklearn.log_model(model, "ridge_model")
            
            # Track best model
            if metrics_dict['rmse'] < best_score:
                best_score = metrics_dict['rmse']
                best_alpha = alpha
                best_model = model
            
            logger.info(f"Alpha {alpha}: RMSE = {metrics_dict['rmse']:.2f}, R² = {metrics_dict['r2']:.3f}")
    
    # Save best model
    if best_model is not None:
        logger.info(f"Best model: alpha = {best_alpha}, RMSE = {best_score:.2f}")
        
        # Save model to file
        with open("best_model_v1.pkl", "wb") as f:
            pickle.dump(best_model, f)
        
        # Save polynomial transformer
        with open("poly_transformer_v1.pkl", "wb") as f:
            pickle.dump(poly_transformer, f)
        
        logger.info("Best model saved as 'best_model_v1.pkl'")
    
    return best_model, best_alpha, best_score

if __name__ == "__main__":
    logger.info("Starting MLFlow experiment version 1...")
    best_model, best_alpha, best_score = run_mlflow_experiment()
    logger.info("MLFlow experiment completed successfully.") 