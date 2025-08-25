# coding: utf-8

"""
Data Cleaning Script - Version 2 (Final) - Debugged
D602 Task 2 - Flight Delay Analysis

This script cleans data to only include departures from the chosen airport
and implements comprehensive data cleaning steps.

Author: Student
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def filter_by_airport(df):
    """
    Filter data to only include departures from Atlanta airport.
    Prioritizes filtering by numeric ID, falls back to airport code.
    (Fix for Warning 3)
    """
    logger.info("Filtering to Atlanta departures...")
    
    # Atlanta Hartsfield-Jackson International Airport ID and Code
    atl_airport_id = 10397
    atl_airport_code = 'ATL' # Assuming 'ATL' is the airport code for Atlanta
    
    initial_rows = len(df)
    df_filtered = df.copy() # Initialize with a copy of the original dataframe
    
    # Attempt to filter by numeric airport ID first
    numeric_id_cols = ['ORIGINAIRPORTID', 'OriginAirportID'] # 'ORIGIN_AIRPORT_ID' if it was renamed this way
    found_id_col = None
    for col in numeric_id_cols:
        if col in df.columns:
            found_id_col = col
            break
            
    if found_id_col:
        # Ensure the ID column is numeric, coercing errors to NaN and then filling to avoid issues
        df[found_id_col] = pd.to_numeric(df[found_id_col], errors='coerce')
        df_filtered = df[df[found_id_col] == atl_airport_id].copy()
        logger.info(f"Filtered from {initial_rows} to {len(df_filtered)} rows using numeric airport ID '{found_id_col}'.")
    else:
        # Fallback: If numeric ID column is not found, try filtering by airport code
        code_cols = ['ORIGIN_AIRPORT_CODE', 'ORIGIN'] # Check for the newly mapped code column or original 'ORIGIN'
        found_code_col = None
        for col in code_cols:
            if col in df.columns:
                found_code_col = col
                break
                
        if found_code_col:
            df_filtered = df[df[found_code_col] == atl_airport_code].copy()
            logger.info(f"Filtered from {initial_rows} to {len(df_filtered)} rows using airport code '{found_code_col}'.")
        else:
            logger.warning("Could not find any suitable airport ID or code column ('ORIGINAIRPORTID', 'ORIGIN_AIRPORT_CODE', 'ORIGIN'). Using all data without filtering for Atlanta.")
            df_filtered = df.copy() # No filtering applied, use all data
            
    return df_filtered

def clean_missing_values(df):
    """
    Data cleaning step 1: Handle missing values in critical columns.
    Updated key columns to match new naming conventions.
    """
    logger.info("Cleaning step 1: Handling missing values...")
    
    # Columns that cannot have missing values for analysis (use standardized names)
    critical_columns = ['DepDelay', 'YEAR', 'MONTH', 'DAYOFMONTH', 'ORIGIN_AIRPORT_CODE', 'DEST_AIRPORT_CODE']
    
    initial_rows = len(df)
    removed_rows = 0
    for col in critical_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"Removing {missing_count} rows with missing {col}")
                df = df.dropna(subset=[col])
                removed_rows += missing_count # Accumulate removed rows
    
    logger.info(f"Removed {initial_rows - len(df)} rows with missing critical values")
    return df

def remove_outliers(df):
    """
    Data cleaning step 2: Remove extreme outliers in delay times.
    """
    logger.info("Cleaning step 2: Removing extreme outliers...")
    
    initial_rows = len(df)
    
    # Remove flights with extreme delays (> 24 hours or < -1 hour)
    if 'DepDelay' in df.columns: # Use standardized 'DepDelay'
        df['DepDelay'] = pd.to_numeric(df['DepDelay'], errors='coerce') # Ensure numeric
        df = df[df['DepDelay'] <= 1440]  # 24 hours in minutes
        df = df[df['DepDelay'] >= -60]   # Allow 1 hour early
    
    # Remove flights with invalid times
    if 'CRSDepTime' in df.columns: # Use standardized 'CRSDepTime'
        df['CRSDepTime'] = pd.to_numeric(df['CRSDepTime'], errors='coerce') # Ensure numeric
        df = df[df['CRSDepTime'] >= 0]
        df = df[df['CRSDepTime'] <= 2359]
    
    logger.info(f"Removed {initial_rows - len(df)} rows with extreme values")
    return df

def clean_duplicates_and_invalid(df):
    """
    Data cleaning step 3: Remove duplicates and invalid entries.
    Updated column names for checks.
    """
    logger.info("Cleaning step 3: Removing duplicates and invalid entries...")
    
    initial_rows = len(df)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Remove rows where year is not 2025 (assuming this is 2025 data)
    if 'YEAR' in df.columns:
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce') # Ensure numeric
        df = df[df['YEAR'] == 2025]
    
    # Remove rows with invalid month/day combinations (use standardized names)
    if 'MONTH' in df.columns and 'DAYOFMONTH' in df.columns:
        df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
        df['DAYOFMONTH'] = pd.to_numeric(df['DAYOFMONTH'], errors='coerce')
        df = df[df['MONTH'].between(1, 12)]
        df = df[df['DAYOFMONTH'].between(1, 31)]
    
    logger.info(f"Removed {initial_rows - len(df)} rows with duplicates and invalid entries")
    return df

def add_cleaning_features(df):
    """
    Add features that may be useful for analysis after cleaning.
    Updated 'DEPDELAY' to standardized name.
    """
    logger.info("Adding cleaning-related features...")
    
    # Add a flag for delayed flights
    if 'DepDelay' in df.columns: # Use standardized 'DepDelay'
        df['IS_DELAYED'] = (df['DepDelay'] > 0).astype(int)
    
    # Add delay category
    if 'DepDelay' in df.columns: # Use standardized 'DepDelay'
        conditions = [
            (df['DepDelay'] <= 0),
            (df['DepDelay'] <= 15),
            (df['DepDelay'] <= 60),
            (df['DepDelay'] > 60)
        ]
        choices = ['On Time', 'Minor Delay', 'Moderate Delay', 'Major Delay']
        df['DELAY_CATEGORY'] = np.select(conditions, choices, default='Unknown')
    
    return df

def filter_and_clean_data():
    """
    Clean data to only departures from Atlanta airport and perform comprehensive cleaning.
    This is the final version with enhanced functionality.
    """
    try:
        # Load the imported dataset
        logger.info("Loading imported dataset...")
        df = pd.read_csv("imported_data.csv")
        
        logger.info(f"Initial dataset shape: {df.shape}")
        
        # Filter by airport (will now correctly handle codes or IDs)
        df_filtered = filter_by_airport(df)
        
        # Apply cleaning steps
        df_cleaned = clean_missing_values(df_filtered)
        df_cleaned = remove_outliers(df_cleaned)
        df_cleaned = clean_duplicates_and_invalid(df_cleaned)
        
        # Add features
        df_final = add_cleaning_features(df_cleaned)
        
        # Save the filtered and cleaned dataset
        output_file = "cleaned_data.csv"
        df_final.to_csv(output_file, index=False)
        
        logger.info(f"Data cleaning complete. Saved as '{output_file}'")
        logger.info(f"Final dataset shape: {df_final.shape}")
        
        # Print summary statistics
        if 'DepDelay' in df_final.columns:
            logger.info(f"Delay statistics:")
            logger.info(f"  Mean delay: {df_final['DepDelay'].mean():.2f} minutes")
            logger.info(f"  Median delay: {df_final['DepDelay'].median():.2f} minutes")
            logger.info(f"  Delayed flights: {df_final['IS_DELAYED'].sum()} out of {len(df_final)}")
        
        return df_final
        
    except FileNotFoundError:
        logger.error("Error: imported_data.csv not found. Run import script first.")
        return None
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Starting data cleaning process...")
    df = filter_and_clean_data()
    if df is not None:
        logger.info("Data cleaning script completed successfully.")
    else:
        logger.error("Data cleaning script failed.")