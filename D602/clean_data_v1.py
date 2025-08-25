# coding: utf-8

"""
Data Cleaning Script - Version 1
D602 Task 2 - Flight Delay Analysis

This script cleans data to only include departures from the chosen airport
and implements basic data cleaning steps.

Author: Student
Date: 2025
"""

import pandas as pd
import numpy as np

def filter_and_clean_data():
    """
    Clean data to only departures from Atlanta airport and perform basic cleaning.
    This is the initial version with basic functionality.
    """
    try:
        # Load the imported dataset
        print("Loading imported dataset...")
        df = pd.read_csv("imported_data.csv")
        
        # Filter to only departures from Atlanta airport (ATL = 10397)
        print("Filtering to Atlanta departures...")
        atl_airport_id = 10397  # Atlanta Hartsfield-Jackson International Airport
        
        # Check if we have the right column for filtering
        if 'ORIGINAIRPORTID' in df.columns:
            df_filtered = df[df['ORIGINAIRPORTID'] == atl_airport_id].copy()
        elif 'OriginAirportID' in df.columns:
            df_filtered = df[df['OriginAirportID'] == atl_airport_id].copy()
        else:
            print("Warning: Could not find airport ID column. Using all data.")
            df_filtered = df.copy()
        
        print(f"Filtered dataset shape: {df_filtered.shape}")
        
        # Data cleaning step 1: Remove rows with missing departure delay
        print("Cleaning step 1: Removing rows with missing departure delay...")
        initial_rows = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=['DEPDELAY'])
        print(f"Removed {initial_rows - len(df_filtered)} rows with missing departure delay")
        
        # Data cleaning step 2: Remove extreme outliers (delays > 24 hours)
        print("Cleaning step 2: Removing extreme outliers...")
        initial_rows = len(df_filtered)
        df_filtered = df_filtered[df_filtered['DEPDELAY'] <= 1440]  # 24 hours in minutes
        df_filtered = df_filtered[df_filtered['DEPDELAY'] >= -60]  # Allow 1 hour early
        print(f"Removed {initial_rows - len(df_filtered)} rows with extreme delays")
        
        # Save the filtered dataset
        output_file = "filtered_ATL_dataset_v1.csv"
        df_filtered.to_csv(output_file, index=False)
        
        print(f"Data cleaning complete. Saved as '{output_file}'")
        print(f"Final dataset shape: {df_filtered.shape}")
        
        return df_filtered
        
    except FileNotFoundError:
        print("Error: imported_data.csv not found. Run import script first.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    df = filter_and_clean_data()
    if df is not None:
        print("Data cleaning script completed successfully.")
    else:
        print("Data cleaning script failed.") 