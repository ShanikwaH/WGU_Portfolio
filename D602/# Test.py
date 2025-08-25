# Test.py
# coding: utf-8
"""
Unit tests for the Flight Delay Prediction API
Tests both correctly and incorrectly formatted requests
Cross-platform compatible test file
"""

import pytest
import json
import sys
import os
from fastapi.testclient import TestClient

# Add the current directory to the Python path to import the API module
# CORRECTED LINE 11: Use a raw string for the absolute path
sys.path.insert(0, os.path.dirname(os.path.abspath(r"C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D602/Task 3/API_Python_1_0_0.py")))

# Import the FastAPI app instance from your main API file
try:
    from API_Python_1_0_0 import app, create_airport_encoding, time_to_seconds, predict_delay
    
    # Initialize the TestClient with your FastAPI app instance
    client = TestClient(app)
    API_AVAILABLE = True
except ImportError as e:
    # This block handles cases where API_Python_1_0_0.py itself cannot be imported
    # (e.g., if the file is missing, misnamed, or has a syntax error)
    print(f"WARNING: Could not import API module for testing: {e}")
    API_AVAILABLE = False
    client = None

class TestAPIEndpoints:
    """Test class for API endpoints using FastAPI's TestClient"""
    
    def test_root_endpoint_correct_format(self):
        """Test 1: Root endpoint with correctly formatted request"""
        # Skip this test if the API module couldn't be loaded
        if not API_AVAILABLE:
            pytest.skip("API module not available for testing.")
        
        # Make a GET request to the root endpoint using the TestClient
        response = client.get("/")
        
        # Assert the HTTP status code is 200 (OK)
        assert response.status_code == 200
        
        # Parse the JSON response
        data = response.json()
        
        # Assert specific keys and values in the JSON response
        assert "message" in data
        assert "status" in data
        assert data["status"] == "operational"
        assert "Flight Delay Prediction API is functional" in data["message"]
        
        print("✓ Test 1 passed: Root endpoint returns correct JSON message")
    
    def test_predict_delays_correct_format(self):
        """Test 2: Predict delays endpoint with correctly formatted request"""
        if not API_AVAILABLE:
            pytest.skip("API module not available for testing.")
        
        # Define parameters for the GET request
        params = {
            "departure_airport": "ATL",
            "arrival_airport": "DFW",
            "departure_time": "14:30",
            "arrival_time": "16:45"
        }
        
        # Make a GET request to the /predict/delays endpoint with parameters
        response = client.get("/predict/delays", params=params)
        
        # Assert the HTTP status code is 200 (OK)
        assert response.status_code == 200
        
        # Parse the JSON response
        data = response.json()
        
        # Assert expected keys are present in the response
        assert "average_departure_delay" in data
        assert "departure_airport" in data
        assert "arrival_airport" in data
        assert "departure_time" in data
        assert "arrival_time" in data
        assert "units" in data
        
        # Assert specific values
        assert data["departure_airport"] == "ATL"
        assert data["arrival_airport"] == "DFW"
        assert data["departure_time"] == "14:30"
        assert data["arrival_time"] == "16:45"
        assert data["units"] == "minutes"
        assert isinstance(data["average_departure_delay"], (int, float)) # Check type of delay
        
        print("✓ Test 2 passed: Predict delays endpoint returns correct JSON response")
    
    def test_predict_delays_incorrect_format_invalid_airport(self):
        """Test 3: Predict delays endpoint with incorrectly formatted request (invalid airport)"""
        if not API_AVAILABLE:
            pytest.skip("API module not available for testing.")
        
        # Test with invalid airport code
        params = {
            "departure_airport": "INVALID", # This airport is not in your encodings
            "arrival_airport": "DFW",
            "departure_time": "14:30",
            "arrival_time": "16:45"
        }
        
        response = client.get("/predict/delays", params=params)
        
        # Assert the HTTP status code is 400 (Bad Request) as per your API logic
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
        
        print("✓ Test 3 passed: Invalid airport code returns appropriate error")

    def test_predict_delays_incorrect_format_invalid_time(self):
        """Test 4: Predict delays endpoint with incorrectly formatted request (invalid time)"""
        if not API_AVAILABLE:
            pytest.skip("API module not available for testing.")
        
        # Test with invalid time format (e.g., hour out of range)
        params = {
            "departure_airport": "ATL",
            "arrival_airport": "DFW",
            "departure_time": "25:30",  # Invalid hour (25)
            "arrival_time": "16:45"
        }
        
        response = client.get("/predict/delays", params=params)
        
        # Assert the HTTP status code is 400 (Bad Request)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Time must be in HH:MM format" in data["detail"] # Check for specific error message
        
        print("✓ Test 4 passed: Invalid time format returns appropriate error")

# You can add more TestClient-based tests here for other edge cases,
# such as missing parameters (which would return 422 if using Pydantic directly).

# Your existing utility function tests and file requirement tests would follow here
# (TestUtilityFunctions and TestFileRequirements classes from your Test.py)