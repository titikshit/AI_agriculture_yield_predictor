"""
Configuration file for Agricultural ML Project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"

# API Configuration
IMD_API_BASE_URL = os.getenv('IMD_API_BASE_URL', 'https://mausam.imd.gov.in/api')
WEATHER_API_TIMEOUT = int(os.getenv('WEATHER_API_TIMEOUT', 30))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))

# Data Configuration
DATA_UPDATE_INTERVAL = int(os.getenv('DATA_UPDATE_INTERVAL', 3600))
CACHE_EXPIRY = int(os.getenv('CACHE_EXPIRY', 86400))

# Model Configuration
MODEL_RANDOM_STATE = int(os.getenv('MODEL_RANDOM_STATE', 42))
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
CV_FOLDS = int(os.getenv('CV_FOLDS', 5))

# Indian States and Crops
INDIAN_STATES = [
    'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi', 'Gujarat',
    'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
    'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Punjab', 'Rajasthan',
    'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'West Bengal'
]

MAJOR_CROPS = [
    'Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Soybean', 
    'Barley', 'Gram', 'Mustard', 'Groundnut', 'Sunflower', 'Jowar'
]

# IMD Station Codes (Major Cities)
IMD_STATIONS = {
    'Delhi': {'id': 42182, 'lat': 28.61, 'lon': 77.23},
    'Mumbai': {'id': 43003, 'lat': 19.07, 'lon': 72.88},
    'Chennai': {'id': 43279, 'lat': 13.09, 'lon': 80.27},
    'Kolkata': {'id': 42809, 'lat': 22.57, 'lon': 88.36},
    'Bangalore': {'id': 43295, 'lat': 12.97, 'lon': 77.59},
    'Hyderabad': {'id': 43128, 'lat': 17.39, 'lon': 78.49},
    'Pune': {'id': 43063, 'lat': 18.52, 'lon': 73.86},
    'Ahmedabad': {'id': 42647, 'lat': 23.02, 'lon': 72.57}
}

print(f"Configuration loaded successfully!")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")

