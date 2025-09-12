"""
Real Weather Data Collection from IMD APIs
"""
import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

class IMDWeatherCollector:
    """Collect real weather data from India Meteorological Department"""
    
    def __init__(self):
        self.base_url = IMD_API_BASE_URL
        self.timeout = WEATHER_API_TIMEOUT
        self.max_retries = MAX_RETRIES
        
    def get_current_weather(self, station_id):
        """Get current weather for a station"""
        url = f"{self.base_url}/current_wx_api.php?id={station_id}"
        
        for attempt in range(self.max_retries):
            try:
                print(f"Fetching weather for station {station_id} (attempt {attempt + 1})")
                response = requests.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ Successfully fetched data for station {station_id}")
                    return data
                else:
                    print(f"⚠️ API returned status code: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"⏱️ Timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Request error: {e}")
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response")
                
            if attempt < self.max_retries - 1:
                time.sleep(5)  # Wait before retry
                
        print(f"❌ Failed to fetch data for station {station_id} after {self.max_retries} attempts")
        return None
    
    def get_district_rainfall(self):
        """Get district-wise rainfall data"""
        url = f"{self.base_url}/districtwise_rainfall_api.php"
        
        try:
            print("Fetching district-wise rainfall data...")
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Successfully fetched rainfall data for {len(data)} districts")
                return data
            else:
                print(f"⚠️ API returned status code: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching rainfall data: {e}")
            return None
    
    def collect_all_weather_data(self):
        """Collect weather data from all sources"""
        print("="*60)
        print("COLLECTING REAL WEATHER DATA FROM IMD")
        print("="*60)
        
        all_weather_data = []
        
        # 1. Collect station weather data
        print("\n1. COLLECTING STATION WEATHER DATA")
        print("-" * 40)
        
        for city, info in IMD_STATIONS.items():
            print(f"Processing {city}...")
            current_data = self.get_current_weather(info['id'])
            
            if current_data:
                weather_record = {
                    'timestamp': datetime.now().isoformat(),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'city': city,
                    'state': self.get_state_from_city(city),
                    'station_id': info['id'],
                    'latitude': info['lat'],
                    'longitude': info['lon'],
                    'temperature_c': self.extract_numeric(current_data.get('temperature', 0)),
                    'humidity_percent': self.extract_numeric(current_data.get('humidity', 0)),
                    'pressure_hpa': self.extract_numeric(current_data.get('mslp', 0)),
                    'wind_speed_kmph': self.extract_numeric(current_data.get('wind_speed', 0)),
                    'wind_direction': current_data.get('wind_direction', 'Unknown'),
                    'rainfall_mm': self.extract_numeric(current_data.get('last_24hrs_rainfall', 0)),
                    'weather_condition': current_data.get('weather_code', 'Unknown'),
                    'data_source': 'IMD_Station'
                }
                all_weather_data.append(weather_record)
            
            time.sleep(2)  # Be respectful to the API
        
        # 2. Collect district rainfall data
        print("\n2. COLLECTING DISTRICT RAINFALL DATA")
        print("-" * 40)
        
        rainfall_data = self.get_district_rainfall()
        if rainfall_data and isinstance(rainfall_data, list):
            for district_info in rainfall_data[:20]:  # Limit to first 20 districts
                weather_record = {
                    'timestamp': datetime.now().isoformat(),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'city': district_info.get('district', 'Unknown'),
                    'state': district_info.get('state', 'Unknown'),
                    'station_id': f"DIST_{district_info.get('district_code', 'UNK')}",
                    'latitude': 0,  # Would need geocoding
                    'longitude': 0,
                    'temperature_c': 0,  # Not available in rainfall data
                    'humidity_percent': 0,
                    'pressure_hpa': 0,
                    'wind_speed_kmph': 0,
                    'wind_direction': 'Unknown',
                    'rainfall_mm': self.extract_numeric(district_info.get('rainfall', 0)),
                    'weather_condition': 'Rainfall_Data',
                    'data_source': 'IMD_District_Rainfall'
                }
                all_weather_data.append(weather_record)
        
        # 3. Save collected data
        if all_weather_data:
            self.save_weather_data(all_weather_data)
        
        return all_weather_data
    
    def extract_numeric(self, value):
        """Extract numeric value from string or return 0"""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                import re
                numeric_str = re.findall(r'[\d.]+', value)
                if numeric_str:
                    return float(numeric_str[0])
        except:
            pass
        return 0.0
    
    def get_state_from_city(self, city):
        """Map city to state"""
        city_state_map = {
            'Delhi': 'Delhi',
            'Mumbai': 'Maharashtra',
            'Chennai': 'Tamil Nadu',
            'Kolkata': 'West Bengal',
            'Bangalore': 'Karnataka',
            'Hyderabad': 'Telangana',
            'Pune': 'Maharashtra',
            'Ahmedabad': 'Gujarat'
        }
        return city_state_map.get(city, 'Unknown')
    
    def save_weather_data(self, weather_data):
        """Save weather data to CSV files"""
        print("\n3. SAVING WEATHER DATA")
        print("-" * 40)
        
        df = pd.DataFrame(weather_data)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_file = RAW_DATA_DIR / f"weather_data_{timestamp}.csv"
        
        # Save timestamped version
        df.to_csv(timestamped_file, index=False)
        print(f"✓ Saved timestamped data: {timestamped_file}")
        
        # Save as latest version
        latest_file = RAW_DATA_DIR / "weather_data_latest.csv"
        df.to_csv(latest_file, index=False)
        print(f"✓ Saved latest data: {latest_file}")
        
        print(f"✓ Total records saved: {len(df)}")
        print(f"✓ Data columns: {list(df.columns)}")

def main():
    """Main function to run weather data collection"""
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect weather data
    collector = IMDWeatherCollector()
    weather_data = collector.collect_all_weather_data()
    
    if weather_data:
        print("\n" + "="*60)
        print("WEATHER DATA COLLECTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Collected {len(weather_data)} weather records")
        print("✓ Data saved to data/raw/ directory")
        print("✓ Ready for next step: Agricultural data collection")
    else:
        print("\n❌ Weather data collection failed!")

if __name__ == "__main__":
    main()

