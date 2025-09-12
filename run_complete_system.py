"""
Master script to run the complete agricultural ML system
"""
import subprocess
import sys
import os
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR running {description}: {e}")
        return False

def main():
    """Run the complete system"""
    print("🚀 STARTING COMPLETE AGRICULTURAL ML SYSTEM")
    print("="*60)
    
    scripts_to_run = [
        ('src/data_collection/weather_collector.py', 'Weather Data Collection'),
        ('src/data_collection/agriculture_collector.py', 'Agricultural Data Collection'),
        ('src/ml_models/train_models.py', 'ML Model Training'),
    ]
    
    all_success = True
    
    for script_path, description in scripts_to_run:
        success = run_script(script_path, description)
        if not success:
            all_success = False
            break
    
    if all_success:
        print("\n🎉 ALL SYSTEMS READY!")
        print("="*60)
        print("✅ Data Collection Complete")
        print("✅ ML Models Trained")
        print("✅ Ready for Dashboard")
        print("\nTo start the dashboard, run:")
        print("python src/visualization/dashboard.py")
        print("\nThen visit: http://localhost:8050")
    else:
        print("\n❌ System setup failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()

