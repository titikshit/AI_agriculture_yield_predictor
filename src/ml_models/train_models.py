"""
Machine Learning Pipeline - Fixed Categorical Handling
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from datetime import datetime
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import *
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "data" / "models"
    MODEL_RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

class FixedMLPipeline:
    """ML Pipeline with fixed categorical handling"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
        
    def load_and_merge_data(self):
        """Load and merge data"""
        print("="*60)
        print("🚀 LOADING AND MERGING DATA")
        print("="*60)
        
        # Load agricultural data
        ag_file = RAW_DATA_DIR / "agricultural_data_latest.csv"
        if not ag_file.exists():
            print("❌ No agricultural data found!")
            return None
            
        self.df = pd.read_csv(ag_file)
        print(f"✅ Loaded agricultural data: {len(self.df)} records")
        
        # Load weather data if available
        weather_file = RAW_DATA_DIR / "weather_data_latest.csv"
        if weather_file.exists():
            weather_df = pd.read_csv(weather_file)
            print(f"✅ Loaded weather data: {len(weather_df)} records")
            
            # Create state-level weather averages
            weather_summary = weather_df.groupby('state').agg({
                'temperature_c': 'mean',
                'humidity_percent': 'mean',
                'rainfall_mm': 'mean'
            }).reset_index()
            
            # Merge with agricultural data
            self.df = self.df.merge(weather_summary, left_on='state', right_on='state', how='left')
        else:
            print("⚠️ No weather data found, using agricultural data only")
        
        # Convert all object columns to string (fixes categorical issue)
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.df[col]):
                self.df[col] = self.df[col].astype(str)
        
        # Fill missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        print(f"✅ Final dataset: {len(self.df)} records, {len(self.df.columns)} columns")
        return self.df
        
    def feature_engineering(self):
        """Create additional features"""
        print("\n🔧 FEATURE ENGINEERING")
        print("-" * 40)
        
        # Time features
        if 'year' in self.df.columns:
            self.df['years_since_2000'] = self.df['year'] - 2000
        
        # Find weather columns (handle duplicates from merging)
        temp_cols = [col for col in self.df.columns if 'temperature' in col.lower()]
        rainfall_cols = [col for col in self.df.columns if 'rainfall' in col.lower()]
        humidity_cols = [col for col in self.df.columns if 'humidity' in col.lower()]
        
        # Use the first available weather column or create composite
        if temp_cols:
            self.df['temperature_clean'] = self.df[temp_cols[0]]
            if len(temp_cols) > 1:
                # If multiple temp columns, take the average
                temp_data = self.df[temp_cols].fillna(method='ffill', axis=1)
                self.df['temperature_clean'] = temp_data.mean(axis=1)
        
        if rainfall_cols:
            self.df['rainfall_clean'] = self.df[rainfall_cols[0]]
            if len(rainfall_cols) > 1:
                rainfall_data = self.df[rainfall_cols].fillna(method='ffill', axis=1)
                self.df['rainfall_clean'] = rainfall_data.mean(axis=1)
        
        if humidity_cols:
            self.df['humidity_clean'] = self.df[humidity_cols[0]]
            if len(humidity_cols) > 1:
                humidity_data = self.df[humidity_cols].fillna(method='ffill', axis=1)
                self.df['humidity_clean'] = humidity_data.mean(axis=1)
        
        # Create interaction features
        if 'temperature_clean' in self.df.columns and 'humidity_clean' in self.df.columns:
            self.df['heat_index'] = (self.df['temperature_clean'] * self.df['humidity_clean']) / 100
        
        if 'rainfall_clean' in self.df.columns:
            self.df['drought_risk'] = (self.df['rainfall_clean'] < 400).astype(int)
            self.df['flood_risk'] = (self.df['rainfall_clean'] > 1500).astype(int)
            self.df['optimal_rainfall'] = ((self.df['rainfall_clean'] >= 600) & 
                                         (self.df['rainfall_clean'] <= 1200)).astype(int)
        
        # Area features
        if 'area_hectares' in self.df.columns:
            self.df['log_area'] = np.log1p(self.df['area_hectares'])
        
        # Season encoding
        if 'season' in self.df.columns:
            season_mapping = {'Kharif': 1, 'Rabi': 2, 'Summer': 3}
            self.df['season_numeric'] = self.df['season'].map(season_mapping).fillna(0)
        
        print(f"✅ Feature engineering complete. Dataset now has {len(self.df.columns)} columns")
        return self.df
    
    def prepare_ml_features(self):
        """Prepare features for ML with proper categorical handling"""
        print("\n📋 PREPARING ML FEATURES")
        print("-" * 40)
        
        # Handle categorical variables properly
        categorical_cols = ['state', 'district', 'crop', 'season']
        
        for col in categorical_cols:
            if col in self.df.columns:
                # Convert to string first, then handle missing values
                self.df[col] = self.df[col].astype(str)
                self.df[col] = self.df[col].replace(['nan', 'None', 'NaN'], 'Unknown')
                self.df[col] = self.df[col].fillna('Unknown')
                
                # Label encode
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.encoders[col] = le
                print(f"✅ Encoded {col}: {len(le.classes_)} categories")
        
        # Select features for modeling
        potential_features = [
            'year', 'years_since_2000', 'season_numeric',
            'temperature_clean', 'humidity_clean', 'rainfall_clean',
            'heat_index', 'drought_risk', 'flood_risk', 'optimal_rainfall',
            'log_area', 'area_hectares',
            'state_encoded', 'district_encoded', 'crop_encoded', 'season_encoded'
        ]
        
        # Only include features that exist and are numeric
        self.feature_names = []
        for feature in potential_features:
            if feature in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[feature]):
                    self.feature_names.append(feature)
        
        print(f"✅ Selected {len(self.feature_names)} features:")
        for i, feature in enumerate(self.feature_names):
            print(f"   {i+1:2d}. {feature}")
        
        return self.feature_names
    
    def train_models(self):
        """Train ML models"""
        print("\n🤖 TRAINING MACHINE LEARNING MODELS")
        print("-" * 40)
        
        # Find target variable
        target_candidates = ['yield_tonnes_per_hectare', 'yield_tons_per_hectare', 'yield']
        target_col = None
        
        for candidate in target_candidates:
            if candidate in self.df.columns:
                target_col = candidate
                break
        
        if target_col is None:
            print("❌ No target variable found!")
            print(f"Available columns: {list(self.df.columns)}")
            return None
        
        print(f"✅ Using target variable: {target_col}")
        
        # Prepare data
        X = self.df[self.feature_names].copy()
        y = self.df[target_col].copy()
        
        # Handle missing and infinite values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], y.median())
        
        print(f"✅ Training data shape: X={X.shape}, y={y.shape}")
        print(f"✅ Target range: {y.min():.3f} to {y.max():.3f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=MODEL_RANDOM_STATE
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_config = {
            'Random Forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    random_state=MODEL_RANDOM_STATE,
                    n_jobs=-1
                ),
                'use_scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=MODEL_RANDOM_STATE
                ),
                'use_scaled': False
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'use_scaled': True
            }
        }
        
        # Train each model
        for model_name, config in models_config.items():
            print(f"\n🔄 Training {model_name}...")
            
            try:
                model = config['model']
                use_scaled = config['use_scaled']
                
                # Select appropriate data
                if use_scaled:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    cv_data_X = X_train_scaled
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    cv_data_X = X_train
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(model, cv_data_X, y_train, cv=3, scoring='r2')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception:
                    cv_mean, cv_std = r2, 0
                
                # Store results
                self.results[model_name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'predictions': y_pred,
                    'actuals': y_test
                }
                
                print(f"   ✅ {model_name} Performance:")
                print(f"      MAE:  {mae:.3f}")
                print(f"      RMSE: {rmse:.3f}")
                print(f"      R²:   {r2:.3f}")
                print(f"      CV:   {cv_mean:.3f} ± {cv_std:.3f}")
                
            except Exception as e:
                print(f"   ❌ Failed to train {model_name}: {e}")
        
        return self.results
    
    def create_visualizations(self):
        """Create model performance visualizations"""
        print("\n📊 CREATING VISUALIZATIONS")
        print("-" * 40)
        
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            # 1. Model comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            models = list(self.results.keys())
            r2_scores = [self.results[model]['r2'] for model in models]
            mae_scores = [self.results[model]['mae'] for model in models]
            
            # R² comparison
            bars1 = axes[0, 0].bar(models, r2_scores)
            axes[0, 0].set_title('Model R² Comparison')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_ylim(0, 1)
            for i, v in enumerate(r2_scores):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # MAE comparison
            bars2 = axes[0, 1].bar(models, mae_scores, color='orange')
            axes[0, 1].set_title('Model MAE Comparison')
            axes[0, 1].set_ylabel('Mean Absolute Error')
            
            # Best model predictions vs actual
            best_model = max(models, key=lambda x: self.results[x]['r2'])
            predictions = self.results[best_model]['predictions']
            actuals = self.results[best_model]['actuals']
            
            axes[1, 0].scatter(actuals, predictions, alpha=0.6)
            axes[1, 0].plot([actuals.min(), actuals.max()], 
                           [actuals.min(), actuals.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual Yield')
            axes[1, 0].set_ylabel('Predicted Yield')
            axes[1, 0].set_title(f'Predictions vs Actual ({best_model})')
            
            # Residuals
            residuals = actuals - predictions
            axes[1, 1].scatter(predictions, residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Predicted Yield')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residual Plot')
            
            plt.tight_layout()
            plt.savefig(MODELS_DIR / 'model_performance.png', dpi=300, bbox_inches='tight')
            print("✅ Model performance plots saved")
            plt.close()
            
            # 2. Feature importance (Random Forest)
            if 'Random Forest' in self.results:
                rf_model = self.results['Random Forest']['model']
                importance = rf_model.feature_importances_
                
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True)
                
                plt.figure(figsize=(10, 8))
                top_features = feature_importance_df.tail(12)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title('Feature Importance (Random Forest)')
                plt.tight_layout()
                plt.savefig(MODELS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
                print("✅ Feature importance plot saved")
                plt.close()
            
        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")
    
    def save_models_and_data(self):
        """Save trained models and processed data"""
        print("\n💾 SAVING MODELS AND DATA")
        print("-" * 40)
        
        if not self.results:
            print("❌ No models to save!")
            return
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_model = self.results[best_model_name]['model']
        
        # Save model files
        model_files = {
            'best_model.pkl': best_model,
            'scaler.pkl': self.scaler,
            'encoders.pkl': self.encoders,
            'feature_names.pkl': self.feature_names
        }
        
        for filename, obj in model_files.items():
            joblib.dump(obj, MODELS_DIR / filename)
            print(f"✅ Saved {filename}")
        
        # Save processed data
        processed_file = PROCESSED_DATA_DIR / 'processed_agricultural_data.csv'
        self.df.to_csv(processed_file, index=False)
        print(f"✅ Processed data saved: {processed_file}")
        
        # Save model summary
        import json
        summary = {
            'best_model': best_model_name,
            'training_date': datetime.now().isoformat(),
            'model_performance': {
                name: {k: v for k, v in results.items() 
                       if k not in ['model', 'predictions', 'actuals']} 
                for name, results in self.results.items()
            },
            'feature_names': self.feature_names,
            'data_shape': self.df.shape
        }
        
        with open(MODELS_DIR / 'model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Best model: {best_model_name} (R² = {self.results[best_model_name]['r2']:.3f})")
        return best_model_name
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("🌾" + "="*58 + "🌾")
        print("   FIXED AGRICULTURAL ML PIPELINE")
        print("🌾" + "="*58 + "🌾")
        
        # Execute pipeline steps
        if self.load_and_merge_data() is None:
            return None
        
        self.feature_engineering()
        self.prepare_ml_features()
        
        results = self.train_models()
        if results is None:
            return None
        
        self.create_visualizations()
        best_model = self.save_models_and_data()
        
        print("\n🎉" + "="*58 + "🎉")
        print("   PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎉" + "="*58 + "🎉")
        print(f"✅ Data processed: {len(self.df):,} records")
        print(f"✅ Features created: {len(self.feature_names)}")
        print(f"✅ Models trained: {len(results)}")
        print(f"✅ Best model: {best_model}")
        print(f"✅ Visualizations created")
        print("\n🚀 Ready to launch dashboard!")
        
        return results

def main():
    """Main function"""
    pipeline = FixedMLPipeline()
    results = pipeline.run_complete_pipeline()
    
    if results:
        print("\n🎯 Next steps:")
        print("   1. Run: python src/visualization/dashboard.py")
        print("   2. Open: http://localhost:8050")
    else:
        print("\n❌ Pipeline failed!")

if __name__ == "__main__":
    main()
