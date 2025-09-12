"""
Machine Learning Pipeline for Agricultural Data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from datetime import datetime
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

warnings.filterwarnings('ignore')

class AgriculturalMLPipeline:
    """Complete ML Pipeline for Agricultural Data Analysis"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
        
    def load_and_merge_data(self):
        """Load and merge weather and agricultural data"""
        print("="*60)
        print("LOADING AND MERGING DATA")
        print("="*60)
        
        # Load weather data
        weather_file = RAW_DATA_DIR / "weather_data_latest.csv"
        if weather_file.exists():
            weather_df = pd.read_csv(weather_file)
            print(f"✓ Loaded weather data: {len(weather_df)} records")
        else:
            print("⚠️ No weather data found, creating synthetic data...")
            weather_df = self.create_synthetic_weather_data()
        
        # Load agricultural data
        ag_file = RAW_DATA_DIR / "agricultural_data_latest.csv"
        if ag_file.exists():
            ag_df = pd.read_csv(ag_file)
            print(f"✓ Loaded agricultural data: {len(ag_df)} records")
        else:
            print("❌ No agricultural data found! Run agriculture_collector.py first")
            return None
        
        # Merge datasets
        print("\nMerging datasets...")
        
        # Create state-level weather averages
        weather_summary = weather_df.groupby('state').agg({
            'temperature_c': 'mean',
            'humidity_percent': 'mean',
            'rainfall_mm': 'mean',
            'pressure_hpa': 'mean',
            'wind_speed_kmph': 'mean'
        }).reset_index()
        
        # Merge with agricultural data
        merged_df = ag_df.merge(weather_summary, left_on='state', right_on='state', how='left')
        
        # Fill missing weather data with defaults from agricultural data
        weather_cols = ['temperature_c', 'humidity_percent', 'rainfall_mm']
        for col in weather_cols:
            if col in merged_df.columns and col.replace('_c', '').replace('_percent', '').replace('_mm', '') in ag_df.columns:
                # Use agricultural data weather when API weather is missing
                api_col = col
                ag_col = col.replace('_c', '').replace('_percent', '').replace('_mm', '')
                if ag_col in ag_df.columns:
                    merged_df[api_col] = merged_df[api_col].fillna(merged_df[ag_col])
        
        # Fill remaining missing values
        for col in ['pressure_hpa', 'wind_speed_kmph']:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
        
        self.df = merged_df
        print(f"✓ Merged dataset created: {len(self.df)} records")
        print(f"✓ Columns: {list(self.df.columns)}")
        
        return self.df
    
    def create_synthetic_weather_data(self):
        """Create synthetic weather data for testing"""
        synthetic_weather = []
        
        for state in INDIAN_STATES[:10]:  # First 10 states
            weather_record = {
                'state': state,
                'temperature_c': np.random.uniform(20, 35),
                'humidity_percent': np.random.uniform(40, 80),
                'rainfall_mm': np.random.uniform(300, 1500),
                'pressure_hpa': np.random.uniform(1005, 1020),
                'wind_speed_kmph': np.random.uniform(5, 15)
            }
            synthetic_weather.append(weather_record)
        
        return pd.DataFrame(synthetic_weather)
    
    def feature_engineering(self):
        """Create additional features for better ML performance"""
        print("\nFEATURE ENGINEERING")
        print("-" * 40)
        
        # Time-based features
        if 'year' in self.df.columns:
            self.df['years_since_2000'] = self.df['year'] - 2000
        
        # Weather-based features
        if 'temperature_c' in self.df.columns and 'humidity_percent' in self.df.columns:
            self.df['heat_index'] = (self.df['temperature_c'] * self.df['humidity_percent']) / 100
        
        if 'rainfall_mm' in self.df.columns:
            self.df['drought_risk'] = (self.df['rainfall_mm'] < 400).astype(int)
            self.df['flood_risk'] = (self.df['rainfall_mm'] > 1500).astype(int)
            self.df['optimal_rainfall'] = ((self.df['rainfall_mm'] >= 600) & 
                                         (self.df['rainfall_mm'] <= 1200)).astype(int)
        
        # Productivity features
        if 'area_hectares' in self.df.columns:
            self.df['log_area'] = np.log1p(self.df['area_hectares'])
            
            # Area categories
            area_bins = [0, 10000, 50000, 100000, float('inf')]
            area_labels = ['Small', 'Medium', 'Large', 'Very_Large']
            self.df['area_category'] = pd.cut(self.df['area_hectares'], 
                                            bins=area_bins, labels=area_labels)
        
        # Crop-specific features
        if 'crop' in self.df.columns and 'yield_tonnes_per_hectare' in self.df.columns:
            crop_avg_yield = self.df.groupby('crop')['yield_tonnes_per_hectare'].transform('mean')
            self.df['yield_vs_crop_avg'] = self.df['yield_tonnes_per_hectare'] / crop_avg_yield
        
        # Season encoding
        if 'season' in self.df.columns:
            season_encoding = {'Kharif': 1, 'Rabi': 2, 'Summer': 3}
            self.df['season_code'] = self.df['season'].map(season_encoding).fillna(0)
        
        print("✓ Feature engineering completed")
        print(f"✓ Total columns now: {len(self.df.columns)}")
        
        return self.df
    
    def prepare_features(self):
        """Prepare features for ML models"""
        print("\nPREPARING FEATURES FOR ML")
        print("-" * 40)
        
        # Encode categorical variables
        categorical_columns = ['state', 'district', 'crop', 'season', 'area_category']
        
        for col in categorical_columns:
            if col in self.df.columns:
                # Handle missing values
                self.df[col] = self.df[col].fillna('Unknown')
                
                # Label encode
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                print(f"✓ Encoded {col}: {len(le.classes_)} unique values")
        
        # Select features for modeling
        potential_features = [
            'year', 'years_since_2000', 'season_code',
            'temperature_c', 'humidity_percent', 'rainfall_mm', 
            'pressure_hpa', 'wind_speed_kmph',
            'heat_index', 'drought_risk', 'flood_risk', 'optimal_rainfall',
            'log_area', 'area_hectares',
            'state_encoded', 'district_encoded', 'crop_encoded', 
            'season_encoded', 'area_category_encoded'
        ]
        
        # Only include features that exist in the dataset
        self.feature_names = [col for col in potential_features if col in self.df.columns]
        
        print(f"✓ Selected {len(self.feature_names)} features for modeling")
        print(f"✓ Features: {self.feature_names}")
        
        return self.feature_names
    
    def train_ml_models(self):
        """Train multiple ML models"""
        print("\nTRAINING ML MODELS")
        print("-" * 40)
        
        # Prepare data
        target_col = 'yield_tonnes_per_hectare'
        if target_col not in self.df.columns:
            print(f"❌ Target column '{target_col}' not found!")
            return None
        
        X = self.df[self.feature_names].fillna(0)
        y = self.df[target_col].fillna(y.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], y.median())
        
        print(f"✓ Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=MODEL_RANDOM_STATE
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_to_train = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=MODEL_RANDOM_STATE,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=MODEL_RANDOM_STATE
            ),
            'Linear Regression': LinearRegression()
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            try:
                # Choose appropriate data
                if name == 'Linear Regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                              cv=CV_FOLDS, scoring='r2')
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    cv_scores = cross_val_score(model, X_train, y_train, 
                                              cv=CV_FOLDS, scoring='r2')
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'predictions': y_pred,
                    'test_targets': y_test
                }
                
                print(f"✓ {name} Results:")
                print(f"   MAE: {mae:.3f}")
                print(f"   RMSE: {rmse:.3f}")
                print(f"   R²: {r2:.3f}")
                print(f"   CV R² (mean±std): {cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        return self.results
    
    def save_models(self):
        """Save the best model and preprocessing objects"""
        print("\nSAVING MODELS")
        print("-" * 40)
        
        if not self.results:
            print("❌ No models to save!")
            return
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_model = self.results[best_model_name]['model']
        
        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model and preprocessing objects
        model_files = {
            'best_model.pkl': best_model,
            'scaler.pkl': self.scaler,
            'encoders.pkl': self.encoders,
            'feature_names.pkl': self.feature_names
        }
        
        for filename, obj in model_files.items():
            filepath = MODELS_DIR / filename
            joblib.dump(obj, filepath)
            print(f"✓ Saved {filename}")
        
        # Save model performance summary
        summary = {
            'best_model': best_model_name,
            'training_date': datetime.now().isoformat(),
            'model_performance': {name: {k: v for k, v in results.items() 
                                       if k not in ['model', 'predictions', 'test_targets']} 
                                for name, results in self.results.items()},
            'feature_names': self.feature_names,
            'data_shape': self.df.shape
        }
        
        import json
        with open(MODELS_DIR / 'model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Best model: {best_model_name} (R² = {self.results[best_model_name]['r2']:.3f})")
        print(f"✓ Model summary saved to {MODELS_DIR / 'model_summary.json'}")
    
    def create_visualizations(self):
        """Create model performance visualizations"""
        print("\nCREATING VISUALIZATIONS")
        print("-" * 40)
        
        if not self.results:
            print("❌ No results to visualize!")
            return
        
        # Set up matplotlib
        plt.style.use('default')
        
        # 1. Model comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² comparison
        models = list(self.results.keys())
        r2_scores = [self.results[model]['r2'] for model in models]
        cv_scores = [self.results[model]['cv_r2_mean'] for model in models]
        
        x_pos = np.arange(len(models))
        axes[0, 0].bar(x_pos - 0.2, r2_scores, 0.4, label='Test R²', alpha=0.8)
        axes[0, 0].bar(x_pos + 0.2, cv_scores, 0.4, label='CV R²', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model R² Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE comparison
        mae_scores = [self.results[model]['mae'] for model in models]
        axes[0, 1].bar(models, mae_scores, alpha=0.8, color='orange')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_title('Model MAE Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predictions vs Actual (best model)
        best_model = max(models, key=lambda x: self.results[x]['r2'])
        predictions = self.results[best_model]['predictions']
        actuals = self.results[best_model]['test_targets']
        
        axes[1, 0].scatter(actuals, predictions, alpha=0.6)
        axes[1, 0].plot([actuals.min(), actuals.max()], 
                       [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Yield (tonnes/ha)')
        axes[1, 0].set_ylabel('Predicted Yield (tonnes/ha)')
        axes[1, 0].set_title(f'Predictions vs Actual ({best_model})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = actuals - predictions
        axes[1, 1].scatter(predictions, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Yield (tonnes/ha)')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = MODELS_DIR / 'model_performance.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✓ Model performance chart saved: {viz_file}")
        
        plt.show()
        
        # 2. Feature importance (for Random Forest)
        if 'Random Forest' in self.results:
            rf_model = self.results['Random Forest']['model']
            importance = rf_model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importance (Random Forest)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            # Save feature importance
            feat_file = MODELS_DIR / 'feature_importance.png'
            plt.savefig(feat_file, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance chart saved: {feat_file}")
            
            plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("="*60)
        print("AGRICULTURAL ML PIPELINE - COMPLETE RUN")
        print("="*60)
        
        # Step 1: Load and merge data
        if self.load_and_merge_data() is None:
            return None
        
        # Step 2: Feature engineering
        self.feature_engineering()
        
        # Step 3: Prepare features
        self.prepare_features()
        
        # Step 4: Train models
        results = self.train_ml_models()
        if results is None:
            return None
        
        # Step 5: Save models
        self.save_models()
        
        # Step 6: Create visualizations
        self.create_visualizations()
        
        # Step 7: Save processed data
        processed_file = PROCESSED_DATA_DIR / 'processed_agricultural_data.csv'
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(processed_file, index=False)
        print(f"✓ Processed data saved: {processed_file}")
        
        print("\n" + "="*60)
        print("ML PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Processed {len(self.df)} records")
        print(f"✓ Trained {len(results)} models")
        print(f"✓ Best model R²: {max([r['r2'] for r in results.values()]):.3f}")
        print(f"✓ Models saved to: {MODELS_DIR}")
        print("✓ Ready for dashboard deployment!")
        
        return results

def main():
    """Main function to run the ML pipeline"""
    pipeline = AgriculturalMLPipeline()
    results = pipeline.run_complete_pipeline()
    
    if results:
        print("\n🎉 ML Pipeline completed successfully!")
        print("Next step: Run the dashboard!")
    else:
        print("\n❌ ML Pipeline failed!")

if __name__ == "__main__":
    main()

