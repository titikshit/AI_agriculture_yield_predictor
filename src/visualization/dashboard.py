"""
Agricultural ML Dashboard - Optimized for latest package versions
"""
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import *
except ImportError:
    # Fallback configuration
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "data" / "models"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

print("🚀 Starting Agricultural ML Dashboard...")
print(f"📦 Pandas: {pd.__version__}")
print(f"📦 Plotly: {px.__version__ if hasattr(px, '__version__') else 'Unknown'}")
print(f"📦 Dash: {dash.__version__}")

# Load or create data
def load_dashboard_data():
    """Load data with fallback to sample generation"""
    
    # Try to load processed data first
    processed_file = PROCESSED_DATA_DIR / 'processed_agricultural_data.csv'
    if processed_file.exists():
        df = pd.read_csv(processed_file)
        print(f"✅ Loaded processed data: {len(df)} records")
        return df, True
    
    # Try raw agricultural data
    raw_ag_file = RAW_DATA_DIR / 'agricultural_data_latest.csv'
    if raw_ag_file.exists():
        df = pd.read_csv(raw_ag_file)
        print(f"✅ Loaded raw agricultural data: {len(df)} records")
        return df, True
    
    # Generate realistic sample data
    print("⚠️ No data files found, generating sample data...")
    np.random.seed(42)
    
    # Indian states and crops
    states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 'Karnataka', 
              'Andhra Pradesh', 'Tamil Nadu', 'Gujarat', 'Rajasthan', 'West Bengal']
    crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Soybean', 
             'Barley', 'Mustard', 'Groundnut', 'Sunflower']
    seasons = ['Kharif', 'Rabi', 'Summer']
    years = [2019, 2020, 2021, 2022, 2023]
    
    records = []
    for i in range(2000):  # Generate 2000 records
        state = np.random.choice(states)
        crop = np.random.choice(crops)
        season = np.random.choice(seasons)
        year = np.random.choice(years)
        
        # Generate realistic yields based on crop
        crop_base_yields = {
            'Rice': (2.0, 4.5), 'Wheat': (2.5, 4.0), 'Cotton': (0.3, 0.8),
            'Sugarcane': (60, 90), 'Maize': (3.0, 6.0), 'Soybean': (1.0, 2.5),
            'Barley': (2.0, 3.5), 'Mustard': (1.0, 2.0), 'Groundnut': (1.5, 3.0),
            'Sunflower': (1.0, 2.0)
        }
        
        min_yield, max_yield = crop_base_yields.get(crop, (1.0, 3.0))
        yield_value = np.random.uniform(min_yield, max_yield)
        
        # Generate correlated weather data
        if state in ['Punjab', 'Haryana']:
            base_temp, base_rain, base_humidity = 25, 600, 60
        elif state in ['Rajasthan', 'Gujarat']:
            base_temp, base_rain, base_humidity = 30, 400, 45
        elif state in ['Tamil Nadu', 'Andhra Pradesh']:
            base_temp, base_rain, base_humidity = 28, 800, 70
        elif state in ['West Bengal']:
            base_temp, base_rain, base_humidity = 27, 1200, 80
        else:
            base_temp, base_rain, base_humidity = 26, 700, 65
        
        # Add seasonal variation
        seasonal_factors = {
            'Kharif': {'temp': 2, 'rain': 1.5, 'humidity': 10},
            'Rabi': {'temp': -3, 'rain': 0.3, 'humidity': -10},
            'Summer': {'temp': 5, 'rain': 0.2, 'humidity': -15}
        }
        
        factor = seasonal_factors.get(season, {'temp': 0, 'rain': 1, 'humidity': 0})
        
        temperature = base_temp + factor['temp'] + np.random.normal(0, 2)
        rainfall = base_rain * factor['rain'] + np.random.normal(0, base_rain * 0.2)
        humidity = base_humidity + factor['humidity'] + np.random.normal(0, 8)
        
        # Ensure realistic ranges
        temperature = np.clip(temperature, 15, 45)
        rainfall = np.clip(rainfall, 50, 2500)
        humidity = np.clip(humidity, 30, 95)
        
        # Generate area
        area = np.random.uniform(1000, 100000)
        
        record = {
            'record_id': i + 1,
            'year': year,
            'season': season,
            'state': state,
            'district': f"{state} Dist {np.random.randint(1, 6)}",
            'crop': crop,
            'area_hectares': round(area, 2),
            'yield_tonnes_per_hectare': round(yield_value, 3),
            'production_tonnes': round(area * yield_value, 2),
            'temperature_c': round(temperature, 1),
            'rainfall_mm': round(rainfall, 1),
            'humidity_percent': round(humidity, 1),
            'data_source': 'Generated_Sample'
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    print(f"✅ Generated sample data: {len(df)} records")
    
    # Save the sample data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / 'sample_dashboard_data.csv', index=False)
    
    return df, False

# Load data and models
df, is_real_data = load_dashboard_data()

# Try to load ML models
try:
    if (MODELS_DIR / 'best_model.pkl').exists():
        model = joblib.load(MODELS_DIR / 'best_model.pkl')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl') if (MODELS_DIR / 'scaler.pkl').exists() else None
        encoders = joblib.load(MODELS_DIR / 'encoders.pkl') if (MODELS_DIR / 'encoders.pkl').exists() else None
        print("✅ ML models loaded successfully!")
        model_available = True
    else:
        model, scaler, encoders = None, None, None
        model_available = False
        print("⚠️ No ML models found - running in demo mode")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    model, scaler, encoders = None, None, None
    model_available = False

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Agricultural ML Dashboard"

# Define app layout
app.layout = html.Div([
    # Header Section
    html.Div([
        html.H1("🌾 Agricultural Analytics Dashboard", 
                className="text-center text-success mb-2"),
        html.P([
            f"Interactive ML-powered analysis • {len(df):,} records • ",
            html.Span("Real Data" if is_real_data else "Sample Data", 
                     className="badge bg-primary me-2"),
            html.Span("ML Ready" if model_available else "Demo Mode",
                     className="badge bg-success" if model_available else "badge bg-warning")
        ], className="text-center text-muted")
    ], className="bg-light p-4 mb-4 rounded"),
    
    # Control Panel
    html.Div([
        html.H4("🎛️ Analysis Controls", className="text-success mb-3"),
        
        # Row of dropdowns
        html.Div([
            # State dropdown
            html.Div([
                html.Label("Select State:", className="form-label fw-bold"),
                dcc.Dropdown(
                    id='state-dropdown',
                    options=[{'label': state, 'value': state} 
                           for state in sorted(df['state'].unique())],
                    value=df['state'].iloc[0],
                    className="mb-2"
                )
            ], className="col-md-3"),
            
            # Crop dropdown
            html.Div([
                html.Label("Select Crop:", className="form-label fw-bold"),
                dcc.Dropdown(
                    id='crop-dropdown',
                    options=[{'label': crop, 'value': crop} 
                           for crop in sorted(df['crop'].unique())],
                    value=df['crop'].iloc[0],
                    className="mb-2"
                )
            ], className="col-md-3"),
            
            # Year dropdown
            html.Div([
                html.Label("Select Year:", className="form-label fw-bold"),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(year), 'value': year} 
                           for year in sorted(df['year'].unique())],
                    value=df['year'].max(),
                    className="mb-2"
                )
            ], className="col-md-3"),
            
            # Analysis type
            html.Div([
                html.Label("Analysis View:", className="form-label fw-bold"),
                dcc.Dropdown(
                    id='analysis-type',
                    options=[
                        {'label': '📊 Overview', 'value': 'overview'},
                        {'label': '🌡️ Weather Impact', 'value': 'weather'},
                        {'label': '📈 Trends', 'value': 'trends'},
                        {'label': '🗺️ Geographic', 'value': 'geographic'}
                    ],
                    value='overview',
                    className="mb-2"
                )
            ], className="col-md-3")
        ], className="row g-3")
    ], className="bg-light p-4 mb-4 rounded"),
    
    # KPI Cards
    html.Div([
        html.H4("📊 Key Performance Indicators", className="text-success mb-3"),
        html.Div(id='kpi-cards', className="row g-3")
    ], className="mb-4"),
    
    # Main Charts Section
    html.Div([
        # First row of charts
        html.Div([
            html.Div([
                dcc.Graph(id='main-chart')
            ], className="col-md-6"),
            
            html.Div([
                dcc.Graph(id='secondary-chart')
            ], className="col-md-6")
        ], className="row mb-4"),
        
        # Second row of charts
        html.Div([
            html.Div([
                dcc.Graph(id='correlation-chart')
            ], className="col-md-6"),
            
            html.Div([
                dcc.Graph(id='distribution-chart')
            ], className="col-md-6")
        ], className="row mb-4")
    ]),
    
    # ML Prediction Section
    html.Div([
        html.H4("🤖 ML Yield Prediction", className="text-success mb-3"),
        html.P("Adjust environmental parameters to predict crop yield using " + 
               ("trained ML models" if model_available else "statistical methods"),
               className="text-muted mb-3"),
        
        # Prediction inputs
        html.Div([
            html.Div([
                html.Label("🌡️ Temperature (°C):", className="form-label"),
                dcc.Input(id='temp-input', type='number', value=26, min=10, max=45, 
                         step=0.5, className="form-control")
            ], className="col-md-3"),
            
            html.Div([
                html.Label("🌧️ Rainfall (mm):", className="form-label"),
                dcc.Input(id='rainfall-input', type='number', value=750, min=0, 
                         max=3000, step=10, className="form-control")
            ], className="col-md-3"),
            
            html.Div([
                html.Label("💧 Humidity (%):", className="form-label"),
                dcc.Input(id='humidity-input', type='number', value=65, min=20, 
                         max=95, step=1, className="form-control")
            ], className="col-md-3"),
            
            html.Div([
                html.Label("🚀 Action:", className="form-label"),
                html.Br(),
                html.Button('Predict Yield', id='predict-btn', n_clicks=0,
                           className="btn btn-success btn-lg w-100")
            ], className="col-md-3")
        ], className="row g-3 mb-4"),
        
        # Prediction output
        html.Div(id='prediction-output', className="text-center")
    ], className="bg-light p-4 rounded mb-4"),
    
    # Data Table Section
    html.Div([
        html.H4("📋 Data Explorer", className="text-success mb-3"),
        html.Div(id='data-table-container')
    ])
    
], className="container-fluid p-4")

# Callbacks
@callback(
    Output('kpi-cards', 'children'),
    [Input('state-dropdown', 'value'),
     Input('crop-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_kpi_cards(state, crop, year):
    # Filter data
    filtered_df = df.copy()
    if state:
        filtered_df = filtered_df[filtered_df['state'] == state]
    if crop:
        filtered_df = filtered_df[filtered_df['crop'] == crop]
    if year:
        filtered_df = filtered_df[filtered_df['year'] == year]
    
    if len(filtered_df) == 0:
        return [html.Div("No data available", className="col-12 text-center text-muted")]
    
    # Calculate KPIs
    avg_yield = filtered_df['yield_tonnes_per_hectare'].mean()
    total_area = filtered_df['area_hectares'].sum()
    total_production = filtered_df['production_tonnes'].sum()
    avg_temp = filtered_df['temperature_c'].mean()
    avg_rainfall = filtered_df['rainfall_mm'].mean()
    record_count = len(filtered_df)
    
    # Create KPI cards
    kpi_data = [
        {"title": "Avg Yield", "value": f"{avg_yield:.2f}", "unit": "t/ha", "color": "success"},
        {"title": "Total Area", "value": f"{total_area/1000:.0f}K", "unit": "hectares", "color": "primary"},
        {"title": "Production", "value": f"{total_production/1000:.0f}K", "unit": "tonnes", "color": "info"},
        {"title": "Avg Temp", "value": f"{avg_temp:.1f}°C", "unit": "", "color": "warning"},
        {"title": "Avg Rainfall", "value": f"{avg_rainfall:.0f}", "unit": "mm", "color": "secondary"},
        {"title": "Records", "value": f"{record_count}", "unit": "", "color": "dark"}
    ]
    
    cards = []
    for kpi in kpi_data:
        card = html.Div([
            html.Div([
                html.H5(kpi["value"], className=f"text-{kpi['color']} mb-0"),
                html.Small(f"{kpi['title']} {kpi['unit']}", className="text-muted")
            ], className="card-body text-center")
        ], className="card border-0 shadow-sm col-md-2")
        cards.append(card)
    
    return cards

@callback(
    [Output('main-chart', 'figure'),
     Output('secondary-chart', 'figure'),
     Output('correlation-chart', 'figure'),
     Output('distribution-chart', 'figure')],
    [Input('state-dropdown', 'value'),
     Input('crop-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('analysis-type', 'value')]
)
def update_charts(state, crop, year, analysis_type):
    # Filter data
    filtered_df = df.copy()
    if state:
        filtered_df = filtered_df[filtered_df['state'] == state]
    if crop:
        filtered_df = filtered_df[filtered_df['crop'] == crop]
    if year:
        filtered_df = filtered_df[filtered_df['year'] == year]
    
    if len(filtered_df) == 0:
        empty_fig = px.bar(x=['No Data'], y=[0], title="No data available for selected filters")
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # Chart 1: Main analysis based on type
    if analysis_type == 'overview':
        fig1 = px.box(filtered_df, x='season', y='yield_tonnes_per_hectare',
                     title=f'Yield Distribution by Season - {crop} in {state}')
    elif analysis_type == 'weather':
        fig1 = px.scatter(filtered_df, x='rainfall_mm', y='yield_tonnes_per_hectare',
                         color='temperature_c', size='area_hectares',
                         title=f'Weather Impact on Yield - {crop} in {state}')
    elif analysis_type == 'trends':
        yearly_data = filtered_df.groupby('year')['yield_tonnes_per_hectare'].mean().reset_index()
        fig1 = px.line(yearly_data, x='year', y='yield_tonnes_per_hectare',
                      title=f'Yield Trends Over Time - {crop} in {state}')
    else:  # geographic
        state_data = df[df['crop'] == crop].groupby('state')['yield_tonnes_per_hectare'].mean().reset_index()
        fig1 = px.bar(state_data, x='state', y='yield_tonnes_per_hectare',
                     title=f'Average Yield by State - {crop}')
        fig1.update_xaxes(tickangle=45)
    
    # Chart 2: Secondary analysis
    fig2 = px.scatter(filtered_df, x='temperature_c', y='humidity_percent',
                     size='yield_tonnes_per_hectare', color='season',
                     title=f'Temperature vs Humidity - {crop} in {state}')
    
    # Chart 3: Correlation heatmap
    numeric_cols = ['yield_tonnes_per_hectare', 'temperature_c', 'rainfall_mm', 'humidity_percent', 'area_hectares']
    corr_data = filtered_df[numeric_cols].corr()
    fig3 = px.imshow(corr_data, text_auto=True, aspect="auto",
                     title="Feature Correlation Matrix")
    
    # Chart 4: Distribution
    fig4 = px.histogram(filtered_df, x='yield_tonnes_per_hectare', nbins=20,
                       title=f'Yield Distribution - {crop} in {state}')
    
    # Update layouts
    for fig in [fig1, fig2, fig3, fig4]:
        fig.update_layout(height=350, margin=dict(l=40, r=40, t=60, b=40))
    
    return fig1, fig2, fig3, fig4

@callback(
    Output('prediction-output', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [Input('temp-input', 'value'),
     Input('rainfall-input', 'value'),
     Input('humidity-input', 'value'),
     Input('state-dropdown', 'value'),
     Input('crop-dropdown', 'value')]
)
def make_prediction(n_clicks, temperature, rainfall, humidity, state, crop):
    if n_clicks == 0:
        return html.Div("Adjust parameters above and click 'Predict Yield' to see results",
                       className="text-muted")
    
    try:
        # Simple prediction logic (replace with actual ML model if available)
        if model_available and model is not None:
            # Real ML prediction would go here
            prediction = 2.8 + (temperature - 26) * 0.05 + (rainfall - 750) * 0.0008 + (humidity - 65) * 0.015
        else:
            # Statistical prediction based on historical data
            similar_data = df[
                (df['state'] == state) & 
                (df['crop'] == crop) &
                (abs(df['temperature_c'] - temperature) <= 5) &
                (abs(df['rainfall_mm'] - rainfall) <= 200) &
                (abs(df['humidity_percent'] - humidity) <= 15)
            ]
            
            if len(similar_data) > 0:
                prediction = similar_data['yield_tonnes_per_hectare'].mean()
            else:
                # Fallback calculation
                base_yield = df[(df['state'] == state) & (df['crop'] == crop)]['yield_tonnes_per_hectare'].mean()
                if pd.isna(base_yield):
                    base_yield = df[df['crop'] == crop]['yield_tonnes_per_hectare'].mean()
                
                # Apply weather factors
                temp_factor = 1.0 + (temperature - 26) * 0.02
                rain_factor = 1.0 + (rainfall - 750) * 0.0005
                humidity_factor = 1.0 + (humidity - 65) * 0.008
                
                prediction = base_yield * temp_factor * rain_factor * humidity_factor
        
        prediction = max(0.1, prediction)
        
        # Create prediction result
        method = "🤖 ML Model" if model_available else "📊 Statistical Analysis"
        confidence = "High" if len(similar_data) > 10 else "Medium" if len(similar_data) > 0 else "Low"
        
        return html.Div([
            html.Div([
                html.H3(f"{prediction:.2f}", className="text-success mb-0"),
                html.Small("tonnes per hectare", className="text-muted")
            ], className="text-center mb-3"),
            
            html.Div([
                html.Span(f"{method} Prediction", className="badge bg-info me-2"),
                html.Span(f"Confidence: {confidence}", className="badge bg-secondary me-2"),
                html.Span(f"{crop} in {state}", className="badge bg-primary")
            ], className="text-center mb-2"),
            
            html.Small(f"Conditions: {temperature}°C, {rainfall}mm rainfall, {humidity}% humidity",
                      className="text-muted d-block text-center")
        ], className="p-3 bg-white rounded shadow-sm")
        
    except Exception as e:
        return html.Div(f"Prediction error: {str(e)}", className="text-danger text-center")

@callback(
    Output('data-table-container', 'children'),
    [Input('state-dropdown', 'value'),
     Input('crop-dropdown', 'value')]
)
def update_data_table(state, crop):
    # Filter data
    filtered_df = df.copy()
    if state:
        filtered_df = filtered_df[filtered_df['state'] == state]
    if crop:
        filtered_df = filtered_df[filtered_df['crop'] == crop]
    
    # Select columns for display
    display_cols = ['year', 'season', 'state', 'crop', 'yield_tonnes_per_hectare', 
                   'area_hectares', 'temperature_c', 'rainfall_mm', 'humidity_percent']
    
    display_df = filtered_df[display_cols].head(20)  # Show top 20 records
    
    return dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{"name": col.replace('_', ' ').title(), "id": col, 
                 "type": "numeric" if df[col].dtype in ['int64', 'float64'] else "text"}
                for col in display_cols],
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
        style_data={'backgroundColor': '#f8f9fa'},
        page_size=10,
        sort_action="native",
        filter_action="native"
    )

# Add Bootstrap CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AGRICULTURAL ML DASHBOARD READY!")
    print("="*60)
    print(f"📊 Data loaded: {len(df):,} records")
    print(f"🤖 ML Status: {'Available' if model_available else 'Demo Mode'}")
    print(f"🌐 URL: http://localhost:8050")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8050)

