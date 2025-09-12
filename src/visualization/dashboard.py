"""
Agricultural ML Dashboard
"""
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

# Load data and models
try:
    df = pd.read_csv(PROCESSED_DATA_DIR / 'processed_agricultural_data.csv')
    model = joblib.load(MODELS_DIR / 'best_model.pkl')
    print("✓ Dashboard data and models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dashboard data: {e}")
    print("Please run the ML pipeline first!")
    exit(1)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🌾 Agricultural ML Dashboard", 
            style={'textAlign': 'center', 'color': '#2E8B57'}),
    
    html.Div([
        html.Label("Select State:"),
        dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': state, 'value': state} for state in df['state'].unique()],
            value=df['state'].iloc[0]
        )
    ], style={'width': '300px', 'margin': '20px'}),
    
    dcc.Graph(id='yield-chart'),
    
    html.Div(id='stats-output', style={'margin': '20px'})
])

@app.callback(
    [Output('yield-chart', 'figure'),
     Output('stats-output', 'children')],
    [Input('state-dropdown', 'value')]
)
def update_dashboard(selected_state):
    filtered_df = df[df['state'] == selected_state]
    
    # Create chart
    fig = px.box(filtered_df, x='crop', y='yield_tonnes_per_hectare',
                 title=f'Crop Yield Distribution - {selected_state}')
    
    # Create stats
    avg_yield = filtered_df['yield_tonnes_per_hectare'].mean()
    total_area = filtered_df['area_hectares'].sum()
    
    stats = html.Div([
        html.H3(f"Statistics for {selected_state}"),
        html.P(f"Average Yield: {avg_yield:.2f} tonnes/hectare"),
        html.P(f"Total Area: {total_area:,.0f} hectares")
    ])
    
    return fig, stats

if __name__ == '__main__':
    print("Starting dashboard server...")
    print("Visit: http://localhost:8050")
    app.run_server(debug=True)

