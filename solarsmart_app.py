# SolarSmart - AI-Driven Solar Panel Performance Forecasting Tool
# Updated for Python 3.13 compatibility

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import requests
import json
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SolarSmart - AI Solar Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class WeatherAPI:
    """Mock Weather API for demonstration"""
    @staticmethod
    def get_weather_data(location="New York", days=7):
        # Simulating weather API response
        dates = [datetime.datetime.now() + timedelta(days=i) for i in range(days)]
        weather_data = []
        
        for date in dates:
            weather_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature': np.random.normal(25, 5),  # Celsius
                'humidity': np.random.normal(60, 15),    # %
                'irradiance': np.random.normal(800, 200), # W/m²
                'wind_speed': np.random.normal(10, 5),   # km/h
                'cloud_cover': np.random.normal(30, 20)  # %
            })
        
        return pd.DataFrame(weather_data)

class SolarDataGenerator:
    """Generate synthetic solar panel data for demonstration"""
    
    @staticmethod
    def generate_historical_data(days=365):
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        data = []
        for i, date in enumerate(dates):
            # Seasonal patterns
            season_factor = 0.8 + 0.4 * np.sin(2 * np.pi * i / 365)
            
            # Daily patterns (peak at noon)
            for hour in range(24):
                hour_factor = max(0, np.sin(np.pi * (hour - 6) / 12))
                
                base_irradiance = 1000 * season_factor * hour_factor
                noise = np.random.normal(0, 50)
                irradiance = max(0, base_irradiance + noise)
                
                # Panel efficiency (degradation over time)
                panel_efficiency = 0.2 - (i * 0.0001)  # Slight degradation
                
                energy_output = irradiance * panel_efficiency * 10  # 10 m² panel
                
                data.append({
                    'datetime': date + pd.Timedelta(hours=hour),
                    'irradiance': irradiance,
                    'temperature': 20 + 15 * hour_factor + np.random.normal(0, 3),
                    'humidity': 50 + np.random.normal(0, 10),
                    'energy_output': max(0, energy_output),
                    'panel_voltage': 24 + np.random.normal(0, 1),
                    'panel_current': max(0, energy_output / 24 + np.random.normal(0, 0.5)),
                    'panel_id': f'Panel_0{i%5 + 1}'
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_anomaly_data():
        """Generate data with some anomalies for testing"""
        normal_data = SolarDataGenerator.generate_historical_data(30)
        
        # Introduce some anomalies
        anomaly_indices = np.random.choice(len(normal_data), size=20, replace=False)
        
        for idx in anomaly_indices:
            # Simulate panel issues
            normal_data.loc[idx, 'energy_output'] *= 0.3  # Significant drop
            normal_data.loc[idx, 'panel_current'] *= 0.3
        
        return normal_data

class SimpleSolarPredictor:
    """Simplified solar energy prediction using Random Forest (no TensorFlow)"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, data):
        """Prepare data for training"""
        features = ['irradiance', 'temperature', 'humidity']
        target = 'energy_output'
        
        # Scale features
        X = self.scaler.fit_transform(data[features])
        y = data[target].values
        
        return X, y
    
    def train(self, data):
        """Train the Random Forest model"""
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {'mae': mae, 'r2': r2}, X_test, y_test
    
    def predict(self, weather_data):
        """Make predictions based on weather forecast"""
        if self.model is None:
            return None
        
        features = ['irradiance', 'temperature', 'humidity']
        X = self.scaler.transform(weather_data[features])
        
        predictions = self.model.predict(X)
        return predictions

class AnomalyDetector:
    """Detect anomalies in solar panel performance"""
    
    def __init__(self):
        self.detector = IsolationForest(contamination=0.1, random_state=42)
        
    def detect_anomalies(self, data):
        """Detect anomalies in solar panel data"""
        features = ['energy_output', 'panel_voltage', 'panel_current', 'temperature']
        
        # Fit the detector
        self.detector.fit(data[features])
        
        # Predict anomalies
        anomalies = self.detector.predict(data[features])
        anomaly_scores = self.detector.score_samples(data[features])
        
        data_copy = data.copy()
        data_copy['anomaly'] = anomalies
        data_copy['anomaly_score'] = anomaly_scores
        
        return data_copy

def main():
    # Header
    st.markdown('<h1 class="main-header">☀️ SolarSmart - AI Solar Forecasting</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Dashboard",
        "📈 Performance Forecasting",
        "🔍 Efficiency Analyzer",
        "🎯 Scenario Simulator",
        "📊 Data Upload"
    ])
    
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📈 Performance Forecasting":
        forecasting_page()
    elif page == "🔍 Efficiency Analyzer":
        efficiency_page()
    elif page == "🎯 Scenario Simulator":
        simulator_page()
    elif page == "📊 Data Upload":
        data_upload_page()

def dashboard_page():
    st.header("Solar Performance Dashboard")
    
    # Generate sample data
    data = SolarDataGenerator.generate_historical_data(30)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_energy = data['energy_output'].sum()
        st.metric("Total Energy (kWh)", f"{total_energy:.1f}", delta="12.5%")
    
    with col2:
        avg_efficiency = data['energy_output'].mean()
        st.metric("Avg Daily Output", f"{avg_efficiency:.1f} kWh", delta="5.2%")
    
    with col3:
        max_output = data['energy_output'].max()
        st.metric("Peak Output", f"{max_output:.1f} kWh", delta="8.1%")
    
    with col4:
        uptime = 98.5
        st.metric("System Uptime", f"{uptime:.1f}%", delta="0.5%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Output Over Time")
        daily_output = data.groupby(data['datetime'].dt.date)['energy_output'].sum().reset_index()
        fig = px.line(daily_output, x='datetime', y='energy_output', 
                     title="Daily Energy Production")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Panel Performance Heatmap")
        panel_performance = data.groupby(['panel_id', data['datetime'].dt.date])['energy_output'].sum().reset_index()
        pivot_data = panel_performance.pivot(index='panel_id', columns='datetime', values='energy_output')
        fig = px.imshow(pivot_data, aspect="auto", title="Panel Performance Heatmap")
        st.plotly_chart(fig, use_container_width=True)

def forecasting_page():
    st.header("Solar Energy Forecasting")
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.text_input("Location", value="New York")
        forecast_days = st.slider("Forecast Days", 1, 14, 7)
    
    with col2:
        panel_capacity = st.number_input("Panel Capacity (kW)", value=10.0, step=0.5)
        panel_efficiency = st.slider("Panel Efficiency (%)", 15, 25, 20)
    
    if st.button("Generate Forecast"):
        # Get weather data
        weather_data = WeatherAPI.get_weather_data(location, forecast_days)
        
        # Generate historical data for training
        historical_data = SolarDataGenerator.generate_historical_data(100)
        
        # Train predictor
        predictor = SimpleSolarPredictor()
        with st.spinner("Training prediction model..."):
            metrics, X_test, y_test = predictor.train(historical_data)
        
        st.success(f"Model trained! R² Score: {metrics['r2']:.3f}, MAE: {metrics['mae']:.2f}")
        
        # Make predictions
        predictions = predictor.predict(weather_data)
        weather_data['predicted_output'] = predictions
        
        # Adjust predictions based on panel specifications
        weather_data['predicted_output'] = (weather_data['predicted_output'] * 
                                          panel_capacity / 10 * panel_efficiency / 20)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Weather Forecast")
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=['Temperature', 'Irradiance', 'Humidity', 'Cloud Cover'])
            
            fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['temperature'],
                                   name='Temperature'), row=1, col=1)
            fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['irradiance'],
                                   name='Irradiance'), row=1, col=2)
            fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['humidity'],
                                   name='Humidity'), row=2, col=1)
            fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['cloud_cover'],
                                   name='Cloud Cover'), row=2, col=2)
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Energy Output Forecast")
            fig = px.line(weather_data, x='date', y='predicted_output',
                         title=f"Predicted Energy Output - {location}")
            fig.update_layout(yaxis_title="Energy Output (kWh)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            total_predicted = weather_data['predicted_output'].sum()
            avg_daily = weather_data['predicted_output'].mean()
            
            st.metric("Total Forecast Energy", f"{total_predicted:.1f} kWh")
            st.metric("Average Daily Output", f"{avg_daily:.1f} kWh")

def efficiency_page():
    st.header("Panel Efficiency Analyzer")
    
    # Generate sample data with anomalies
    data = SolarDataGenerator.generate_anomaly_data()
    
    if st.button("Analyze Panel Performance"):
        # Detect anomalies
        detector = AnomalyDetector()
        analyzed_data = detector.detect_anomalies(data)
        
        # Display results
        anomalies = analyzed_data[analyzed_data['anomaly'] == -1]
        normal_data = analyzed_data[analyzed_data['anomaly'] == 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Anomaly Detection Results")
            st.metric("Total Data Points", len(analyzed_data))
            st.metric("Anomalies Detected", len(anomalies))
            st.metric("Normal Operations", len(normal_data))
            
            # Anomaly timeline
            fig = px.scatter(analyzed_data, x='datetime', y='energy_output',
                           color='anomaly', title="Energy Output with Anomalies")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Panel Performance Analysis")
            
            # Panel comparison
            panel_stats = analyzed_data.groupby('panel_id').agg({
                'energy_output': ['mean', 'std'],
                'anomaly': lambda x: (x == -1).sum()
            }).round(2)
            
            panel_stats.columns = ['Avg Output', 'Std Dev', 'Anomalies']
            st.dataframe(panel_stats)
            
            # Efficiency heatmap
            fig = px.box(analyzed_data, x='panel_id', y='energy_output',
                        title="Panel Performance Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed anomaly report
        if not anomalies.empty:
            st.subheader("Anomaly Report")
            
            anomaly_summary = anomalies.groupby('panel_id').agg({
                'datetime': 'count',
                'energy_output': 'mean',
                'anomaly_score': 'mean'
            }).round(2)
            
            anomaly_summary.columns = ['Anomaly Count', 'Avg Output', 'Avg Anomaly Score']
            st.dataframe(anomaly_summary)
            
            # Recommendations
            st.subheader("Recommendations")
            for panel_id in anomaly_summary.index:
                count = anomaly_summary.loc[panel_id, 'Anomaly Count']
                if count > 5:
                    st.warning(f"⚠️ {panel_id}: High anomaly count ({count}). Consider maintenance.")
                elif count > 2:
                    st.info(f"ℹ️ {panel_id}: Moderate anomalies detected ({count}). Monitor closely.")

def simulator_page():
    st.header("Solar Configuration Simulator")
    
    st.write("Simulate different solar panel configurations to optimize performance.")
    
    # Configuration inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Panel Configuration")
        num_panels = st.slider("Number of Panels", 1, 50, 20)
        panel_wattage = st.slider("Panel Wattage (W)", 250, 500, 400)
        tilt_angle = st.slider("Tilt Angle (degrees)", 0, 60, 30)
    
    with col2:
        st.subheader("Location Settings")
        latitude = st.slider("Latitude", -90.0, 90.0, 40.0)
        azimuth = st.slider("Azimuth (degrees)", 0, 360, 180)
        shading_factor = st.slider("Shading Factor (%)", 0, 50, 10)
    
    with col3:
        st.subheader("Maintenance Settings")
        cleaning_frequency = st.selectbox("Cleaning Frequency", 
                                        ["Weekly", "Monthly", "Quarterly", "Annually"])
        degradation_rate = st.slider("Annual Degradation (%)", 0.3, 1.0, 0.5)
    
    if st.button("Run Simulation"):
        # Simulate different configurations
        configurations = []
        
        # Base configuration
        base_output = simulate_solar_output(num_panels, panel_wattage, tilt_angle, 
                                          latitude, azimuth, shading_factor, 
                                          cleaning_frequency, degradation_rate)
        
        configurations.append({
            'Configuration': 'Current',
            'Annual Output (kWh)': base_output,
            'Panels': num_panels,
            'Tilt': tilt_angle,
            'Efficiency': base_output / (num_panels * panel_wattage * 8760 / 1000) * 100
        })
        
        # Optimized configurations
        for tilt in [20, 35, 45]:
            if tilt != tilt_angle:
                output = simulate_solar_output(num_panels, panel_wattage, tilt, 
                                             latitude, azimuth, shading_factor, 
                                             cleaning_frequency, degradation_rate)
                configurations.append({
                    'Configuration': f'Tilt {tilt}°',
                    'Annual Output (kWh)': output,
                    'Panels': num_panels,
                    'Tilt': tilt,
                    'Efficiency': output / (num_panels * panel_wattage * 8760 / 1000) * 100
                })
        
        # More/fewer panels
        for panels in [num_panels - 5, num_panels + 5]:
            if panels > 0:
                output = simulate_solar_output(panels, panel_wattage, tilt_angle, 
                                             latitude, azimuth, shading_factor, 
                                             cleaning_frequency, degradation_rate)
                configurations.append({
                    'Configuration': f'{panels} Panels',
                    'Annual Output (kWh)': output,
                    'Panels': panels,
                    'Tilt': tilt_angle,
                    'Efficiency': output / (panels * panel_wattage * 8760 / 1000) * 100
                })
        
        # Display results
        df_configs = pd.DataFrame(configurations)
        df_configs = df_configs.round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration Comparison")
            st.dataframe(df_configs)
            
            # Best configuration
            best_config = df_configs.loc[df_configs['Annual Output (kWh)'].idxmax()]
            st.success(f"🏆 Best Configuration: {best_config['Configuration']} "
                      f"({best_config['Annual Output (kWh)']} kWh/year)")
        
        with col2:
            st.subheader("Performance Comparison")
            fig = px.bar(df_configs, x='Configuration', y='Annual Output (kWh)',
                        title="Annual Energy Output by Configuration")
            st.plotly_chart(fig, use_container_width=True)
            
            # ROI Analysis
            st.subheader("ROI Analysis")
            electricity_rate = st.number_input("Electricity Rate ($/kWh)", value=0.12, step=0.01)
            
            df_configs['Annual Savings ($)'] = df_configs['Annual Output (kWh)'] * electricity_rate
            df_configs['20-Year Savings ($)'] = df_configs['Annual Savings ($)'] * 20
            
            fig = px.bar(df_configs, x='Configuration', y='20-Year Savings ($)',
                        title="20-Year Financial Savings by Configuration")
            st.plotly_chart(fig, use_container_width=True)

def simulate_solar_output(num_panels, panel_wattage, tilt_angle, latitude, 
                         azimuth, shading_factor, cleaning_frequency, degradation_rate):
    """Simulate solar output based on configuration parameters"""
    
    # Base calculation
    base_output_per_panel = panel_wattage * 4.5  # 4.5 peak sun hours average
    
    # Tilt angle optimization (simplified)
    tilt_efficiency = 1 - abs(tilt_angle - latitude) * 0.01
    
    # Azimuth efficiency (180° is optimal in Northern Hemisphere)
    azimuth_efficiency = 1 - abs(azimuth - 180) * 0.002
    
    # Shading losses
    shading_efficiency = 1 - (shading_factor / 100)
    
    # Cleaning frequency impact
    cleaning_efficiency = {
        'Weekly': 0.98,
        'Monthly': 0.95,
        'Quarterly': 0.90,
        'Annually': 0.85
    }[cleaning_frequency]
    
    # Annual degradation
    annual_efficiency = 1 - (degradation_rate / 100)
    
    # Calculate total annual output
    total_efficiency = (tilt_efficiency * azimuth_efficiency * 
                       shading_efficiency * cleaning_efficiency * annual_efficiency)
    
    annual_output = (base_output_per_panel * num_panels * 365 * total_efficiency) / 1000
    
    return annual_output

def data_upload_page():
    st.header("Data Upload & Analysis")
    
    st.write("Upload your solar panel data for custom analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Data Summary")
            st.write(df.describe())
            
            # Column selection for analysis
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("Select X-axis", numeric_columns)
                
                with col2:
                    y_axis = st.selectbox("Select Y-axis", numeric_columns)
                
                # Create visualization
                if st.button("Create Visualization"):
                    fig = px.scatter(df, x=x_axis, y=y_axis, 
                                   title=f"{y_axis} vs {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation analysis
                    correlation = df[numeric_columns].corr()
                    fig_corr = px.imshow(correlation, text_auto=True, 
                                       title="Correlation Matrix")
                    st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Show sample data format
        st.subheader("Expected Data Format")
        sample_data = {
            'datetime': ['2024-01-01 08:00:00', '2024-01-01 09:00:00', '2024-01-01 10:00:00'],
            'irradiance': [600, 800, 1000],
            'temperature': [20, 22, 25],
            'energy_output': [2.4, 3.2, 4.0],
            'panel_voltage': [24.1, 24.3, 24.5],
            'panel_current': [2.1, 2.8, 3.5]
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()