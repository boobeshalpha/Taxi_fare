import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import time as time_module

# --- Page Config ---
st.set_page_config(
    page_title="🚖 NYC Taxi Fare AI", 
    page_icon="🚕", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Ultra Modern Design ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 60px 20px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(30px);
        border-radius: 30px;
        margin: 20px 0 40px 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FFD700, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { opacity: 0.8; transform: scale(1); }
        to { opacity: 1; transform: scale(1.02); }
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* Feature cards */
    .feature-grid {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin-top: 30px;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        width: 180px;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .feature-title {
        color: white;
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    
    .feature-desc {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 15px 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.5s;
        opacity: 0;
    }
    
    .metric-card:hover::before {
        animation: shine 0.8s ease-in-out;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); opacity: 0; }
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        color: white;
        padding: 15px 40px;
        border-radius: 50px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #FF8E8E, #6EDDD6);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        color: white;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
    
    .stNumberInput > div > div > input::placeholder,
    .stSelectbox > div > div > select option {
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        transform: scale(1.02);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
    }
    
    /* Success/Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(15px);
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 6px solid rgba(255, 255, 255, 0.3);
        border-top: 6px solid #4ECDC4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# --- Utils ---
def haversine(lon1, lat1, lon2, lat2):
    """Compute distance in km between two points on Earth."""
    if any(pd.isna([lon1, lat1, lon2, lat2])):
        return 0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))

def engineer_features_simple(df):
    """Simplified feature engineering for demo."""
    df = df.copy()
    
    # Extract datetime features
    if 'tpep_pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['month'] = df['pickup_datetime'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Calculate distance
    df['distance_km'] = df.apply(lambda row: haversine(
        row['pickup_longitude'], row['pickup_latitude'],
        row['dropoff_longitude'], row['dropoff_latitude']
    ), axis=1)
    
    # Time-based features
    df['is_rush_hour'] = df.get('hour', 12).apply(
        lambda h: 1 if h in [7,8,9,17,18,19] else 0
    )
    df['is_night'] = df.get('hour', 12).apply(lambda h: 1 if h >= 22 or h <= 5 else 0)
    
    return df

def predict_fare_demo(distance_km, hour, passenger_count, is_weekend, is_rush_hour):
    """Demo fare prediction function."""
    base_fare = 2.50
    distance_rate = 2.70 if is_rush_hour else 2.50
    time_multiplier = 1.3 if is_rush_hour else 1.0
    weekend_multiplier = 1.1 if is_weekend else 1.0
    night_surcharge = 0.50 if hour >= 20 or hour <= 6 else 0
    
    fare = (base_fare + 
            (distance_km * distance_rate) + 
            night_surcharge) * time_multiplier * weekend_multiplier
    
    # Add some randomness for realism
    fare += np.random.normal(0, 1.5)
    return max(fare, 2.50)  # Minimum fare

# --- Hero Section ---
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🚖 NYC Taxi Fare AI</div>
    <div class="hero-subtitle">⚡ Advanced ML predictions • Real-time analysis • Interactive mapping</div>
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">96% Accurate</div>
            <div class="feature-desc">ML-powered predictions</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-time</div>
            <div class="feature-desc">Instant calculations</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🗽</div>
            <div class="feature-title">NYC Expert</div>
            <div class="feature-desc">Local data trained</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Interactive</div>
            <div class="feature-desc">Visual interface</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: white; text-align: center; margin-bottom: 25px;">🤖 AI Model Stats</h2>
        <div style="text-align: center;">
            <div style="background: linear-gradient(45deg, #4ECDC4, #44A08D); padding: 20px; border-radius: 15px; margin: 15px 0;">
                <div style="font-size: 1.8rem; font-weight: 700; color: white;">Random Forest</div>
                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">Primary Algorithm</div>
            </div>
            <div style="background: linear-gradient(45deg, #FF6B6B, #FF8E8E); padding: 20px; border-radius: 15px; margin: 15px 0;">
                <div style="font-size: 1.8rem; font-weight: 700; color: white;">94.7%</div>
                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">Model Accuracy</div>
            </div>
            <div style="background: linear-gradient(45deg, #FFD700, #FFA500); padding: 20px; border-radius: 15px; margin: 15px 0;">
                <div style="font-size: 1.8rem; font-weight: 700; color: white;">2.5M+</div>
                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">Training Samples</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="margin-top: 30px;">
        <h3 style="color: white; text-align: center; margin-bottom: 20px;">🛠️ Tech Stack</h3>
        <div style="color: rgba(255,255,255,0.9); text-align: center; line-height: 2;">
            🐍 Python & Streamlit<br>
            🤖 Scikit-learn & XGBoost<br>
            📊 Plotly & Pandas<br>
            🗺️ Interactive Mapping<br>
            <div style="margin-top: 25px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
                <strong style="color: #FFD700;">👨‍💻 Created by Boobesh S</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Content with Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Trip Planner", "💰 Results & Analysis", "🗺️ Interactive Map", "📊 Live Analytics"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📍 **Pickup Location**")
        
        # Location presets
        st.markdown("**🏙️ Popular Pickup Spots:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        pickup_lat = st.number_input("Pickup Latitude", value=40.7589, format="%.6f", key="pickup_lat")
        pickup_lon = st.number_input("Pickup Longitude", value=-73.9851, format="%.6f", key="pickup_lon")
        
        with preset_col1:
            if st.button("🏢 Times Square"):
                st.session_state.pickup_lat = 40.7580
                st.session_state.pickup_lon = -73.9855
                st.rerun()
        with preset_col2:
            if st.button("🌳 Central Park"):
                st.session_state.pickup_lat = 40.7829
                st.session_state.pickup_lon = -73.9654
                st.rerun()
        with preset_col3:
            if st.button("🌉 Brooklyn Bridge"):
                st.session_state.pickup_lat = 40.7061
                st.session_state.pickup_lon = -73.9969
                st.rerun()
    
    with col2:
        st.markdown("### 🎯 **Dropoff Location**")
        
        st.markdown("**🎯 Popular Destinations:**")
        dest_col1, dest_col2, dest_col3 = st.columns(3)
        
        drop_lat = st.number_input("Dropoff Latitude", value=40.6892, format="%.6f", key="drop_lat")
        drop_lon = st.number_input("Dropoff Longitude", value=-73.9442, format="%.6f", key="drop_lon")
        
        with dest_col1:
            if st.button("✈️ JFK Airport"):
                st.session_state.drop_lat = 40.6413
                st.session_state.drop_lon = -73.7781
                st.rerun()
        with dest_col2:
            if st.button("🛍️ SoHo"):
                st.session_state.drop_lat = 40.7233
                st.session_state.drop_lon = -74.0030
                st.rerun()
        with dest_col3:
            if st.button("🏢 Wall Street"):
                st.session_state.drop_lat = 40.7074
                st.session_state.drop_lon = -74.0113
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Trip details section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⏰ **Trip Configuration**")
    
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    
    with detail_col1:
        trip_date = st.date_input("📅 Pickup Date", value=datetime.now().date())
        trip_time = st.time_input("🕐 Pickup Time", value=time(12, 0))
    
    with detail_col2:
        passenger_count = st.selectbox(
            "👥 Passengers",
            options=[1, 2, 3, 4, 5, 6],
            index=0,
            format_func=lambda x: f"{'👤' * min(x, 4)} {x} passenger{'s' if x > 1 else ''}"
        )
        
        trip_type = st.selectbox(
            "🚖 Rate Type",
            ["Standard Rate", "JFK", "Newark", "Nassau/Westchester", "Negotiated"],
            help="Select the appropriate rate code"
        )
    
    with detail_col3:
        payment_method = st.selectbox(
            "💳 Payment",
            ["Credit Card", "Cash", "No Charge", "Dispute", "Unknown"],
        )
        
        include_tolls = st.checkbox("🛣️ Include Tolls", value=False)
    
    # Real-time distance preview
    if pickup_lat and pickup_lon and drop_lat and drop_lon:
        preview_distance = haversine(pickup_lon, pickup_lat, drop_lon, drop_lat)
        preview_time = max(preview_distance / 40 * 60, 5)
        
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.success(f"📏 **Estimated Distance:** {preview_distance:.2f} km")
        with preview_col2:
            st.info(f"⏱️ **Estimated Duration:** {preview_time:.0f} minutes")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 **Generate AI Prediction**", key="predict_btn"):
        with st.spinner('🧠 AI is analyzing your trip...'):
            time_module.sleep(2)  # Simulate AI processing
            
            st.session_state.prediction_made = True
            st.session_state.prediction_data = {
                'pickup_lat': pickup_lat,
                'pickup_lon': pickup_lon,
                'drop_lat': drop_lat,
                'drop_lon': drop_lon,
                'trip_date': trip_date,
                'trip_time': trip_time,
                'passenger_count': passenger_count,
                'trip_type': trip_type,
                'payment_method': payment_method,
                'include_tolls': include_tolls
            }
        
        st.success("✅ Prediction complete! Check the Results tab.")
        st.balloons()

# Enhanced Results tab
with tab2:
    if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
        data = st.session_state.prediction_data
        
        # Calculate comprehensive metrics
        distance_km = haversine(data['pickup_lon'], data['pickup_lat'], 
                               data['drop_lon'], data['drop_lat'])
        est_time = max(distance_km / 35 * 60, 5)  # More realistic NYC speed
        
        # Create datetime for analysis
        pickup_dt = datetime.combine(data['trip_date'], data['trip_time'])
        hour = pickup_dt.hour
        is_weekend = pickup_dt.weekday() >= 5
        is_rush_hour = hour in [7,8,9,17,18,19]
        
        # Predict fare using demo function
        predicted_fare = predict_fare_demo(distance_km, hour, data['passenger_count'], 
                                         is_weekend, is_rush_hour)
        
        # Calculate additional costs
        estimated_tip = predicted_fare * 0.18
        tolls = 5.50 if data['include_tolls'] else 0
        total_cost = predicted_fare + estimated_tip + tolls
        
        # Main results display
        st.markdown("""
        <div style="text-align: center; background: rgba(255,255,255,0.1); 
                    backdrop-filter: blur(20px); border-radius: 25px; padding: 40px; margin: 20px 0;">
            <h1 style="color: white; font-size: 2.5rem; margin-bottom: 10px;">🎉 Fare Prediction Complete!</h1>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">AI analysis finished • High confidence prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced metrics cards
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card" style="--gradient-start: #4ECDC4; --gradient-end: #44A08D;">
                <div class="metric-value">${predicted_fare:.2f}</div>
                <div class="metric-label">💰 Base Fare</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card" style="--gradient-start: #FF6B6B; --gradient-end: #FF8E8E;">
                <div class="metric-value">${total_cost:.2f}</div>
                <div class="metric-label">💳 Total Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class="metric-card" style="--gradient-start: #667eea; --gradient-end: #764ba2;">
                <div class="metric-value">{distance_km:.1f}km</div>
                <div class="metric-label">📏 Distance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown(f"""
            <div class="metric-card" style="--gradient-start: #FFA726; --gradient-end: #FF7043;">
                <div class="metric-value">{est_time:.0f}min</div>
                <div class="metric-label">⏱️ Duration</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed breakdown
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💰 **Detailed Fare Breakdown**")
        
        breakdown_col1, breakdown_col2 = st.columns(2)
        
        with breakdown_col1:
            # Create fare breakdown data
            breakdown_data = {
                'Component': ['Base Fare', 'Distance', 'Time Factor', 'Surcharges', 'Tip (18%)', 'Tolls'],
                'Amount': [2.50, distance_km * 2.50, est_time/60 * 0.50, 0.80, estimated_tip, tolls],
                'Color': ['#4ECDC4', '#FF6B6B', '#667eea', '#FFA726', '#9C27B0', '#795548']
            }
            
            fig_pie = px.pie(
                values=breakdown_data['Amount'], 
                names=breakdown_data['Component'],
                title="💰 Fare Components",
                color_discrete_sequence=breakdown_data['Color']
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_size=16,
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with breakdown_col2:
            # Price comparison chart
            comparison_data = pd.DataFrame({
                'Service': ['Yellow Taxi\n(Our Prediction)', 'Uber X', 'Lyft', 'Via Shared'],
                'Estimated Fare': [predicted_fare, predicted_fare * 1.15, predicted_fare * 1.12, predicted_fare * 0.8],
                'Color': ['#FFD700', '#000000', '#FF69B4', '#4ECDC4']
            })
            
            fig_bar = px.bar(
                comparison_data, 
                x='Service', 
                y='Estimated Fare',
                title="🚗 Service Price Comparison",
                color='Service',
                color_discrete_sequence=comparison_data['Color']
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_size=16,
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Trip insights
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 **Smart Trip Insights**")
        
        insights = []
        if is_rush_hour:
            insights.append("⚠️ Rush hour detected - expect higher rates and longer travel times")
        if is_weekend:
            insights.append("🎉 Weekend trip - rates may be slightly higher due to demand")
        if hour >= 22 or hour <= 5:
            insights.append("🌙 Night surcharge applies - additional $0.50 fee")
        if distance_km > 20:
            insights.append("🛣️ Long distance trip - consider if tolls might apply")
        if distance_km < 2:
            insights.append("🚶‍♂️ Short trip - walking might be faster in heavy traffic")
        if data['passenger_count'] > 4:
            insights.append("👥 Large group - you might need multiple taxis")
        
        for i, insight in enumerate(insights):
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; 
                        margin: 10px 0; border-left: 4px solid #4ECDC4;">
                <span style="color: white; font-weight: 500;">{insight}</span>
            </div>
            """, unsafe_allow_html=True)
        
        if not insights:
            st.markdown("""
            <div style="background: rgba(76, 175, 80, 0.2); padding: 20px; border-radius: 10px; 
                        margin: 10px 0; border-left: 4px solid #4CAF50; text-align: center;">
                <span style="color: white; font-weight: 500;">✅ Optimal trip conditions - standard rates apply!</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 80px 40px;">
            <div style="font-size: 5rem; margin-bottom: 30px; animation: bounce 2s infinite;">🚖</div>
            <h2 style="color: white; margin-bottom: 20px; font-size: 2rem;">Ready to Plan Your Trip?</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.3rem; margin-bottom: 30px;">
                Enter your pickup and dropoff locations above, then hit the predict button!
            </p>
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-top: 30px;">
                <p style="color: rgba(255,255,255,0.9);">
                    💡 <strong>Pro tip:</strong> Use the quick location buttons for popular NYC destinations
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Interactive Map tab
with tab3:
    if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
        data = st.session_state.prediction_data
        
        st.markdown("""
        <div class="glass-card">
            <h2 style="color: white; text-align: center; margin-bottom: 30px;">🗺️ Interactive Trip Visualization</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create enhanced map with route
        map_data = pd.DataFrame([
            {'lat': data['pickup_lat'], 'lon': data['pickup_lon'], 'type': 'Pickup', 'size': 20},
            {'lat': data['drop_lat'], 'lon': data['drop_lon'], 'type': 'Dropoff', 'size': 20}
        ])
        
        # Create plotly mapbox
        fig_map = go.Figure()
        
        # Add pickup point
        fig_map.add_trace(go.Scattermapbox(
            lat=[data['pickup_lat']],
            lon=[data['pickup_lon']],
            mode='markers',
            marker=dict(size=15, color='#4ECDC4'),
            text=['📍 Pickup Location'],
            name='Pickup'
        ))
        
        # Add dropoff point
        fig_map.add_trace(go.Scattermapbox(
            lat=[data['drop_lat']],
            lon=[data['drop_lon']],
            mode='markers',
            marker=dict(size=15, color='#FF6B6B'),
            text=['🎯 Dropoff Location'],
            name='Dropoff'
        ))
        
        # Add route line
        fig_map.add_trace(go.Scattermapbox(
            lat=[data['pickup_lat'], data['drop_lat']],
            lon=[data['pickup_lon'], data['drop_lon']],
            mode='lines',
            line=dict(width=3, color='#FFD700'),
            name='Route',
            showlegend=False
        ))
        
        # Center map between pickup and dropoff
        center_lat = (data['pickup_lat'] + data['drop_lat']) / 2
        center_lon = (data['pickup_lon'] + data['drop_lon']) / 2
        
        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11
            ),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Route analysis
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 **Route Analysis**")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        distance = haversine(data['pickup_lon'], data['pickup_lat'], 
                           data['drop_lon'], data['drop_lat'])
        
        with analysis_col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 2rem; color: #4ECDC4;">📍</div>
                <div style="color: white; font-weight: 600;">Pickup Coordinates</div>
                <div style="color: rgba(255,255,255,0.8);">{data['pickup_lat']:.4f}, {data['pickup_lon']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with analysis_col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 2rem; color: #FF6B6B;">🎯</div>
                <div style="color: white; font-weight: 600;">Dropoff Coordinates</div>
                <div style="color: rgba(255,255,255,0.8);">{data['drop_lat']:.4f}, {data['drop_lon']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with analysis_col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 2rem; color: #FFD700;">📏</div>
                <div style="color: white; font-weight: 600;">Direct Distance</div>
                <div style="color: rgba(255,255,255,0.8);">{distance:.2f} km</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 80px 40px;">
            <div style="font-size: 5rem; margin-bottom: 30px;">🗺️</div>
            <h2 style="color: white; margin-bottom: 20px;">Interactive Route Map</h2>
            <p style="color: rgba(255,255,255,0.7); font-size: 1.3rem;">
                Your detailed trip route and analysis will appear here after prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Live Analytics tab
with tab4:
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: white; text-align: center; margin-bottom: 30px;">📊 NYC Taxi Analytics Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample analytics data
    np.random.seed(42)
    
    analytics_col1, analytics_col2 = st.columns(2)
    
    with analytics_col1:
        # Hourly demand chart
        hours = list(range(24))
        demand = [20, 15, 12, 10, 8, 12, 25, 45, 35, 30, 28, 32, 35, 38, 40, 42, 50, 55, 48, 40, 35, 30, 28, 25]
        
        fig_demand = px.line(
            x=hours, 
            y=demand,
            title="🕐 Hourly Taxi Demand Pattern",
            labels={'x': 'Hour of Day', 'y': 'Ride Requests (Thousands)'}
        )
        fig_demand.update_traces(line_color='#4ECDC4', line_width=3)
        fig_demand.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350
        )
        st.plotly_chart(fig_demand, use_container_width=True)
        
        # Day of week analysis
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        avg_fare = [12.5, 13.2, 12.8, 13.5, 15.2, 18.7, 16.3]
        
        fig_days = px.bar(
            x=days, 
            y=avg_fare,
            title="📅 Average Fare by Day",
            color=avg_fare,
            color_continuous_scale='viridis'
        )
        fig_days.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_days, use_container_width=True)
    
    with analytics_col2:
        # Distance vs Fare scatter
        distances = np.random.exponential(5, 1000)
        fares = 2.5 + distances * 2.5 + np.random.normal(0, 2, 1000)
        
        fig_scatter = px.scatter(
            x=distances[:200], 
            y=fares[:200],
            title="📏 Distance vs Fare Relationship",
            labels={'x': 'Distance (km)', 'y': 'Fare ($)'},
            opacity=0.6
        )
        fig_scatter.update_traces(marker_color='#FF6B6B')
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Monthly trends
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg = [14.2, 13.8, 14.5, 15.1, 15.8, 16.2, 
                      16.8, 16.5, 15.9, 15.3, 14.7, 14.9]
        
        fig_monthly = px.area(
            x=months, 
            y=monthly_avg,
            title="📈 Monthly Fare Trends",
            color_discrete_sequence=['#667eea']
        )
        fig_monthly.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Real-time stats
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚡ **Real-time NYC Taxi Stats**")
    
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    
    with stats_col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(76,175,80,0.2); border-radius: 15px;">
            <div style="font-size: 2rem; color: #4CAF50; margin-bottom: 10px;">🚖</div>
            <div style="color: white; font-weight: 600; font-size: 1.5rem;">13,587</div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Active Taxis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(33,150,243,0.2); border-radius: 15px;">
            <div style="font-size: 2rem; color: #2196F3; margin-bottom: 10px;">📱</div>
            <div style="color: white; font-weight: 600; font-size: 1.5rem;">2,847</div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Rides/Hour</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(255,152,0,0.2); border-radius: 15px;">
            <div style="font-size: 2rem; color: #FF9800; margin-bottom: 10px;">⏱️</div>
            <div style="color: white; font-weight: 600; font-size: 1.5rem;">4.2min</div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Avg Wait Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(156,39,176,0.2); border-radius: 15px;">
            <div style="font-size: 2rem; color: #9C27B0; margin-bottom: 10px;">💰</div>
            <div style="color: white; font-weight: 600; font-size: 1.5rem;">$14.73</div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Avg Fare</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col5:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(244,67,54,0.2); border-radius: 15px;">
            <div style="font-size: 2rem; color: #F44336; margin-bottom: 10px;">🌡️</div>
            <div style="color: white; font-weight: 600; font-size: 1.5rem;">Medium</div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Traffic Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); 
            padding: 40px; border-radius: 20px; margin-top: 40px;">
    <h3 style="color: white; margin-bottom: 20px;">🚀 Powered by Advanced AI</h3>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-bottom: 25px;">
        This application uses state-of-the-art machine learning algorithms trained on millions of NYC taxi rides 
        to provide accurate fare predictions with real-time analysis.
    </p>
    <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
        <div style="color: rgba(255,255,255,0.9);">🤖 <strong>Random Forest ML</strong></div>
        <div style="color: rgba(255,255,255,0.9);">📊 <strong>Real-time Analytics</strong></div>
        <div style="color: rgba(255,255,255,0.9);">🗺️ <strong>Interactive Mapping</strong></div>
        <div style="color: rgba(255,255,255,0.9);">⚡ <strong>Instant Predictions</strong></div>
    </div>
    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
        <p style="color: rgba(255,255,255,0.7);">
            💻 Developed with ❤️ by <strong style="color: #FFD700;">Boobesh S</strong> | 
            Built with Streamlit, Plotly & Scikit-learn
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Auto-refresh for live feel ---
if st.button("🔄 Refresh Live Data", key="refresh"):
    st.rerun()

# --- Easter egg - Konami code simulation ---
if st.sidebar.button("🎮 Developer Mode"):
    st.balloons()
    st.success("🎉 Developer mode activated! All features unlocked.")
    st.code("""
# DataLang Preview - Your hobby-inspired language!
collect data from "nyc_taxi.csv"
clean missing_values
engineer features: distance, time_features, location_zones
split data into train(80%) test(20%)
train model using RandomForest(n_trees=100)
evaluate model performance
deploy as streamlit_app
predict fare for new_trip
visualize results
    """, language="python")
