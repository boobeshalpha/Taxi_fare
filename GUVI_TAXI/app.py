import streamlit as st
import pandas as pd
import math
from datetime import datetime
from predict import load_model
from features import engineer_features

st.set_page_config(page_title="Taxi_fare", page_icon="🚖") 

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    km = 6371 * c
    return km

def run_streamlit_app(model_path):
    
    st.title("🚖 Taxi Fare Prediction")

    st.markdown(
        """
        This app predicts taxi fares based on your input parameters.
        Please provide pickup/dropoff coordinates, time, passenger count, and other info.
        """
    )

    model = load_model(model_path)

    with st.form("input_form"):
        st.header("Trip Details")

        pickup_lat = st.number_input(
            "Pickup Latitude", value=40.75, format="%.6f",
            help="Latitude for pickup location (-90 to 90)",
            min_value=-90.0, max_value=90.0)
        pickup_lon = st.number_input(
            "Pickup Longitude", value=-73.98, format="%.6f",
            help="Longitude for pickup location (-180 to 180)",
            min_value=-180.0, max_value=180.0)

        drop_lat = st.number_input(
            "Dropoff Latitude", value=40.65, format="%.6f",
            help="Latitude for dropoff location (-90 to 90)",
            min_value=-90.0, max_value=90.0)
        drop_lon = st.number_input(
            "Dropoff Longitude", value=-73.78, format="%.6f",
            help="Longitude for dropoff location (-180 to 180)",
            min_value=-180.0, max_value=180.0)

        dt = st.date_input("Pickup Date")
        tm = st.time_input("Pickup Time")

        passenger_count = st.slider(
            "Passenger Count", min_value=1, max_value=6, value=1,
            help="Number of passengers")

        st.header("Additional Info")

        ratecode_options = {
            "Standard rate": 1,
            "JFK": 2,
            "Newark": 3,
            "Nassau or Westchester": 4,
            "Negotiated fare": 5,
            "Group ride": 6,
            "Unknown": None
        }
        ratecode = st.selectbox(
            "RatecodeID", options=list(ratecode_options.keys()), index=0,
            help="Select the rate code of the trip")

        store_and_fwd_flag = st.selectbox(
            "Store and Forward Flag",
            options=["N", "Y", None],
            index=0,
            help="Indicates whether the trip record was held in vehicle memory before sending to the vendor")

        payment_type_options = {
            "Credit card": 1,
            "Cash": 2,
            "No charge": 3,
            "Dispute": 4,
            "Unknown": 5,
            "Voided trip": 6
        }
        payment_type = st.selectbox(
            "Payment Type", options=list(payment_type_options.keys()), index=0,
            help="Select the payment method")

        fare_amount = st.number_input(
            "Fare Amount (optional)", min_value=0.0, value=0.0, step=0.5,
            help="Enter fare amount if known (else leave zero)")

        submit = st.form_submit_button("Predict Fare")

    if submit:
        # Calculate distance
        distance_km = haversine(pickup_lon, pickup_lat, drop_lon, drop_lat)
        st.write(f"**Distance:** {distance_km:.2f} km")

        # Estimate travel time assuming avg speed 40 km/h
        est_time = distance_km / 40 * 60  # minutes
        st.write(f"**Estimated travel time:** {est_time:.1f} minutes")

        fare_per_km = fare_amount / distance_km if distance_km > 0 else 0.0
        st.write(f"**Fare per km (auto-calculated):** ${fare_per_km:.2f}")

        row = pd.DataFrame([{
            'pickup_latitude': pickup_lat,
            'pickup_longitude': pickup_lon,
            'dropoff_latitude': drop_lat,
            'dropoff_longitude': drop_lon,
            'tpep_pickup_datetime': f"{dt} {tm}",
            'passenger_count': passenger_count,
            'fare_amount': fare_amount,
            'fare_per_km': fare_per_km,
            'RatecodeID': ratecode_options[ratecode],
            'store_and_fwd_flag': store_and_fwd_flag,
            'tip_amount': 0,
            'payment_type': payment_type_options[payment_type],
            'mta_tax': 0,
            'tolls_amount': 0,
            'VendorID': 1,
            'improvement_surcharge': 0,
            'extra': 0
        }])

        row = engineer_features(row)

        # Fill missing expected columns from OneHotEncoder
        if 'preproc' in model.named_steps:
            cat_cols = model.named_steps['preproc'].transformers_[1][2]
            for col in cat_cols:
                if col not in row.columns:
                    row[col] = 0


        fare = model.predict(row)[0]
        st.success(f"Predicted Fare: ${fare:.2f}")
