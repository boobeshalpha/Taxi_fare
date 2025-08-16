import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

def run_eda(df, out_dir):
    """
    Perform comprehensive EDA on taxi data and save all plots to `out_dir`.

    Generates:
      1. Fare amount distribution (clipped at 99th percentile)
      2. Fare vs Distance scatter plot
      3. Trips by Pickup Hour
      4. Correlation heatmap (numeric)
      5. Fare by Pickup Hour (boxplot)
      6. Fare by Day of Week (boxplot)
      7. Fare by Month (if available)
      8. Distribution of Trip Distance and Duration
      9. Fare per km and per min by trip distance bins
     10. Trip counts by Pickup Day and Hour (heatmap)
     11. Fare by Night (is_night) and Weekend (is_weekend) (boxplots)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Total Amount Distribution
    if 'total_amount' in df.columns:
        plt.figure(figsize=(8,4))
        clipped = df['total_amount'].clip(upper=df['total_amount'].quantile(0.99))
        sns.histplot(clipped, bins=100)
        plt.title('Total Amount (Clipped)')
        plt.xlabel('Total Amount')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'total_amount_hist.png'))
        plt.close()

    # 2. Fare vs Distance scatter
    if 'trip_distance_km' in df.columns and 'total_amount' in df.columns:
        sample = df.sample(min(5000, len(df)), random_state=42)
        plt.figure(figsize=(8,5))
        sns.scatterplot(x='trip_distance_km', y='total_amount', data=sample, alpha=0.5)
        plt.title('Fare vs Distance (sample)')
        plt.xlabel('Distance (km)')
        plt.ylabel('Total Amount')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fare_vs_distance.png'))
        plt.close()

    # 3. Trips by Pickup Hour (countplot)
    if 'pickup_hour' in df.columns:
        plt.figure(figsize=(10,4))
        sns.countplot(x='pickup_hour', data=df, color='skyblue')
        plt.title('Trips by Pickup Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Trips')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'trips_by_hour.png'))
        plt.close()

    # 4. Correlation heatmap (numeric features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, cmap='coolwarm', annot=False)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'))
        plt.close()

    # 5. Boxplot: Fare by Pickup Hour (trend in hourly fares)
    if 'pickup_hour' in df.columns and 'total_amount' in df.columns:
        plt.figure(figsize=(11,5))
        sns.boxplot(x='pickup_hour', y='total_amount', data=df, showfliers=False)
        plt.title('Fare Distribution by Pickup Hour')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fare_by_hour.png'))
        plt.close()

    # 6. Boxplot: Fare by Day of Week (trend)
    if 'pickup_dayofweek' in df.columns and 'total_amount' in df.columns:
        plt.figure(figsize=(10,4))
        sns.boxplot(x='pickup_dayofweek', y='total_amount', data=df, showfliers=False)
        plt.title('Fare by Day of Week (0=Mon ... 6=Sun)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fare_by_dayofweek.png'))
        plt.close()

    # 7. Boxplot: Fare by Month (if available)
    if 'pickup_month' in df.columns and 'total_amount' in df.columns:
        plt.figure(figsize=(10,4))
        sns.boxplot(x='pickup_month', y='total_amount', data=df, showfliers=False)
        plt.title('Fare by Month')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fare_by_month.png'))
        plt.close()

    # 8. Distribution of trip distance & duration
    if 'trip_distance_km' in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df['trip_distance_km'].clip(upper=df['trip_distance_km'].quantile(0.99)), bins=100)
        plt.title('Distribution of Trip Distance (km, clipped)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'trip_distance_hist.png'))
        plt.close()
    if 'trip_duration_min' in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df['trip_duration_min'].clip(upper=df['trip_duration_min'].quantile(0.99)), bins=100)
        plt.title('Distribution of Trip Duration (min, clipped)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'trip_duration_hist.png'))
        plt.close()

    # 9. Fare per km/min by trip distance bins
    # Calculate these fields if not present
    if 'fare_per_km' not in df.columns and 'trip_distance_km' in df.columns and 'total_amount' in df.columns:
        df['fare_per_km'] = df['total_amount'] / df['trip_distance_km'].replace(0, np.nan)
    if 'fare_per_min' not in df.columns and 'trip_duration_min' in df.columns and 'total_amount' in df.columns:
        df['fare_per_min'] = df['total_amount'] / df['trip_duration_min'].replace(0, np.nan)

    if 'fare_per_km' in df.columns and 'trip_distance_km' in df.columns:
        df['distance_bin'] = pd.cut(df['trip_distance_km'], bins=[0,1,2,5,10,20,50,100])
        plt.figure(figsize=(10,4))
        sns.boxplot(x='distance_bin', y='fare_per_km', data=df, showfliers=False)
        plt.title('Fare per km by Trip Distance Bin')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fare_per_km_by_distance_bin.png'))
        plt.close()
    if 'fare_per_min' in df.columns and 'pickup_hour' in df.columns:
        plt.figure(figsize=(10,4))
        sns.boxplot(x='pickup_hour', y='fare_per_min', data=df, showfliers=False)
        plt.title('Fare per Minute by Pickup Hour')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fare_per_min_by_hour.png'))
        plt.close()

    # 10. Trip count heatmap (pickup_dayofweek x pickup_hour)
    if 'pickup_dayofweek' in df.columns and 'pickup_hour' in df.columns:
        pivot = pd.pivot_table(df, index='pickup_dayofweek', columns='pickup_hour', values='total_amount', aggfunc='count').fillna(0)
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot, cmap='Blues')
        plt.title('Trip Counts: Day of Week vs Hour')
        plt.ylabel('Day of Week (0=Mon ... 6=Sun)')
        plt.xlabel('Hour of Day')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'tripcount_day_hour_heatmap.png'))
        plt.close()

    # 11. Fare by Night vs Day and Weekend vs Weekday
    if 'is_night' in df.columns and 'total_amount' in df.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(x='is_night', y='total_amount', data=df, showfliers=False)
        plt.title('Fare by Night (1=Night, 0=Day)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fare_by_night.png'))
        plt.close()
    if 'is_weekend' in df.columns and 'total_amount' in df.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(x='is_weekend', y='total_amount', data=df, showfliers=False)
        plt.title('Fare by Weekend (1=Weekend, 0=Weekday)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fare_by_weekend.png'))
        plt.close()

    print(f"✅ EDA completed. All advanced plots saved to: {out_dir}")
