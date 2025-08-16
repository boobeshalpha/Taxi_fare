import pandas as pd

# Default dataset path
DEFAULT_DATA_PATH = r"C:\Users\keert\OneDrive\Desktop\GUVI_TAXI\taxi_fare.csv"

def load_data(path=DEFAULT_DATA_PATH):
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    print("\n✅ Dataset Loaded")
    print(f"Shape: {df.shape}")
    print("\n📄 First 5 rows:")
    print(df.head())
    print("\nℹ️ Data Types & Non-Null Counts:")
    print(df.info())
    print("\n🔍 Missing Values:")
    print(df.isnull().sum())
    print("\n🔁 Duplicate Rows:", df.duplicated().sum())
    return df


def basic_cleaning(df):
    df = df.rename(columns=lambda x: x.strip())  # clean column names
    df = df.drop_duplicates()
    if 'total_amount' in df.columns:
        df = df[df['total_amount'] > 0]
    return df
