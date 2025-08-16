import argparse
from data_utils import load_data, basic_cleaning, DEFAULT_DATA_PATH
from features import engineer_features
from eda import run_eda
from model import train_and_evaluate
from predict import load_model, predict_from_csv
from app import run_streamlit_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "streamlit"], required=True)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_path", default="outputs/best_model.pkl")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--predict_csv", default=DEFAULT_DATA_PATH)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        df = load_data(args.data)
        df = basic_cleaning(df)
        df = engineer_features(df)

        # ✅ Automatically detect target column
        if "total_amount" in df.columns:
            target_col = "total_amount"
        elif "fare_amount" in df.columns:
            target_col = "fare_amount"
        else:
            raise ValueError("No suitable target column found (expected 'total_amount' or 'fare_amount').")

        run_eda(df, f"{args.out_dir}/eda")
        train_and_evaluate(df, target_col, args.out_dir, tune=args.tune)

    elif args.mode == "predict":
        model = load_model(args.model_path)
        predict_from_csv(model, args.predict_csv, f"{args.out_dir}/predictions.csv")

    elif args.mode == "streamlit":
        run_streamlit_app(args.model_path)
