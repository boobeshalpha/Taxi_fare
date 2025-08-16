import os

def train():
    os.system('python main.py --mode train --data taxi_fare.csv --out_dir outputs --tune')

def predict():
    os.system('python main.py --mode predict --model_path outputs/best_model.pkl --predict_csv taxi_fare.csv --out_dir outputs')

def app():
    os.system('streamlit run main.py -- --mode streamlit --model_path outputs/best_model.pkl')

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Train model")
    print("2. Predict from CSV")
    print("3. Launch Streamlit app")

    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        train()
    elif choice == "2":
        predict()
    elif choice == "3":
        app()
    else:
        print("Invalid choice.")
