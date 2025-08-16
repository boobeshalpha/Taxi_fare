🚖 GUVI Taxi Fare Prediction

Predicting taxi fares using machine learning and data analysis techniques.
This project is part of the GUVI Data Science & ML learning journey.

✨ Project Highlights

📂 Dataset: Taxi trip records with fares

🛠 Tech Stack: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit

⚙️ Pipeline: Data Cleaning → Feature Engineering → EDA → Model Training → Deployment

🎯 Goal: Build a regression model to accurately predict taxi fares

📊 Workflow

Data Preprocessing

Handle missing values

Convert datetime features

Remove anomalies

Feature Engineering

Haversine distance (pickup ↔ drop-off)

Time-based features (hour, day, month, weekday)

Trip duration

Exploratory Data Analysis (EDA)

Distribution of fares

Correlation between features

Outlier detection

Modeling & Evaluation

Linear Regression, Ridge, Lasso

Random Forest, Gradient Boosting

Hyperparameter tuning (RandomizedSearchCV)

Evaluation using RMSE & R²

Deployment

Best model saved with joblib

Interactive Streamlit app for predictions

⚡ How to Run

Run the project with:

python run.py

📈 Example Outputs

✅ Model comparison metrics (RMSE, R²)

📊 Visualizations of fare trends

🌐 Web app to predict fares instantly
