
# 🚖 GUVI Taxi Fare Prediction

Predicting taxi fares using **machine learning and data analysis techniques**.
This project is part of the **GUVI Data Science & ML learning journey**.

---

## ✨ Project Highlights

* 📂 **Dataset**: Taxi trip records with fares
* 🛠 **Tech Stack**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit
* ⚙️ **Pipeline**: Data Cleaning → Feature Engineering → EDA → Model Training → Deployment
* 🎯 **Goal**: Build a regression model to accurately predict taxi fares

---

## 📊 Workflow

1. **Data Preprocessing**

   * Handle missing values
   * Convert datetime features
   * Remove anomalies

2. **Feature Engineering**

   * Haversine distance (pickup ↔ drop-off)
   * Time-based features (hour, day, month, weekday)
   * Trip duration

3. **Exploratory Data Analysis (EDA)**

   * Distribution of fares
   * Correlation between features
   * Outlier detection

4. **Modeling & Evaluation**

   * Linear Regression, Ridge, Lasso
   * Random Forest, Gradient Boosting
   * Hyperparameter tuning (RandomizedSearchCV)
   * Evaluation using RMSE & R²

5. **Deployment**

   * Best model saved with `joblib`
   * Interactive **Streamlit app** for predictions

---

## ⚡ How to Run

Run the project with:

```bash
python run.py
```

---

## 📈 Example Outputs

* ✅ Model comparison metrics (RMSE, R²)
* 📊 Visualizations of fare trends
* 🌐 Web app to predict fares instantly

---

## 📜 License
---
Link for CSV file:https://drive.google.com/file/d/1VUb9ucTsroGDBOPcwpOfXwzDi-rd4wqQ/view?usp=sharing

This project is created for **educational purposes under GUVI**.
Feel free to use and adapt it for learning and research.


