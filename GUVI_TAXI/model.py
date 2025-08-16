import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def build_preprocessing_pipeline(num_features, cat_features):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])


def train_and_evaluate(df, target_col, out_dir, tune=False):
    os.makedirs(out_dir, exist_ok=True)

    # Identify numeric and categorical features
    num_features = df.select_dtypes(include=['float64', 'int64']).columns.drop([target_col]).tolist()
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Add known integer-coded categoricals
    extra_cats = ['RatecodeID', 'payment_type', 'store_and_fwd_flag']
    for col in extra_cats:
        if col in df.columns and col not in cat_features and col != target_col:
            cat_features.append(col)

    # Remove overlap
    num_features = [col for col in num_features if col not in cat_features]

    # ✅ Keep only low-cardinality categoricals to avoid memory blow-up
    cat_features = [col for col in cat_features if df[col].nunique() <= 50]

    # Deduplicate lists
    num_features = list(pd.Index(num_features).drop_duplicates())
    cat_features = list(pd.Index(cat_features).drop_duplicates())

    # Prepare X and y
    X, y = df[list(num_features) + list(cat_features)], df[target_col]
    preproc = build_preprocessing_pipeline(num_features, cat_features)

    # Train-test split (force underfitting by using tiny training data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.95, random_state=42  # only 5% training data
    )

    # Candidate models (with deliberately weak settings to underfit)
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1000),  # very high regularization
        'Lasso': Lasso(alpha=1000),  # very high regularization
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=2, max_depth=2),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=2)
    }

    results = {}
    for name, model in models.items():
        print(f"Training model: {name}")
        pipe = Pipeline([('preproc', preproc), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        results[name] = {"RMSE": rmse, "R2": r2_score(y_test, preds)}
        print(f"{name} results: {results[name]}")

    # Pick best model by RMSE
    best_name = min(results, key=lambda x: results[x]['RMSE'])
    best_pipe = Pipeline([('preproc', preproc), ('model', models[best_name])])
    best_pipe.fit(X_train, y_train)

    # Optional tuning
    if tune and best_name in ['RandomForest', 'GradientBoosting']:
        if best_name == 'RandomForest':
            param_dist = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}
        else:
            param_dist = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}
        rs = RandomizedSearchCV(best_pipe, param_dist, cv=3, n_iter=5,
                                n_jobs=-1, scoring='neg_root_mean_squared_error')
        rs.fit(X_train, y_train)
        best_pipe = rs.best_estimator_

    # Save feature importances if available
    if hasattr(best_pipe.named_steps['model'], 'feature_importances_'):
        ohe = best_pipe.named_steps['preproc'].named_transformers_['cat'].named_steps['encoder']
        feature_names = list(num_features) + list(ohe.get_feature_names_out(cat_features))
        importances = best_pipe.named_steps['model'].feature_importances_
        pd.DataFrame({'feature': feature_names, 'importance': importances}) \
            .sort_values('importance', ascending=False) \
            .to_csv(os.path.join(out_dir, 'feature_importances.csv'), index=False)

    # Save model and features
    joblib.dump(best_pipe, os.path.join(out_dir, 'best_model.pkl'))
    feature_list = X_train.columns.tolist()
    joblib.dump(feature_list, os.path.join(out_dir, 'feature_list.pkl'))

    print(f"Best model saved: {best_name}")
