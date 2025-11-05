import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import json

from config import TRAIN_DATA_URL, TEAMS_URL
from feature_engineering import get_team_strength, create_features

def load_data(filepath, file_type):
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        if df.empty:
            print(f"Warning: {file_type} data from {filepath} is empty or unreadable.")
        return df
    except Exception as e:
        print(f"Error loading {file_type} data from {filepath}: {e}")
        return pd.DataFrame()

def get_models_and_param_grids():
    param_grids = {
        "Linear Regression": {
            "model": LinearRegression(),
            "grid": {},
            "search_type": "grid"
        },
        "Ridge Regression": {
            "model": Ridge(),
            "grid": {
                'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]
            },
            "search_type": "grid"
        },
        "Lasso Regression": {
            "model": Lasso(),
            "grid": {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            "search_type": "grid"
        },
        "Support Vector (SVR)": {
            "model": SVR(),
            "grid": {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.1, 0.2, 0.5]
            },
            "search_type": "random"
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "grid": {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10]
            },
            "search_type": "random"
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            "grid": {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            },
            "search_type": "random"
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
            "grid": {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70],
                'max_depth': [-1, 10, 20]
            },
            "search_type": "random"
        },
        "CatBoost": {
            "model": cb.CatBoostRegressor(random_state=42, verbose=0, thread_count=-1),
            "grid": {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8, 10],
            },
            "search_type": "random"
        }
    }
    return param_grids

def run_tuning():
    print("--- Starting Hyperparameter Tuning ---")
    
    print(f"Loading training data (2024-25 season) from {TRAIN_DATA_URL}...")
    train_df = load_data(TRAIN_DATA_URL, "Training")
    teams_df = load_data(TEAMS_URL, "Teams")

    if train_df.empty or teams_df.empty:
        print("Required data for training is missing. Exiting.")
        return

    _, team_name_map = get_team_strength(teams_df)

    if 'GW' not in train_df.columns and 'round' in train_df.columns: train_df['GW'] = train_df['round']
    train_df['GW'] = pd.to_numeric(train_df['GW'], errors='coerce')
    train_df = train_df.dropna(subset=['GW', 'position'])
    train_df['GW'] = train_df['GW'].astype(int)

    print("Creating features for all positions...")
    X_train, y_train, _ = create_features(train_df, team_name_map)
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)

    if X_train.empty or y_train.empty:
        print("Feature creation resulted in empty data. Skipping.")
        return

    models_to_tune = get_models_and_param_grids()
    best_params_all = {}

    for name, config in models_to_tune.items():
        if not config["grid"]:
            print(f"\nSkipping {name} (no parameters to tune).")
            best_params_all[name] = {}
            continue
        
        print(f"\n--- Tuning {name} ---")
        start_time = time.time()
        
        search = None
        if config["search_type"] == "grid":
            print(f"Running GridSearchCV...")
            search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["grid"],
                cv=3,
                n_jobs=-1,
                scoring='neg_mean_squared_error'
            )
        elif config["search_type"] == "random":
            print(f"Running RandomizedSearchCV (n_iter=10)...")
            search = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["grid"],
                n_iter=10, 
                cv=3,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                random_state=42
            )
        
        search.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"  Best Score (RMSE): {np.sqrt(-search.best_score_):.4f}")
        print(f"  Best Parameters: {search.best_params_}")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        
        best_params_all[name] = search.best_params_

    print("\n\n--- ALL TUNING COMPLETE ---")
    print("Saving best parameters to best_params.json...")
    
    try:
        with open('best_params.json', 'w') as f:
            json.dump(best_params_all, f, indent=4)
        print("  Successfully saved to best_params.json")
    except Exception as e:
        print(f"  Error saving parameters: {e}")

if __name__ == "__main__":
    run_tuning()