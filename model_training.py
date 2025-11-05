import os
import joblib
import json # <-- NEW IMPORT
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import pandas as pd
import lightgbm as lgb
import catboost as cb

# Define the directory to save models
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True) # Create the directory if it doesn't exist

POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
PARAMS_FILE = 'best_params.json'

# *** NEW: Function to load parameters from file ***
def load_best_parameters():
    """
    Loads tuned parameters from best_params.json.
    If the file doesn't exist, it returns default parameters.
    """
    
    # These are your original, un-tuned parameters.
    # They will be used as a fallback if best_params.json is not found.
    DEFAULT_PARAMS = {
        "Linear Regression": {},
        "Ridge Regression": {'alpha': 1.0},
        "Lasso Regression": {'alpha': 0.1},
        "Support Vector (SVR)": {'C': 1.0, 'epsilon': 0.1},
        "Random Forest": {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
        "XGBoost": {'objective': 'reg:squarederror', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42, 'n_jobs': -1},
        "LightGBM": {'random_state': 42, 'n_jobs': -1},
        "CatBoost": {'random_state': 42, 'verbose': 0, 'thread_count': -1}
    }

    if not os.path.exists(PARAMS_FILE):
        print(f"Warning: '{PARAMS_FILE}' not found. Using default model parameters.")
        return DEFAULT_PARAMS

    try:
        with open(PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
        print(f"Successfully loaded tuned parameters from '{PARAMS_FILE}'.")
        
        # Merge with defaults to add random_state, n_jobs etc.
        # This ensures keys not in the JSON (like 'random_state') are still set.
        for model_name, params in DEFAULT_PARAMS.items():
            if model_name in best_params:
                # Start with default params (like random_state)
                # and overwrite with tuned params (like alpha)
                full_params = params.copy()
                full_params.update(best_params[model_name])
                best_params[model_name] = full_params
            else:
                # If model is not in JSON, use its default
                best_params[model_name] = params
                
        return best_params
        
    except Exception as e:
        print(f"Error loading '{PARAMS_FILE}': {e}. Using default model parameters.")
        return DEFAULT_PARAMS


def get_models():
    """ 
    Returns a dictionary of 8 different regression models,
    initialized with the best parameters from best_params.json.
    """
    
    # Load parameters from file (or defaults if file not found)
    params = load_best_parameters()
    
    models = {
        # The ** operator unpacks the dictionary of parameters
        "Linear Regression": LinearRegression(**params.get("Linear Regression", {})),
        "Ridge Regression": Ridge(**params.get("Ridge Regression", {})),
        "Lasso Regression": Lasso(**params.get("Lasso Regression", {})),
        "Support Vector (SVR)": SVR(**params.get("Support Vector (SVR)", {})),
        "Random Forest": RandomForestRegressor(**params.get("Random Forest", {})),
        "XGBoost": xgb.XGBRegressor(**params.get("XGBoost", {})),
        "LightGBM": lgb.LGBMRegressor(**params.get("LightGBM", {})),
        "CatBoost": cb.CatBoostRegressor(**params.get("CatBoost", {}))
    }
    return models

def train_models(models, X_train, y_train):
    """ Trains all models in the dictionary. """
    trained_models = {}
    X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)

    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_train_numeric, y_train)
            trained_models[name] = model
            print(f"  {name} trained.")
        except Exception as e:
            print(f"    Error training {name}: {e}")
    return trained_models

def save_models(trained_models, position):
    """ 
    Saves the trained models with a position prefix.
    """
    for name, model in trained_models.items():
        filename = os.path.join(MODEL_DIR, f"{position}_{name.replace(' ', '_')}.joblib")
        try:
            joblib.dump(model, filename)
            print(f"  Saved {position} {name} to {filename}")
        except Exception as e:
            print(f"    Error saving {name}: {e}")

def load_models():
    """ 
    Loads all pre-trained models
    """
    loaded_models = {}
    model_names = get_models().keys()
    
    print("Loading pre-trained models...")
    for pos in POSITIONS:
        loaded_models[pos] = {}
        for name in model_names:
            filename = os.path.join(MODEL_DIR, f"{pos}_{name.replace(' ', '_')}.joblib")
            if os.path.exists(filename):
                try:
                    model = joblib.load(filename)
                    loaded_models[pos][name] = model
                except Exception as e:
                     print(f"    Error loading {pos} {name} from {filename}: {e}")
            else:
                print(f"    Warning: Model file not found for {pos} {name}. It will be skipped.")
    
    print(f"Loaded {sum(len(v) for v in loaded_models.values())} models.")
    return loaded_models