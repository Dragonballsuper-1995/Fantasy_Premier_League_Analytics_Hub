import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import pandas as pd

# Define the directory to save models
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True) # Create the directory if it doesn't exist

POSITIONS = ['GK', 'DEF', 'MID', 'FWD']

def get_models():
    """ Returns a dictionary of 6 different regression models. """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Support Vector (SVR)": SVR(C=1.0, epsilon=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, n_jobs=-1)
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
    Loads all 24 pre-trained models (6 models x 4 positions)
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