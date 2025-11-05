import pandas as pd
from config import TRAIN_DATA_URL, TEAMS_URL
from feature_engineering import get_team_strength, create_features
from model_training import get_models, train_models, save_models, POSITIONS
import os

def load_data(filepath, file_type):
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        if df.empty:
            print(f"Warning: {file_type} data from {filepath} is empty or unreadable.")
        return df
    except Exception as e:
        print(f"Error loading {file_type} data from {filepath}: {e}")
        return pd.DataFrame()

def run_training():
    print("--- Starting Offline Model Training ---")

    print(f"Loading training data (2024-25 season) from {TRAIN_DATA_URL}...")
    train_df = load_data(TRAIN_DATA_URL, "Training")
    teams_df = load_data(TEAMS_URL, "Teams")

    if train_df.empty or teams_df.empty:
        print("Required data for training is missing. Exiting.")
        return

    team_id_map, team_name_map = get_team_strength(teams_df)

    if 'GW' not in train_df.columns and 'round' in train_df.columns: train_df['GW'] = train_df['round']
    train_df['GW'] = pd.to_numeric(train_df['GW'], errors='coerce')
    train_df = train_df.dropna(subset=['GW', 'position'])
    train_df['GW'] = train_df['GW'].astype(int)

    for pos in POSITIONS:
        print(f"\n--- Processing models for {pos} ---")
        
        pos_train_df = train_df[train_df['position'] == pos].copy()
        
        if pos_train_df.empty:
            print(f"No training data found for {pos}. Skipping.")
            continue

        print(f"Creating features for {pos}...")
        X_train, y_train, df_clean = create_features(pos_train_df, team_name_map)

        if X_train.empty or y_train.empty:
            print(f"Feature creation for {pos} resulted in empty data. Skipping.")
            continue

        models_to_train = get_models()
        trained_models = train_models(models_to_train, X_train, y_train)

        if trained_models:
            print(f"Saving models for {pos}...")
            save_models(trained_models, pos)

    print("\n--- Offline training complete. All models saved. ---")

if __name__ == "__main__":
    run_training()