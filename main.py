import pandas as pd
import numpy as np
import json
import time
import os
from config import (
    HISTORY_DATA_URL, FIXTURES_URL, TEAMS_URL, 
    OUTPUT_JSON_FILE, PLAYER_RAW_URL
)
from feature_engineering import get_team_strength
from model_training import load_models, get_models, POSITIONS

def load_data(filepath, file_type):
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        if df.empty:
            print(f"Warning: {file_type} data from {filepath} is empty or unreadable.")
        return df
    except Exception as e:
        print(f"Error loading {file_type} data from {filepath}: {e}")
        return pd.DataFrame()

def run_prediction_pipeline():
    start_time = time.time()

    print("Loading data sources...")
    history_df = load_data(HISTORY_DATA_URL, "History")
    fixtures_df = load_data(FIXTURES_URL, "Fixtures")
    teams_df = load_data(TEAMS_URL, "Teams")
    players_raw_df = load_data(PLAYER_RAW_URL, "Player Raw")

    if history_df.empty or fixtures_df.empty or teams_df.empty or players_raw_df.empty:
        print("One or more essential data files failed to load. Exiting.")
        return

    print("\nLoading pre-trained models...")
    trained_models = load_models()
    if not trained_models or all(len(v) == 0 for v in trained_models.values()):
        print("Error: No pre-trained models found. Run 'python train_offline.py' first.")
        return

    team_id_map, team_name_map = get_team_strength(teams_df)

    try:
        players_raw_df['id'] = pd.to_numeric(players_raw_df['id'], errors='coerce')
        players_raw_df = players_raw_df.dropna(subset=['id'])
        players_raw_df = players_raw_df.drop_duplicates(subset=['id'], keep='first')
        players_raw_df['id'] = players_raw_df['id'].astype(int)
        
        player_meta_map = players_raw_df.set_index('id')[
            ['web_name', 'chance_of_playing_next_round', 'news', 'now_cost']
        ].to_dict('index')
    except KeyError as e:
        print(f"Error creating player map from players_raw.csv: Missing column {e}")
        return

    next_gw_series = fixtures_df[fixtures_df['finished'] == False]['event']
    if next_gw_series.empty:
        last_played_gw = history_df['GW'].max() if 'GW' in history_df.columns and not history_df.empty else 0
        next_gw = last_played_gw + 1 if last_played_gw > 0 else 1
        print(f"Warning: No future GWs found. Assuming next GW is {next_gw}.")
        if next_gw > 38: print("End of season."); return
    else:
        next_gw = int(next_gw_series.min())
    print(f"Predicting for GW {next_gw} onwards.")

    print("Preparing feature data for batch prediction...")
    all_pred_rows_data = [] 
    all_pred_metadata = [] 

    feature_cols = [
        'total_points_lag1', 'minutes_lag1', 'goals_scored_lag1', 'assists_lag1', 
        'clean_sheets_lag1', 'bonus_lag1', 'bps_lag1', 'ict_index_lag1', 
        'influence_lag1', 'creativity_lag1', 'threat_lag1',
        'expected_goals_lag1', 'expected_assists_lag1', 'expected_goal_involvements_lag1', 'expected_goals_conceded_lag1',
        'form_points_last_5', 'form_ict_last_5', 'form_minutes_last_5',
        'form_xg_last_5', 'form_xa_last_5', 'form_xgi_last_5', 'form_xgc_last_5',
        'opponent_attack_strength', 'opponent_defence_strength', 'was_home'
    ]

    if 'GW' not in history_df.columns and 'round' in history_df.columns: history_df['GW'] = history_df['round']
    history_df['GW'] = pd.to_numeric(history_df['GW'], errors='coerce').fillna(0).astype(int)
    if 'element' not in history_df.columns:
        print("Error: 'element' column (player ID) not found in history_df. Cannot proceed.")
        return
    history_df['element'] = pd.to_numeric(history_df['element'], errors='coerce')
    history_df = history_df.dropna(subset=['element'])
    history_df['element'] = history_df['element'].astype(int)
    history_df = history_df.sort_values(by=['element', 'GW'])

    grouped = history_df.groupby('element')
    history_df['form_points_last_5'] = grouped['total_points'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    history_df['form_ict_last_5'] = grouped['ict_index'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    history_df['form_minutes_last_5'] = grouped['minutes'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    history_df['form_xg_last_5'] = grouped['expected_goals'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    history_df['form_xa_last_5'] = grouped['expected_assists'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    history_df['form_xgi_last_5'] = grouped['expected_goal_involvements'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    history_df['form_xgc_last_5'] = grouped['expected_goals_conceded'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    
    latest_player_data = history_df.loc[history_df.groupby('element')['GW'].idxmax()]

    for _, last_gw_data in latest_player_data.iterrows():
        element_id = last_gw_data.get('element') 
        name = last_gw_data.get('name')
        player_team_name = last_gw_data.get('team')
        player_pos = last_gw_data.get('position')
        
        if pd.isna(element_id) or pd.isna(name) or pd.isna(player_team_name) or pd.isna(player_pos) or player_team_name not in team_name_map or player_pos not in POSITIONS:
            continue
            
        element_id = int(element_id)
        player_team_id = team_name_map[player_team_name]['id']

        base_pred_data_dict = {
            'total_points_lag1': last_gw_data.get('total_points', 0),
            'minutes_lag1': last_gw_data.get('minutes', 0),
            'goals_scored_lag1': last_gw_data.get('goals_scored', 0),
            'assists_lag1': last_gw_data.get('assists', 0),
            'clean_sheets_lag1': last_gw_data.get('clean_sheets', 0),
            'bonus_lag1': last_gw_data.get('bonus', 0),
            'bps_lag1': last_gw_data.get('bps', 0),
            'ict_index_lag1': last_gw_data.get('ict_index', 0.0),
            'influence_lag1': last_gw_data.get('influence', 0.0),
            'creativity_lag1': last_gw_data.get('creativity', 0.0),
            'threat_lag1': last_gw_data.get('threat', 0.0),
            'expected_goals_lag1': last_gw_data.get('expected_goals', 0.0),
            'expected_assists_lag1': last_gw_data.get('expected_assists', 0.0),
            'expected_goal_involvements_lag1': last_gw_data.get('expected_goal_involvements', 0.0),
            'expected_goals_conceded_lag1': last_gw_data.get('expected_goals_conceded', 0.0),
            'form_points_last_5': last_gw_data.get('form_points_last_5', 0.0),
            'form_ict_last_5': last_gw_data.get('form_ict_last_5', 0.0),
            'form_minutes_last_5': last_gw_data.get('form_minutes_last_5', 0.0),
            'form_xg_last_5': last_gw_data.get('form_xg_last_5', 0.0),
            'form_xa_last_5': last_gw_data.get('form_xa_last_5', 0.0),
            'form_xgi_last_5': last_gw_data.get('form_xgi_last_5', 0.0),
            'form_xgc_last_5': last_gw_data.get('form_xgc_last_5', 0.0),
        }

        player_fixtures = fixtures_df[
            ((fixtures_df['team_h'] == player_team_id) | (fixtures_df['team_a'] == player_team_id)) &
            (fixtures_df['event'] >= next_gw)
        ].sort_values('event').head(5)

        for _, fixture in player_fixtures.iterrows():
            fixture_gw = int(fixture['event'])
            is_home = (fixture['team_h'] == player_team_id)
            opponent_team_id_raw = fixture['team_a'] if is_home else fixture['team_h']
            try: opponent_team_id = int(opponent_team_id_raw) if pd.notna(opponent_team_id_raw) else None
            except ValueError: opponent_team_id = None

            opp_full_name, opp_att, opp_def = "Unknown", 1000, 1000
            if opponent_team_id is not None and opponent_team_id in team_id_map:
                opp_strength_data = team_id_map[opponent_team_id]
                opp_full_name = opp_strength_data.get('name', 'Unknown')
                opp_att = opp_strength_data.get('strength_attack_away' if is_home else 'strength_attack_home', 1000)
                opp_def = opp_strength_data.get('strength_defence_away' if is_home else 'strength_defence_home', 1000)

            current_pred_data_dict = base_pred_data_dict.copy()
            current_pred_data_dict['opponent_attack_strength'] = opp_att
            current_pred_data_dict['opponent_defence_strength'] = opp_def
            current_pred_data_dict['was_home'] = 1 if is_home else 0

            all_pred_rows_data.append(current_pred_data_dict)
            opponent_str = f"{opp_full_name} {'(H)' if is_home else '(A)'}"
            all_pred_metadata.append({'element_id': element_id, 'name': name, 'gw': fixture_gw, 'opponent': opponent_str, 'position': player_pos})

    if not all_pred_rows_data:
        print("No feature rows generated for prediction.")
        return

    X_predict_batch = pd.DataFrame(all_pred_rows_data)
    metadata_df = pd.DataFrame(all_pred_metadata, index=X_predict_batch.index)
    X_predict_batch = X_predict_batch.reindex(columns=feature_cols, fill_value=0)
    X_predict_batch_numeric = X_predict_batch.apply(pd.to_numeric, errors='coerce').fillna(0)

    print(f"Predicting {len(X_predict_batch)} fixtures in batch...")
    batch_predictions = {}
    
    for model_name in get_models().keys():
        final_preds_for_model = np.zeros(len(X_predict_batch_numeric))
        for pos in POSITIONS:
            if pos not in trained_models or model_name not in trained_models[pos]:
                continue
            model = trained_models[pos][model_name]
            pos_indices = metadata_df[metadata_df['position'] == pos].index
            if pos_indices.empty:
                continue
            pos_X_predict = X_predict_batch_numeric.loc[pos_indices]
            pos_preds = model.predict(pos_X_predict)
            final_preds_for_model[pos_indices] = pos_preds
            
        batch_predictions[model_name] = [float(max(0.0, np.round(p, 2))) for p in final_preds_for_model]
    
    prediction_time = time.time()
    print(f"Batch prediction finished in {prediction_time - start_time:.2f} seconds.")

    print("Restructuring results into final JSON format...")
    final_player_data_map = {} 

    for i, meta in enumerate(all_pred_metadata):
        element_id = meta['element_id']
        player_name = meta['name']
        
        if element_id not in final_player_data_map:
             player_hist = history_df[history_df['element'] == element_id].tail(5)
             if player_hist.empty: continue 
             last_gw_info = player_hist.iloc[-1]
             
             status_info = player_meta_map.get(element_id, {})
             chance_raw = status_info.get('chance_of_playing_next_round')
             chance_of_playing = 100 if pd.isna(chance_raw) else int(chance_raw)
             
             news_raw = status_info.get('news')
             news = "" if pd.isna(news_raw) else news_raw 
             
             cost = float(status_info.get('now_cost', 0)) / 10.0
             
             points_raw = player_hist['total_points'].fillna(0).astype(float).tolist()
             labels_raw = player_hist['GW'].astype(int).tolist()
             opponents_raw = []
             
             for _, gw_row in player_hist.iterrows():
                 opp_team_identifier = gw_row.get('opponent_team') 
                 was_home = gw_row.get('was_home', False)
                 opp_name_hist = "N/A"
                 try:
                     opp_id_int = int(opp_team_identifier)
                     if opp_id_int in team_id_map:
                         opp_name_hist = team_id_map[opp_id_int].get('name', 'N/A')
                 except (ValueError, TypeError):
                     if isinstance(opp_team_identifier, str) and opp_team_identifier in team_name_map:
                          opp_name_hist = opp_team_identifier
                 
                 location = " (H)" if was_home else " (A)"
                 opponents_raw.append(f"{opp_name_hist}{location}")

             num_missing = 5 - len(points_raw)
             if labels_raw:
                 start_gw = labels_raw[0]
                 pad_labels = [f"GW{start_gw - i}" for i in range(num_missing, 0, -1) if start_gw - i > 0]
                 pad_labels = ["Start"] * (num_missing - len(pad_labels)) + pad_labels
             else:
                 pad_labels = [f"Start-{j}" for j in range(num_missing, 0, -1)]
             
             pad_points = [0.0] * num_missing
             pad_opponents = ["-"] * num_missing
             current_labels = pad_labels + [f"GW{g}" for g in labels_raw]
             current_points = pad_points + points_raw
             current_opponents = pad_opponents + opponents_raw

             final_player_data_map[element_id] = {
                 "id": element_id,
                 "web_name": status_info.get('web_name', player_name),
                 "team": last_gw_info.get('team', 'Unknown'),
                 "position": last_gw_info.get('position', 'UNK'),
                 "prev_gw_points": int(last_gw_info.get('total_points', 0)),
                 "last_5_gw_points": current_points,
                 "last_5_gw_labels": current_labels,
                 "last_5_gw_opponents": current_opponents,
                 "next_opponent": "N/A",
                 "upcoming_predictions": [],
                 "chance_of_playing": chance_of_playing,
                 "news": news, # This is now a clean string
                 "cost": cost
             }

        multiplier = 1.0
        if meta['gw'] == next_gw:
            multiplier = final_player_data_map[element_id]['chance_of_playing'] / 100.0
        
        fixture_preds = {}
        for model_name, preds in batch_predictions.items():
            raw_pred = preds[i]
            adjusted_pred = raw_pred * multiplier
            fixture_preds[model_name] = round(adjusted_pred, 2)
            
        final_player_data_map[element_id]['upcoming_predictions'].append({
             "gw": meta['gw'],
             "opponent": meta['opponent'],
             "predictions": fixture_preds
        })
        if meta['gw'] == next_gw:
             final_player_data_map[element_id]['next_opponent'] = meta['opponent']

    final_player_data_list = list(final_player_data_map.values())

    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_player_data_list, f, indent=4, ensure_ascii=False, default=str)
        end_time = time.time()
        print(f"\nSUCCESS: All predictions generated and saved to '{OUTPUT_JSON_FILE}'.")
        print(f"Total time: {end_time - start_time:.2f} seconds.")
    except TypeError as e:
        print(f"\nError saving JSON: {e}")

if __name__ == "__main__":
    run_prediction_pipeline()