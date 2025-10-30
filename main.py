# # # main.py (Updated for dynamic labels, opponent tooltips, and full team names)
# # import pandas as pd
# # import numpy as np
# # import json
# # from config import TRAIN_DATA_URL, HISTORY_DATA_URL, FIXTURES_URL, TEAMS_URL, OUTPUT_JSON_FILE
# # from feature_engineering import get_team_strength, create_features
# # from model_training import get_models, train_models

# # def load_data(filepath, file_type):
# #     """ Loads a CSV, skipping bad lines. """
# #     try:
# #         df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')
# #         if df.empty:
# #             print(f"Warning: {file_type} data from {filepath} is empty or unreadable.")
# #         return df
# #     except Exception as e:
# #         print(f"Error loading {file_type} data from {filepath}: {e}")
# #         return pd.DataFrame()

# # def run_prediction_pipeline():

# #     # --- 1. LOAD ALL DATA ---
# #     print("Loading all data sources...")
# #     train_df = load_data(TRAIN_DATA_URL, "Training") # 2024-25
# #     history_df = load_data(HISTORY_DATA_URL, "History") # 2025-26 past GWs
# #     fixtures_df = load_data(FIXTURES_URL, "Fixtures") # 2025-26 all GWs
# #     teams_df = load_data(TEAMS_URL, "Teams") # Team strength

# #     if train_df.empty or history_df.empty or fixtures_df.empty or teams_df.empty:
# #         print("One or more essential data files failed to load. Exiting.")
# #         return

# #     # --- 2. PREPARE TEAM & FIXTURE DATA ---
# #     team_id_map, team_name_map = get_team_strength(teams_df)

# #     # Find the next gameweek to predict
# #     next_gw_series = fixtures_df[fixtures_df['finished'] == False]['event']
# #     if next_gw_series.empty:
# #         last_played_gw = history_df['GW'].max() if not history_df.empty else 0
# #         next_gw = last_played_gw + 1 if last_played_gw > 0 else 1
# #         print(f"Warning: No future GWs found. Assuming next GW is {next_gw}.")
# #         if next_gw > 38:
# #              print("End of season detected.")
# #              return
# #     else:
# #         next_gw = int(next_gw_series.min())

# #     print(f"\nPredicting for GW {next_gw} onwards.")

# #     # --- 3. TRAIN MODELS ---
# #     print("Creating training features...")
# #     if 'GW' not in train_df.columns and 'round' in train_df.columns: train_df['GW'] = train_df['round']
# #     train_df['GW'] = pd.to_numeric(train_df['GW'], errors='coerce')
# #     train_df = train_df.dropna(subset=['GW'])
# #     train_df['GW'] = train_df['GW'].astype(int)

# #     X_train, y_train, _ = create_features(train_df, team_name_map)

# #     models_to_train = get_models()
# #     trained_models = train_models(models_to_train, X_train, y_train)

# #     if not trained_models:
# #         print("No models were trained successfully. Exiting.")
# #         return

# #     # --- 4. GENERATE PREDICTIONS FOR ALL PLAYERS ---
# #     print(f"Generating predictions up to GW {min(38, next_gw + 4)}...")

# #     final_player_data = []
# #     feature_cols = X_train.columns

# #     if 'GW' not in history_df.columns and 'round' in history_df.columns: history_df['GW'] = history_df['round']
# #     history_df['GW'] = pd.to_numeric(history_df['GW'], errors='coerce')
# #     history_df = history_df.dropna(subset=['GW'])
# #     history_df['GW'] = history_df['GW'].astype(int)
# #     history_df = history_df.sort_values(by=['name', 'GW'])

# #     # Group by player to get their latest stats and form
# #     for name, player_gws in history_df.groupby('name'):

# #         if player_gws.empty: continue

# #         last_gw_data = player_gws.iloc[-1]
# #         player_team_name = last_gw_data['team']
# #         last_played_gw_num = last_gw_data['GW']

# #         if player_team_name not in team_name_map: continue

# #         player_team_id = team_name_map[player_team_name]['id']

# #         # --- Calculate current form features ---
# #         form_points_last_5 = player_gws['total_points'].rolling(window=5, min_periods=1).mean().iloc[-1]
# #         form_ict_last_5 = player_gws['ict_index'].rolling(window=5, min_periods=1).mean().iloc[-1]
# #         form_minutes_last_5 = player_gws['minutes'].rolling(window=5, min_periods=1).mean().iloc[-1]

# #         # --- Base feature dict ---
# #         base_pred_data_dict = {
# #             'total_points_lag1': last_gw_data.get('total_points', 0), # Use .get for safety
# #             'minutes_lag1': last_gw_data.get('minutes', 0),
# #             'goals_scored_lag1': last_gw_data.get('goals_scored', 0),
# #             'assists_lag1': last_gw_data.get('assists', 0),
# #             'clean_sheets_lag1': last_gw_data.get('clean_sheets', 0),
# #             'bonus_lag1': last_gw_data.get('bonus', 0),
# #             'bps_lag1': last_gw_data.get('bps', 0),
# #             'ict_index_lag1': last_gw_data.get('ict_index', 0.0),
# #             'influence_lag1': last_gw_data.get('influence', 0.0),
# #             'creativity_lag1': last_gw_data.get('creativity', 0.0),
# #             'threat_lag1': last_gw_data.get('threat', 0.0),
# #             'form_points_last_5': form_points_last_5,
# #             'form_ict_last_5': form_ict_last_5,
# #             'form_minutes_last_5': form_minutes_last_5,
# #             'opponent_attack_strength': 1000,
# #             'opponent_defence_strength': 1000,
# #             'was_home': 0
# #         }

# #         # --- Find next 5 fixtures & predict ---
# #         player_fixtures = fixtures_df[
# #             ((fixtures_df['team_h'] == player_team_id) | (fixtures_df['team_a'] == player_team_id)) &
# #             (fixtures_df['event'] >= next_gw)
# #         ].sort_values('event').head(5)

# #         upcoming_predictions_list = []
# #         next_opponent_details = "N/A"

# #         for index, fixture in player_fixtures.iterrows():
# #             fixture_gw = int(fixture['event'])
# #             is_home = (fixture['team_h'] == player_team_id)
# #             opponent_team_id_raw = fixture['team_a'] if is_home else fixture['team_h']
# #             try: opponent_team_id = int(opponent_team_id_raw) if pd.notna(opponent_team_id_raw) else None
# #             except ValueError: opponent_team_id = None

# #             opp_full_name = "Unknown"
# #             opp_attack_strength = 1000
# #             opp_defence_strength = 1000

# #             if opponent_team_id is not None and opponent_team_id in team_id_map:
# #                 opp_strength_data = team_id_map[opponent_team_id]
# #                 opp_full_name = opp_strength_data.get('name', 'Unknown')
# #                 opp_attack_strength = opp_strength_data.get('strength_attack_away' if is_home else 'strength_attack_home', 1000)
# #                 opp_defence_strength = opp_strength_data.get('strength_defence_away' if is_home else 'strength_defence_home', 1000)

# #             current_pred_data_dict = base_pred_data_dict.copy()
# #             current_pred_data_dict['opponent_attack_strength'] = opp_attack_strength
# #             current_pred_data_dict['opponent_defence_strength'] = opp_defence_strength
# #             current_pred_data_dict['was_home'] = 1 if is_home else 0

# #             X_pred_row = pd.DataFrame([current_pred_data_dict])
# #             X_pred_row = X_pred_row.reindex(columns=feature_cols, fill_value=0)
# #             X_pred_row_numeric = X_pred_row.apply(pd.to_numeric, errors='coerce').fillna(0)

# #             fixture_predictions = {}
# #             for model_name, model in trained_models.items():
# #                 try:
# #                     if model_name in trained_models:
# #                         pred = model.predict(X_pred_row_numeric)[0]
# #                         fixture_predictions[model_name] = float(max(0.0, np.round(pred, 2)))
# #                     else: fixture_predictions[model_name] = 0.0
# #                 except Exception: fixture_predictions[model_name] = 0.0

# #             location_str = "(H)" if is_home else "(A)"
# #             opponent_str = f"{opp_full_name} {location_str}"
# #             upcoming_predictions_list.append({
# #                 "gw": fixture_gw,
# #                 "opponent": opponent_str,
# #                 "predictions": fixture_predictions
# #             })

# #             if fixture_gw == next_gw:
# #                 next_opponent_details = opponent_str

# #         # --- Get UI data (Last 5 GWs Info) ---
# #         prev_gw_points = last_gw_data.get('total_points', 0)

# #         last_5_gws_data = player_gws.tail(5)
# #         last_5_gw_points_raw = last_5_gws_data['total_points'].fillna(0).astype(float).tolist()
# #         last_5_gw_labels_raw = last_5_gws_data['GW'].astype(int).tolist()
# #         # *** NEW ***: Get opponents for the last 5 GWs
# #         last_5_gw_opponents_raw = []
# #         for _, gw_row in last_5_gws_data.iterrows():
# #             opp_id = gw_row.get('opponent_team') # Opponent ID from history
# #             was_home = gw_row.get('was_home', False)
# #             opp_name = "N/A"
# #             try: # Try converting opp_id to int if it's not already
# #                  opp_id_int = int(opp_id) if pd.notna(opp_id) else None
# #                  if opp_id_int in team_id_map:
# #                       opp_name = team_id_map[opp_id_int].get('name', 'N/A')
# #             except (ValueError, TypeError): # Handle cases where opp_id is not a number like expected
# #                  # If opponent_team is a name string (as in vaastav data)
# #                  if isinstance(opp_id, str) and opp_id in team_name_map:
# #                       opp_name = opp_id # Use the name directly
# #             location = " (H)" if was_home else " (A)"
# #             last_5_gw_opponents_raw.append(f"{opp_name}{location}")


# #         # Pad if less than 5 games played
# #         num_missing = 5 - len(last_5_gw_points_raw)
# #         if num_missing > 0:
# #             pad_labels = [f"Start-{i}" for i in range(num_missing, 0, -1)]
# #             pad_points = [0.0] * num_missing
# #             pad_opponents = ["-"] * num_missing # Add placeholder opponents
# #             last_5_gw_labels = pad_labels + [f"GW{gw}" for gw in last_5_gw_labels_raw]
# #             last_5_gw_points = pad_points + last_5_gw_points_raw
# #             last_5_gw_opponents = pad_opponents + last_5_gw_opponents_raw
# #         else:
# #             last_5_gw_labels = [f"GW{gw}" for gw in last_5_gw_labels_raw]
# #             last_5_gw_points = last_5_gw_points_raw
# #             last_5_gw_opponents = last_5_gw_opponents_raw


# #         # Build the final JSON object for this player
# #         player_json = {
# #             "id": name,
# #             "web_name": name,
# #             "team": player_team_name,
# #             "prev_gw_points": int(prev_gw_points),
# #             "last_5_gw_points": last_5_gw_points,
# #             "last_5_gw_labels": last_5_gw_labels,
# #             "last_5_gw_opponents": last_5_gw_opponents, # *** NEW *** Add opponents list
# #             "next_opponent": next_opponent_details,
# #             "upcoming_predictions": upcoming_predictions_list
# #         }

# #         final_player_data.append(player_json)

# #     # --- 5. SAVE THE FINAL FILE ---
# #     try:
# #         with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
# #             json.dump(final_player_data, f, indent=4, ensure_ascii=False, default=str)
# #         print(f"\nSUCCESS: All predictions generated and saved to '{OUTPUT_JSON_FILE}'.")
# #         print("You can now open your fpl_predictor.html file.")
# #     except TypeError as e:
# #         print(f"\nError saving JSON: {e}")

# # if __name__ == "__main__":
# #     run_prediction_pipeline()




# import pandas as pd
# import numpy as np
# import json
# import time
# import os
# from config import HISTORY_DATA_URL, FIXTURES_URL, TEAMS_URL, OUTPUT_JSON_FILE
# from feature_engineering import get_team_strength # We only need get_team_strength
# from model_training import load_models, get_models, POSITIONS # Load models, get_models (for names), and POSITIONS

# def load_data(filepath, file_type):
#     """ Loads a CSV, skipping bad lines. """
#     try:
#         df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')
#         if df.empty:
#             print(f"Warning: {file_type} data from {filepath} is empty or unreadable.")
#         return df
#     except Exception as e:
#         print(f"Error loading {file_type} data from {filepath}: {e}")
#         return pd.DataFrame()

# def run_prediction_pipeline():
#     start_time = time.time()

#     # --- 1. LOAD DATA & MODELS ---
#     print("Loading data sources...")
#     history_df = load_data(HISTORY_DATA_URL, "History") # 2025-26 past GWs
#     fixtures_df = load_data(FIXTURES_URL, "Fixtures")
#     teams_df = load_data(TEAMS_URL, "Teams")

#     if history_df.empty or fixtures_df.empty or teams_df.empty:
#         print("One or more essential data files failed to load. Exiting.")
#         return

#     print("\nLoading pre-trained models...")
#     trained_models = load_models() # This now returns {'GK': {...}, 'DEF': {...}, ...}
#     if not trained_models or all(len(v) == 0 for v in trained_models.values()):
#         print("Error: No pre-trained models found. Run 'python train_offline.py' first.")
#         return

#     # --- 2. PREPARE LOOKUPS & FIND NEXT GW ---
#     team_id_map, team_name_map = get_team_strength(teams_df)

#     next_gw_series = fixtures_df[fixtures_df['finished'] == False]['event']
#     if next_gw_series.empty:
#         last_played_gw = history_df['GW'].max() if 'GW' in history_df.columns and not history_df.empty else 0
#         next_gw = last_played_gw + 1 if last_played_gw > 0 else 1
#         print(f"Warning: No future GWs found. Assuming next GW is {next_gw}.")
#         if next_gw > 38: print("End of season."); return
#     else:
#         next_gw = int(next_gw_series.min())
#     print(f"Predicting for GW {next_gw} onwards.")

#     # --- 3. PREPARE BATCH FEATURE DATA ---
#     print("Preparing feature data for batch prediction...")
#     all_pred_rows_data = [] # List of feature dicts
#     all_pred_metadata = [] # List of (player_name, gw, opponent_str, position) tuples

#     # Define feature columns our models expect (must match feature_engineering.py)
#     feature_cols = [
#         'total_points_lag1', 'minutes_lag1', 'goals_scored_lag1', 'assists_lag1', 
#         'clean_sheets_lag1', 'bonus_lag1', 'bps_lag1', 'ict_index_lag1', 
#         'influence_lag1', 'creativity_lag1', 'threat_lag1',
#         'expected_goals_lag1', 'expected_assists_lag1', 'expected_goal_involvements_lag1', 'expected_goals_conceded_lag1',
#         'form_points_last_5', 'form_ict_last_5', 'form_minutes_last_5',
#         'form_xg_last_5', 'form_xa_last_5', 'form_xgi_last_5', 'form_xgc_last_5',
#         'opponent_attack_strength', 'opponent_defence_strength', 'was_home'
#     ]

#     # Pre-calculate rolling form features for all players in history_df
#     if 'GW' not in history_df.columns and 'round' in history_df.columns: history_df['GW'] = history_df['round']
#     history_df['GW'] = pd.to_numeric(history_df['GW'], errors='coerce').fillna(0).astype(int)
#     history_df = history_df.sort_values(by=['name', 'GW'])

#     # Shift(1) to get *past* form, not including current row
#     grouped = history_df.groupby('name')
#     # Fillna(0) for new columns if they don't exist
#     history_df['form_points_last_5'] = grouped['total_points'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_ict_last_5'] = grouped['ict_index'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_minutes_last_5'] = grouped['minutes'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xg_last_5'] = grouped['expected_goals'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xa_last_5'] = grouped['expected_assists'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xgi_last_5'] = grouped['expected_goal_involvements'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xgc_last_5'] = grouped['expected_goals_conceded'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    
#     # Get the latest row for each player, which now contains their form *up to* that game
#     latest_player_data = history_df.loc[history_df.groupby('name')['GW'].idxmax()]

#     # Iterate through unique players
#     for name_index, last_gw_data in latest_player_data.iterrows():
#         name = last_gw_data.get('name')
#         player_team_name = last_gw_data.get('team')
#         player_pos = last_gw_data.get('position') # *** NEW: Get player position ***
        
#         if not all([name, player_team_name, player_pos, player_team_name in team_name_map, player_pos in POSITIONS]):
#             continue # Skip if missing critical data or invalid position

#         player_team_id = team_name_map[player_team_name]['id']

#         # Base features from the last completed GW
#         base_pred_data_dict = {
#             'total_points_lag1': last_gw_data.get('total_points', 0),
#             'minutes_lag1': last_gw_data.get('minutes', 0),
#             'goals_scored_lag1': last_gw_data.get('goals_scored', 0),
#             'assists_lag1': last_gw_data.get('assists', 0),
#             'clean_sheets_lag1': last_gw_data.get('clean_sheets', 0),
#             'bonus_lag1': last_gw_data.get('bonus', 0),
#             'bps_lag1': last_gw_data.get('bps', 0),
#             'ict_index_lag1': last_gw_data.get('ict_index', 0.0),
#             'influence_lag1': last_gw_data.get('influence', 0.0),
#             'creativity_lag1': last_gw_data.get('creativity', 0.0),
#             'threat_lag1': last_gw_data.get('threat', 0.0),
#             'expected_goals_lag1': last_gw_data.get('expected_goals', 0.0),
#             'expected_assists_lag1': last_gw_data.get('expected_assists', 0.0),
#             'expected_goal_involvements_lag1': last_gw_data.get('expected_goal_involvements', 0.0),
#             'expected_goals_conceded_lag1': last_gw_data.get('expected_goals_conceded', 0.0),
#             'form_points_last_5': last_gw_data.get('form_points_last_5', 0.0),
#             'form_ict_last_5': last_gw_data.get('form_ict_last_5', 0.0),
#             'form_minutes_last_5': last_gw_data.get('form_minutes_last_5', 0.0),
#             'form_xg_last_5': last_gw_data.get('form_xg_last_5', 0.0),
#             'form_xa_last_5': last_gw_data.get('form_xa_last_5', 0.0),
#             'form_xgi_last_5': last_gw_data.get('form_xgi_last_5', 0.0),
#             'form_xgc_last_5': last_gw_data.get('form_xgc_last_5', 0.0),
#         }

#         # Find next 5 fixtures
#         player_fixtures = fixtures_df[
#             ((fixtures_df['team_h'] == player_team_id) | (fixtures_df['team_a'] == player_team_id)) &
#             (fixtures_df['event'] >= next_gw)
#         ].sort_values('event').head(5)

#         for _, fixture in player_fixtures.iterrows():
#             fixture_gw = int(fixture['event'])
#             is_home = (fixture['team_h'] == player_team_id)
#             opponent_team_id_raw = fixture['team_a'] if is_home else fixture['team_h']
#             try: opponent_team_id = int(opponent_team_id_raw) if pd.notna(opponent_team_id_raw) else None
#             except ValueError: opponent_team_id = None

#             opp_full_name, opp_att, opp_def = "Unknown", 1000, 1000
#             if opponent_team_id is not None and opponent_team_id in team_id_map:
#                 opp_strength_data = team_id_map[opponent_team_id]
#                 opp_full_name = opp_strength_data.get('name', 'Unknown')
#                 opp_att = opp_strength_data.get('strength_attack_away' if is_home else 'strength_attack_home', 1000)
#                 opp_def = opp_strength_data.get('strength_defence_away' if is_home else 'strength_defence_home', 1000)

#             current_pred_data_dict = base_pred_data_dict.copy()
#             current_pred_data_dict['opponent_attack_strength'] = opp_att
#             current_pred_data_dict['opponent_defence_strength'] = opp_def
#             current_pred_data_dict['was_home'] = 1 if is_home else 0

#             all_pred_rows_data.append(current_pred_data_dict)
#             opponent_str = f"{opp_full_name} {'(H)' if is_home else '(A)'}"
#             # *** NEW: Add position to metadata ***
#             all_pred_metadata.append({'player_name': name, 'gw': fixture_gw, 'opponent': opponent_str, 'position': player_pos})

#     if not all_pred_rows_data:
#         print("No feature rows generated for prediction.")
#         return

#     # Convert to DataFrame
#     X_predict_batch = pd.DataFrame(all_pred_rows_data)
#     # Store positions separately
#     positions_series = pd.Series([meta['position'] for meta in all_pred_metadata], index=X_predict_batch.index)
#     # Prepare numeric features DataFrame
#     X_predict_batch = X_predict_batch.reindex(columns=feature_cols, fill_value=0)
#     X_predict_batch_numeric = X_predict_batch.apply(pd.to_numeric, errors='coerce').fillna(0)

#     # --- 4. BATCH PREDICT PER POSITION (FAST) ---
#     print(f"Predicting {len(X_predict_batch)} fixtures in batch...")
#     batch_predictions = {} # Stores final prediction lists { 'Linear Regression': [ ... ], ... }
    
#     for model_name in get_models().keys():
#         # print(f"  Predicting with {model_name}...")
#         final_preds_for_model = np.zeros(len(X_predict_batch_numeric)) # Array to hold all preds
        
#         for pos in POSITIONS:
#             # Check if we have models for this position
#             if pos not in trained_models or model_name not in trained_models[pos]:
#                 continue # Skip if model failed training or no players
                
#             model = trained_models[pos][model_name]
            
#             # Get indices for rows that match this position
#             pos_indices = positions_series[positions_series == pos].index
#             if pos_indices.empty:
#                 continue
                
#             # Get the feature rows for just these players
#             pos_X_predict = X_predict_batch_numeric.loc[pos_indices]
            
#             # Predict in one batch
#             pos_preds = model.predict(pos_X_predict)
            
#             # Place predictions back into the correct slots
#             final_preds_for_model[pos_indices] = pos_preds
            
#         # Add the complete list of predictions for this model
#         batch_predictions[model_name] = [float(max(0.0, np.round(p, 2))) for p in final_preds_for_model]
    
#     prediction_time = time.time()
#     print(f"Batch prediction finished in {prediction_time - start_time:.2f} seconds.")

#     # --- 5. RESTRUCTURE RESULTS ---
#     print("Restructuring results into final JSON format...")
#     final_player_data_map = {} 

#     # Populate map with predictions
#     for i, meta in enumerate(all_pred_metadata):
#         player_name = meta['player_name']
        
#         if player_name not in final_player_data_map:
#              # Initialize player entry
#              # Use loc for safe access
#              last_gw_info_series = latest_player_data[latest_player_data['name'] == player_name]
#              if last_gw_info_series.empty: continue
#              last_gw_info = last_gw_info_series.iloc[0]
             
#              player_hist = history_df[history_df['name'] == player_name].tail(5)

#              points_raw = player_hist['total_points'].fillna(0).astype(float).tolist()
#              labels_raw = player_hist['GW'].astype(int).tolist()
#              opponents_raw = []
             
#              # Logic to get past opponents
#              for _, gw_row in player_hist.iterrows():
#                  opp_team_identifier = gw_row.get('opponent_team') 
#                  was_home = gw_row.get('was_home', False)
#                  opp_name_hist = "N/A"
#                  try:
#                      opp_id_int = int(opp_team_identifier)
#                      if opp_id_int in team_id_map:
#                          opp_name_hist = team_id_map[opp_id_int].get('name', 'N/A')
#                  except (ValueError, TypeError):
#                      if isinstance(opp_team_identifier, str) and opp_team_identifier in team_name_map:
#                           opp_name_hist = opp_team_identifier # Use the name directly
                 
#                  location = " (H)" if was_home else " (A)"
#                  opponents_raw.append(f"{opp_name_hist}{location}")

#              num_missing = 5 - len(points_raw)
#              if labels_raw:
#                  start_gw = labels_raw[0]
#                  pad_labels = [f"GW{start_gw - i}" for i in range(num_missing, 0, -1)]
#              else:
#                  pad_labels = [f"Start-{j}" for j in range(num_missing, 0, -1)]
             
#              pad_points = [0.0] * num_missing
#              pad_opponents = ["-"] * num_missing
#              current_labels = pad_labels + [f"GW{g}" for g in labels_raw]
#              current_points = pad_points + points_raw
#              current_opponents = pad_opponents + opponents_raw

#              final_player_data_map[player_name] = {
#                  "id": player_name,
#                  "web_name": player_name,
#                  "team": last_gw_info.get('team', 'Unknown'),
#                  "position": last_gw_info.get('position', 'UNK'),
#                  "prev_gw_points": int(last_gw_info.get('total_points', 0)),
#                  "last_5_gw_points": current_points,
#                  "last_5_gw_labels": current_labels,
#                  "last_5_gw_opponents": current_opponents,
#                  "next_opponent": "N/A",
#                  "upcoming_predictions": []
#              }

#         # Add predictions for the current fixture
#         fixture_preds = {model_name: preds[i] for model_name, preds in batch_predictions.items()}
#         final_player_data_map[player_name]['upcoming_predictions'].append({
#              "gw": meta['gw'],
#              "opponent": meta['opponent'],
#              "predictions": fixture_preds
#         })
#         if meta['gw'] == next_gw:
#              final_player_data_map[player_name]['next_opponent'] = meta['opponent']

#     final_player_data_list = list(final_player_data_map.values())

#     # --- 6. SAVE THE FINAL FILE ---
#     try:
#         with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
#             json.dump(final_player_data_list, f, indent=4, ensure_ascii=False, default=str)
#         end_time = time.time()
#         print(f"\nSUCCESS: All predictions generated and saved to '{OUTPUT_JSON_FILE}'.")
#         print(f"Total time: {end_time - start_time:.2f} seconds.")
#     except TypeError as e:
#         print(f"\nError saving JSON: {e}")

# if __name__ == "__main__":
#     run_prediction_pipeline()


# import pandas as pd
# import numpy as np
# import json
# import time
# import os
# # *** UPDATED ***: Import PLAYER_RAW_URL
# from config import (
#     HISTORY_DATA_URL, FIXTURES_URL, TEAMS_URL, 
#     OUTPUT_JSON_FILE, PLAYER_RAW_URL
# )
# from feature_engineering import get_team_strength
# from model_training import load_models, get_models, POSITIONS

# def load_data(filepath, file_type):
#     """ Loads a CSV, skipping bad lines. """
#     try:
#         df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')
#         if df.empty:
#             print(f"Warning: {file_type} data from {filepath} is empty or unreadable.")
#         return df
#     except Exception as e:
#         print(f"Error loading {file_type} data from {filepath}: {e}")
#         return pd.DataFrame()

# def run_prediction_pipeline():
#     start_time = time.time()

#     # --- 1. LOAD DATA & MODELS ---
#     print("Loading data sources...")
#     history_df = load_data(HISTORY_DATA_URL, "History")
#     fixtures_df = load_data(FIXTURES_URL, "Fixtures")
#     teams_df = load_data(TEAMS_URL, "Teams")
#     # *** NEW ***: Load players_raw.csv
#     players_raw_df = load_data(PLAYER_RAW_URL, "Player Raw")

#     if history_df.empty or fixtures_df.empty or teams_df.empty or players_raw_df.empty:
#         print("One or more essential data files failed to load. Exiting.")
#         return

#     print("\nLoading pre-trained models...")
#     trained_models = load_models()
#     if not trained_models or all(len(v) == 0 for v in trained_models.values()):
#         print("Error: No pre-trained models found. Run 'python train_offline.py' first.")
#         return

#     # --- 2. PREPARE LOOKUPS & FIND NEXT GW ---
#     team_id_map, team_name_map = get_team_strength(teams_df)

#     # *** NEW ***: Create a status/news map from players_raw.csv
#     # Use 'web_name' as the key. If there are duplicate web_name rows, drop duplicates
#     # to ensure the index is unique before calling to_dict(orient='index').
#     try:
#         required_cols = ['chance_of_playing_next_round', 'news', 'now_cost']
#         if 'web_name' not in players_raw_df.columns:
#             raise KeyError('web_name')

#         missing = [c for c in required_cols if c not in players_raw_df.columns]
#         if missing:
#             raise KeyError(','.join(missing))

#         # Drop duplicate web_name entries (keep first). This prevents
#         # ValueError: "DataFrame index must be unique for orient='index'."
#         players_clean = players_raw_df.drop_duplicates(subset='web_name', keep='first')
#         player_meta_map = players_clean.set_index('web_name')[required_cols].to_dict('index')
#         if len(player_meta_map) < len(players_clean):
#             # Defensive: this should not happen after drop_duplicates, but warn if it does
#             print("Warning: player_meta_map length differs from players_raw after dedupe.")
#     except KeyError as e:
#         print(f"Error creating player map from players_raw.csv: Missing column(s) {e}")
#         print("Please ensure 'web_name', 'chance_of_playing_next_round', 'news', and 'now_cost' columns exist in players_raw.csv.")
#         return

#     next_gw_series = fixtures_df[fixtures_df['finished'] == False]['event']
#     if next_gw_series.empty:
#         last_played_gw = history_df['GW'].max() if 'GW' in history_df.columns and not history_df.empty else 0
#         next_gw = last_played_gw + 1 if last_played_gw > 0 else 1
#         print(f"Warning: No future GWs found. Assuming next GW is {next_gw}.")
#         if next_gw > 38: print("End of season."); return
#     else:
#         next_gw = int(next_gw_series.min())
#     print(f"Predicting for GW {next_gw} onwards.")

#     # --- 3. PREPARE BATCH FEATURE DATA ---
#     print("Preparing feature data for batch prediction...")
#     all_pred_rows_data = [] # List of feature dicts
#     all_pred_metadata = [] # List of (player_name, gw, opponent_str, position) tuples

#     # Define feature columns
#     feature_cols = [
#         'total_points_lag1', 'minutes_lag1', 'goals_scored_lag1', 'assists_lag1', 
#         'clean_sheets_lag1', 'bonus_lag1', 'bps_lag1', 'ict_index_lag1', 
#         'influence_lag1', 'creativity_lag1', 'threat_lag1',
#         'expected_goals_lag1', 'expected_assists_lag1', 'expected_goal_involvements_lag1', 'expected_goals_conceded_lag1',
#         'form_points_last_5', 'form_ict_last_5', 'form_minutes_last_5',
#         'form_xg_last_5', 'form_xa_last_5', 'form_xgi_last_5', 'form_xgc_last_5',
#         'opponent_attack_strength', 'opponent_defence_strength', 'was_home'
#     ]

#     # Pre-calculate rolling form features
#     if 'GW' not in history_df.columns and 'round' in history_df.columns: history_df['GW'] = history_df['round']
#     history_df['GW'] = pd.to_numeric(history_df['GW'], errors='coerce').fillna(0).astype(int)
#     history_df = history_df.sort_values(by=['name', 'GW'])

#     grouped = history_df.groupby('name')
#     history_df['form_points_last_5'] = grouped['total_points'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_ict_last_5'] = grouped['ict_index'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_minutes_last_5'] = grouped['minutes'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xg_last_5'] = grouped['expected_goals'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xa_last_5'] = grouped['expected_assists'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xgi_last_5'] = grouped['expected_goal_involvements'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xgc_last_5'] = grouped['expected_goals_conceded'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    
#     latest_player_data = history_df.loc[history_df.groupby('name')['GW'].idxmax()]

#     # Iterate through unique players
#     for name_index, last_gw_data in latest_player_data.iterrows():
#         name = last_gw_data.get('name')
#         player_team_name = last_gw_data.get('team')
#         player_pos = last_gw_data.get('position')
        
#         if not all([name, player_team_name, player_pos, player_team_name in team_name_map, player_pos in POSITIONS]):
#             continue

#         player_team_id = team_name_map[player_team_name]['id']

#         # Base features
#         base_pred_data_dict = {
#             'total_points_lag1': last_gw_data.get('total_points', 0),
#             'minutes_lag1': last_gw_data.get('minutes', 0),
#             'goals_scored_lag1': last_gw_data.get('goals_scored', 0),
#             'assists_lag1': last_gw_data.get('assists', 0),
#             'clean_sheets_lag1': last_gw_data.get('clean_sheets', 0),
#             'bonus_lag1': last_gw_data.get('bonus', 0),
#             'bps_lag1': last_gw_data.get('bps', 0),
#             'ict_index_lag1': last_gw_data.get('ict_index', 0.0),
#             'influence_lag1': last_gw_data.get('influence', 0.0),
#             'creativity_lag1': last_gw_data.get('creativity', 0.0),
#             'threat_lag1': last_gw_data.get('threat', 0.0),
#             'expected_goals_lag1': last_gw_data.get('expected_goals', 0.0),
#             'expected_assists_lag1': last_gw_data.get('expected_assists', 0.0),
#             'expected_goal_involvements_lag1': last_gw_data.get('expected_goal_involvements', 0.0),
#             'expected_goals_conceded_lag1': last_gw_data.get('expected_goals_conceded', 0.0),
#             'form_points_last_5': last_gw_data.get('form_points_last_5', 0.0),
#             'form_ict_last_5': last_gw_data.get('form_ict_last_5', 0.0),
#             'form_minutes_last_5': last_gw_data.get('form_minutes_last_5', 0.0),
#             'form_xg_last_5': last_gw_data.get('form_xg_last_5', 0.0),
#             'form_xa_last_5': last_gw_data.get('form_xa_last_5', 0.0),
#             'form_xgi_last_5': last_gw_data.get('form_xgi_last_5', 0.0),
#             'form_xgc_last_5': last_gw_data.get('form_xgc_last_5', 0.0),
#         }

#         # Find next 5 fixtures
#         player_fixtures = fixtures_df[
#             ((fixtures_df['team_h'] == player_team_id) | (fixtures_df['team_a'] == player_team_id)) &
#             (fixtures_df['event'] >= next_gw)
#         ].sort_values('event').head(5)

#         for _, fixture in player_fixtures.iterrows():
#             fixture_gw = int(fixture['event'])
#             is_home = (fixture['team_h'] == player_team_id)
#             opponent_team_id_raw = fixture['team_a'] if is_home else fixture['team_h']
#             try: opponent_team_id = int(opponent_team_id_raw) if pd.notna(opponent_team_id_raw) else None
#             except ValueError: opponent_team_id = None

#             opp_full_name, opp_att, opp_def = "Unknown", 1000, 1000
#             if opponent_team_id is not None and opponent_team_id in team_id_map:
#                 opp_strength_data = team_id_map[opponent_team_id]
#                 opp_full_name = opp_strength_data.get('name', 'Unknown')
#                 opp_att = opp_strength_data.get('strength_attack_away' if is_home else 'strength_attack_home', 1000)
#                 opp_def = opp_strength_data.get('strength_defence_away' if is_home else 'strength_defence_home', 1000)

#             current_pred_data_dict = base_pred_data_dict.copy()
#             current_pred_data_dict['opponent_attack_strength'] = opp_att
#             current_pred_data_dict['opponent_defence_strength'] = opp_def
#             current_pred_data_dict['was_home'] = 1 if is_home else 0

#             all_pred_rows_data.append(current_pred_data_dict)
#             opponent_str = f"{opp_full_name} {'(H)' if is_home else '(A)'}"
#             all_pred_metadata.append({'player_name': name, 'gw': fixture_gw, 'opponent': opponent_str, 'position': player_pos})

#     if not all_pred_rows_data:
#         print("No feature rows generated for prediction.")
#         return

#     # Convert to DataFrame
#     X_predict_batch = pd.DataFrame(all_pred_rows_data)
#     positions_series = pd.Series([meta['position'] for meta in all_pred_metadata], index=X_predict_batch.index)
#     X_predict_batch = X_predict_batch.reindex(columns=feature_cols, fill_value=0)
#     X_predict_batch_numeric = X_predict_batch.apply(pd.to_numeric, errors='coerce').fillna(0)

#     # --- 4. BATCH PREDICT PER POSITION (FAST) ---
#     print(f"Predicting {len(X_predict_batch)} fixtures in batch...")
#     batch_predictions = {}
    
#     for model_name in get_models().keys():
#         final_preds_for_model = np.zeros(len(X_predict_batch_numeric))
#         for pos in POSITIONS:
#             if pos not in trained_models or model_name not in trained_models[pos]:
#                 continue
#             model = trained_models[pos][model_name]
#             pos_indices = positions_series[positions_series == pos].index
#             if pos_indices.empty:
#                 continue
#             pos_X_predict = X_predict_batch_numeric.loc[pos_indices]
#             pos_preds = model.predict(pos_X_predict)
#             final_preds_for_model[pos_indices] = pos_preds
            
#         batch_predictions[model_name] = [float(max(0.0, np.round(p, 2))) for p in final_preds_for_model]
    
#     prediction_time = time.time()
#     print(f"Batch prediction finished in {prediction_time - start_time:.2f} seconds.")

#     # --- 5. RESTRUCTURE RESULTS ---
#     print("Restructuring results into final JSON format...")
#     final_player_data_map = {} 

#     for i, meta in enumerate(all_pred_metadata):
#         player_name = meta['player_name']
        
#         if player_name not in final_player_data_map:
#              # Initialize player entry
#              last_gw_info_series = latest_player_data[latest_player_data['name'] == player_name]
#              if last_gw_info_series.empty: continue
#              last_gw_info = last_gw_info_series.iloc[0]
             
#              # *** NEW: Get player status, news, and cost ***
#              status_info = player_meta_map.get(player_name, {})
#              chance_raw = status_info.get('chance_of_playing_next_round')
#              # Default to 100 if info is missing (e.g., new player)
#              chance_of_playing = 100 if pd.isna(chance_raw) else int(chance_raw)
#              news = status_info.get('news', "")
#              # Cost is in 10s (e.g., 55 -> 5.5m). Divide by 10.
#              cost = float(status_info.get('now_cost', 0)) / 10.0
             
#              player_hist = history_df[history_df['name'] == player_name].tail(5)

#              points_raw = player_hist['total_points'].fillna(0).astype(float).tolist()
#              labels_raw = player_hist['GW'].astype(int).tolist()
#              opponents_raw = []
             
#              for _, gw_row in player_hist.iterrows():
#                  opp_team_identifier = gw_row.get('opponent_team') 
#                  was_home = gw_row.get('was_home', False)
#                  opp_name_hist = "N/A"
#                  try:
#                      opp_id_int = int(opp_team_identifier)
#                      if opp_id_int in team_id_map:
#                          opp_name_hist = team_id_map[opp_id_int].get('name', 'N/A')
#                  except (ValueError, TypeError):
#                      if isinstance(opp_team_identifier, str) and opp_team_identifier in team_name_map:
#                           opp_name_hist = opp_team_identifier
                 
#                  location = " (H)" if was_home else " (A)"
#                  opponents_raw.append(f"{opp_name_hist}{location}")

#              num_missing = 5 - len(points_raw)
#              if labels_raw:
#                  start_gw = labels_raw[0]
#                  pad_labels = [f"GW{start_gw - i}" for i in range(num_missing, 0, -1) if start_gw - i > 0]
#                  pad_labels = ["Start"] * (num_missing - len(pad_labels)) + pad_labels # Handle edge case
#              else:
#                  pad_labels = [f"Start-{j}" for j in range(num_missing, 0, -1)]
             
#              pad_points = [0.0] * num_missing
#              pad_opponents = ["-"] * num_missing
#              current_labels = pad_labels + [f"GW{g}" for g in labels_raw]
#              current_points = pad_points + points_raw
#              current_opponents = pad_opponents + opponents_raw

#              final_player_data_map[player_name] = {
#                  "id": player_name,
#                  "web_name": player_name,
#                  "team": last_gw_info.get('team', 'Unknown'),
#                  "position": last_gw_info.get('position', 'UNK'),
#                  "prev_gw_points": int(last_gw_info.get('total_points', 0)),
#                  "last_5_gw_points": current_points,
#                  "last_5_gw_labels": current_labels,
#                  "last_5_gw_opponents": current_opponents,
#                  "next_opponent": "N/A",
#                  "upcoming_predictions": [],
#                  # *** NEW: Add status, news, and cost ***
#                  "chance_of_playing": chance_of_playing,
#                  "news": news,
#                  "cost": cost
#              }

#         # *** NEW: Apply prediction multiplier ***
#         multiplier = final_player_data_map[player_name]['chance_of_playing'] / 100.0
        
#         fixture_preds = {}
#         for model_name, preds in batch_predictions.items():
#             # Apply multiplier to the prediction for this fixture
#             raw_pred = preds[i]
#             adjusted_pred = raw_pred * multiplier
#             fixture_preds[model_name] = round(adjusted_pred, 2) # Round after multiplying
            
#         final_player_data_map[player_name]['upcoming_predictions'].append({
#              "gw": meta['gw'],
#              "opponent": meta['opponent'],
#              "predictions": fixture_preds
#         })
#         if meta['gw'] == next_gw:
#              final_player_data_map[player_name]['next_opponent'] = meta['opponent']

#     final_player_data_list = list(final_player_data_map.values())

#     # --- 6. SAVE THE FINAL FILE ---
#     try:
#         with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
#             json.dump(final_player_data_list, f, indent=4, ensure_ascii=False, default=str)
#         end_time = time.time()
#         print(f"\nSUCCESS: All predictions generated and saved to '{OUTPUT_JSON_FILE}'.")
#         print(f"Total time: {end_time - start_time:.2f} seconds.")
#     except TypeError as e:
#         print(f"\nError saving JSON: {e}")

# if __name__ == "__main__":
#     run_prediction_pipeline()


# import pandas as pd
# import numpy as np
# import json
# import time
# import os
# # *** UPDATED ***: Import PLAYER_RAW_URL
# from config import (
#     HISTORY_DATA_URL, FIXTURES_URL, TEAMS_URL, 
#     OUTPUT_JSON_FILE, PLAYER_RAW_URL
# )
# from feature_engineering import get_team_strength
# from model_training import load_models, get_models, POSITIONS

# def load_data(filepath, file_type):
#     """ Loads a CSV, skipping bad lines. """
#     try:
#         # Set low_memory=False to help with mixed data types
#         df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8', low_memory=False)
#         if df.empty:
#             print(f"Warning: {file_type} data from {filepath} is empty or unreadable.")
#         return df
#     except Exception as e:
#         print(f"Error loading {file_type} data from {filepath}: {e}")
#         return pd.DataFrame()

# def run_prediction_pipeline():
#     start_time = time.time()

#     # --- 1. LOAD DATA & MODELS ---
#     print("Loading data sources...")
#     history_df = load_data(HISTORY_DATA_URL, "History")
#     fixtures_df = load_data(FIXTURES_URL, "Fixtures")
#     teams_df = load_data(TEAMS_URL, "Teams")
#     # *** NEW ***: Load players_raw.csv
#     players_raw_df = load_data(PLAYER_RAW_URL, "Player Raw")

#     if history_df.empty or fixtures_df.empty or teams_df.empty or players_raw_df.empty:
#         print("One or more essential data files failed to load. Exiting.")
#         return

#     print("\nLoading pre-trained models...")
#     trained_models = load_models()
#     if not trained_models or all(len(v) == 0 for v in trained_models.values()):
#         print("Error: No pre-trained models found. Run 'python train_offline.py' first.")
#         return

#     # --- 2. PREPARE LOOKUPS & FIND NEXT GW ---
#     team_id_map, team_name_map = get_team_strength(teams_df)

#     # *** NEW ***: Create a status/news map from players_raw.csv
#     # Use 'id' (the element_id) as the key, as it's the unique identifier
#     try:
#         # Ensure 'id' column is numeric and set as index
#         players_raw_df['id'] = pd.to_numeric(players_raw_df['id'], errors='coerce')
#         players_raw_df = players_raw_df.dropna(subset=['id'])
#         players_raw_df = players_raw_df.drop_duplicates(subset=['id'], keep='first')
        
#         player_meta_map = players_raw_df.set_index('id')[
#             ['web_name', 'chance_of_playing_next_round', 'news', 'now_cost']
#         ].to_dict('index')
#     except KeyError as e:
#         print(f"Error creating player map from players_raw.csv: Missing column {e}")
#         print("Please ensure 'id', 'web_name', 'chance_of_playing_next_round', 'news', and 'now_cost' columns exist.")
#         return

#     next_gw_series = fixtures_df[fixtures_df['finished'] == False]['event']
#     if next_gw_series.empty:
#         last_played_gw = history_df['GW'].max() if 'GW' in history_df.columns and not history_df.empty else 0
#         next_gw = last_played_gw + 1 if last_played_gw > 0 else 1
#         print(f"Warning: No future GWs found. Assuming next GW is {next_gw}.")
#         if next_gw > 38: print("End of season."); return
#     else:
#         next_gw = int(next_gw_series.min())
#     print(f"Predicting for GW {next_gw} onwards.")

#     # --- 3. PREPARE BATCH FEATURE DATA ---
#     print("Preparing feature data for batch prediction...")
#     all_pred_rows_data = [] # List of feature dicts
#     all_pred_metadata = [] # List of (element_id, gw, opponent_str, position) tuples

#     # Define feature columns
#     feature_cols = [
#         'total_points_lag1', 'minutes_lag1', 'goals_scored_lag1', 'assists_lag1', 
#         'clean_sheets_lag1', 'bonus_lag1', 'bps_lag1', 'ict_index_lag1', 
#         'influence_lag1', 'creativity_lag1', 'threat_lag1',
#         'expected_goals_lag1', 'expected_assists_lag1', 'expected_goal_involvements_lag1', 'expected_goals_conceded_lag1',
#         'form_points_last_5', 'form_ict_last_5', 'form_minutes_last_5',
#         'form_xg_last_5', 'form_xa_last_5', 'form_xgi_last_5', 'form_xgc_last_5',
#         'opponent_attack_strength', 'opponent_defence_strength', 'was_home'
#     ]

#     # Pre-calculate rolling form features
#     if 'GW' not in history_df.columns and 'round' in history_df.columns: history_df['GW'] = history_df['round']
#     history_df['GW'] = pd.to_numeric(history_df['GW'], errors='coerce').fillna(0).astype(int)
#     # *** FIX ***: Use 'element' as the persistent ID for grouping
#     if 'element' not in history_df.columns:
#         print("Error: 'element' column (player ID) not found in history_df. Cannot proceed.")
#         return
#     history_df['element'] = pd.to_numeric(history_df['element'], errors='coerce').dropna()
#     history_df = history_df.sort_values(by=['element', 'GW'])

#     grouped = history_df.groupby('element')
#     history_df['form_points_last_5'] = grouped['total_points'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_ict_last_5'] = grouped['ict_index'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_minutes_last_5'] = grouped['minutes'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xg_last_5'] = grouped['expected_goals'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xa_last_5'] = grouped['expected_assists'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xgi_last_5'] = grouped['expected_goal_involvements'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
#     history_df['form_xgc_last_5'] = grouped['expected_goals_conceded'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    
#     latest_player_data = history_df.loc[history_df.groupby('element')['GW'].idxmax()]

#     # Iterate through unique players
#     for element_id, last_gw_data in latest_player_data.iterrows():
#         # element_id is now the index from .loc
#         name = last_gw_data.get('name')
#         player_team_name = last_gw_data.get('team')
#         player_pos = last_gw_data.get('position')
        
#         if not all([name, player_team_name, player_pos, player_team_name in team_name_map, player_pos in POSITIONS]):
#             continue

#         player_team_id = team_name_map[player_team_name]['id']

#         # Base features
#         base_pred_data_dict = {
#             'total_points_lag1': last_gw_data.get('total_points', 0),
#             'minutes_lag1': last_gw_data.get('minutes', 0),
#             'goals_scored_lag1': last_gw_data.get('goals_scored', 0),
#             'assists_lag1': last_gw_data.get('assists', 0),
#             'clean_sheets_lag1': last_gw_data.get('clean_sheets', 0),
#             'bonus_lag1': last_gw_data.get('bonus', 0),
#             'bps_lag1': last_gw_data.get('bps', 0),
#             'ict_index_lag1': last_gw_data.get('ict_index', 0.0),
#             'influence_lag1': last_gw_data.get('influence', 0.0),
#             'creativity_lag1': last_gw_data.get('creativity', 0.0),
#             'threat_lag1': last_gw_data.get('threat', 0.0),
#             'expected_goals_lag1': last_gw_data.get('expected_goals', 0.0),
#             'expected_assists_lag1': last_gw_data.get('expected_assists', 0.0),
#             'expected_goal_involvements_lag1': last_gw_data.get('expected_goal_involvements', 0.0),
#             'expected_goals_conceded_lag1': last_gw_data.get('expected_goals_conceded', 0.0),
#             'form_points_last_5': last_gw_data.get('form_points_last_5', 0.0),
#             'form_ict_last_5': last_gw_data.get('form_ict_last_5', 0.0),
#             'form_minutes_last_5': last_gw_data.get('form_minutes_last_5', 0.0),
#             'form_xg_last_5': last_gw_data.get('form_xg_last_5', 0.0),
#             'form_xa_last_5': last_gw_data.get('form_xa_last_5', 0.0),
#             'form_xgi_last_5': last_gw_data.get('form_xgi_last_5', 0.0),
#             'form_xgc_last_5': last_gw_data.get('form_xgc_last_5', 0.0),
#         }

#         # Find next 5 fixtures
#         player_fixtures = fixtures_df[
#             ((fixtures_df['team_h'] == player_team_id) | (fixtures_df['team_a'] == player_team_id)) &
#             (fixtures_df['event'] >= next_gw)
#         ].sort_values('event').head(5)

#         for _, fixture in player_fixtures.iterrows():
#             fixture_gw = int(fixture['event'])
#             is_home = (fixture['team_h'] == player_team_id)
#             opponent_team_id_raw = fixture['team_a'] if is_home else fixture['team_h']
#             try: opponent_team_id = int(opponent_team_id_raw) if pd.notna(opponent_team_id_raw) else None
#             except ValueError: opponent_team_id = None

#             opp_full_name, opp_att, opp_def = "Unknown", 1000, 1000
#             if opponent_team_id is not None and opponent_team_id in team_id_map:
#                 opp_strength_data = team_id_map[opponent_team_id]
#                 opp_full_name = opp_strength_data.get('name', 'Unknown')
#                 opp_att = opp_strength_data.get('strength_attack_away' if is_home else 'strength_attack_home', 1000)
#                 opp_def = opp_strength_data.get('strength_defence_away' if is_home else 'strength_defence_home', 1000)

#             current_pred_data_dict = base_pred_data_dict.copy()
#             current_pred_data_dict['opponent_attack_strength'] = opp_att
#             current_pred_data_dict['opponent_defence_strength'] = opp_def
#             current_pred_data_dict['was_home'] = 1 if is_home else 0

#             all_pred_rows_data.append(current_pred_data_dict)
#             opponent_str = f"{opp_full_name} {'(H)' if is_home else '(A)'}"
#             all_pred_metadata.append({'element_id': element_id, 'name': name, 'gw': fixture_gw, 'opponent': opponent_str, 'position': player_pos})

#     if not all_pred_rows_data:
#         print("No feature rows generated for prediction.")
#         return

#     # Convert to DataFrame
#     X_predict_batch = pd.DataFrame(all_pred_rows_data)
#     # Store metadata separately
#     metadata_df = pd.DataFrame(all_pred_metadata, index=X_predict_batch.index)
#     X_predict_batch = X_predict_batch.reindex(columns=feature_cols, fill_value=0)
#     X_predict_batch_numeric = X_predict_batch.apply(pd.to_numeric, errors='coerce').fillna(0)

#     # --- 4. BATCH PREDICT PER POSITION (FAST) ---
#     print(f"Predicting {len(X_predict_batch)} fixtures in batch...")
#     batch_predictions = {}
    
#     for model_name in get_models().keys():
#         final_preds_for_model = np.zeros(len(X_predict_batch_numeric))
#         for pos in POSITIONS:
#             if pos not in trained_models or model_name not in trained_models[pos]:
#                 continue
#             model = trained_models[pos][model_name]
#             pos_indices = metadata_df[metadata_df['position'] == pos].index
#             if pos_indices.empty:
#                 continue
#             pos_X_predict = X_predict_batch_numeric.loc[pos_indices]
#             pos_preds = model.predict(pos_X_predict)
#             final_preds_for_model[pos_indices] = pos_preds
            
#         batch_predictions[model_name] = [float(max(0.0, np.round(p, 2))) for p in final_preds_for_model]
    
#     prediction_time = time.time()
#     print(f"Batch prediction finished in {prediction_time - start_time:.2f} seconds.")

#     # --- 5. RESTRUCTURE RESULTS ---
#     print("Restructuring results into final JSON format...")
#     final_player_data_map = {} 

#     for i, meta in enumerate(all_pred_metadata):
#         element_id = meta['element_id']
        
#         if element_id not in final_player_data_map:
#              # Initialize player entry
#              # Get up to last 5 history rows for this player by element id
#              player_hist = history_df[history_df['element'] == element_id].tail(5)
#              # If no rows found by element id, try fallback by player name
#              if player_hist.empty:
#                  fallback = history_df[history_df['name'] == meta.get('name')]
#                  if not fallback.empty:
#                      player_hist = fallback.tail(5)

#              # If still empty, create a default last_gw_info dict to avoid iloc[-1] on empty
#              if not player_hist.empty:
#                  last_gw_info = player_hist.iloc[-1]
#              else:
#                  last_gw_info = {
#                      'name': meta.get('name', 'Unknown'),
#                      'team': 'Unknown',
#                      'position': 'UNK',
#                      'total_points': 0
#                  }
             
#              # *** NEW: Get player status, news, and cost ***
#              status_info = player_meta_map.get(element_id, {})
#              chance_raw = status_info.get('chance_of_playing_next_round')
#              # Default to 100 if info is missing (e.g., new player, or NaN)
#              chance_of_playing = 100 if pd.isna(chance_raw) else int(chance_raw)
#              news = status_info.get('news', "")
#              # Cost is in 10s (e.g., 55 -> 5.5m). Divide by 10.
#              cost = float(status_info.get('now_cost', 0)) / 10.0

#              points_raw = player_hist['total_points'].fillna(0).astype(float).tolist()
#              labels_raw = player_hist['GW'].astype(int).tolist()
#              opponents_raw = []
             
#              for _, gw_row in player_hist.iterrows():
#                  opp_team_identifier = gw_row.get('opponent_team') 
#                  was_home = gw_row.get('was_home', False)
#                  opp_name_hist = "N/A"
#                  try:
#                      opp_id_int = int(opp_team_identifier)
#                      if opp_id_int in team_id_map:
#                          opp_name_hist = team_id_map[opp_id_int].get('name', 'N/A')
#                  except (ValueError, TypeError):
#                      if isinstance(opp_team_identifier, str) and opp_team_identifier in team_name_map:
#                           opp_name_hist = opp_team_identifier
                 
#                  location = " (H)" if was_home else " (A)"
#                  opponents_raw.append(f"{opp_name_hist}{location}")

#              num_missing = 5 - len(points_raw)
#              if labels_raw:
#                  start_gw = labels_raw[0]
#                  pad_labels = [f"GW{start_gw - i}" for i in range(num_missing, 0, -1) if start_gw - i > 0]
#                  pad_labels = ["Start"] * (num_missing - len(pad_labels)) + pad_labels
#              else:
#                  pad_labels = [f"Start-{j}" for j in range(num_missing, 0, -1)]
             
#              pad_points = [0.0] * num_missing
#              pad_opponents = ["-"] * num_missing
#              current_labels = pad_labels + [f"GW{g}" for g in labels_raw]
#              current_points = pad_points + points_raw
#              current_opponents = pad_opponents + opponents_raw

#              final_player_data_map[element_id] = {
#                  "id": element_id,
#                  "web_name": last_gw_info.get('name', 'Unknown'),
#                  "team": last_gw_info.get('team', 'Unknown'),
#                  "position": last_gw_info.get('position', 'UNK'),
#                  "prev_gw_points": int(last_gw_info.get('total_points', 0)),
#                  "last_5_gw_points": current_points,
#                  "last_5_gw_labels": current_labels,
#                  "last_5_gw_opponents": current_opponents,
#                  "next_opponent": "N/A",
#                  "upcoming_predictions": [],
#                  # *** NEW: Add status, news, and cost ***
#                  "chance_of_playing": chance_of_playing,
#                  "news": news,
#                  "cost": cost
#              }

#         # *** NEW: Apply prediction multiplier ***
#         # Only apply multiplier for the *next* gameweek
#         multiplier = 1.0 # Default
#         if meta['gw'] == next_gw:
#             multiplier = final_player_data_map[element_id]['chance_of_playing'] / 100.0
        
#         fixture_preds = {}
#         for model_name, preds in batch_predictions.items():
#             raw_pred = preds[i]
#             adjusted_pred = raw_pred * multiplier
#             fixture_preds[model_name] = round(adjusted_pred, 2)
            
#         final_player_data_map[element_id]['upcoming_predictions'].append({
#              "gw": meta['gw'],
#              "opponent": meta['opponent'],
#              "predictions": fixture_preds
#         })
#         if meta['gw'] == next_gw:
#              final_player_data_map[element_id]['next_opponent'] = meta['opponent']

#     final_player_data_list = list(final_player_data_map.values())

#     # --- 6. SAVE THE FINAL FILE ---
#     try:
#         with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
#             json.dump(final_player_data_list, f, indent=4, ensure_ascii=False, default=str)
#         end_time = time.time()
#         print(f"\nSUCCESS: All predictions generated and saved to '{OUTPUT_JSON_FILE}'.")
#         print(f"Total time: {end_time - start_time:.2f} seconds.")
#     except TypeError as e:
#         print(f"\nError saving JSON: {e}")

# if __name__ == "__main__":
#     run_prediction_pipeline()



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
    """ Loads a CSV, skipping bad lines. """
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

    # --- 1. LOAD DATA & MODELS ---
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

    # --- 2. PREPARE LOOKUPS & FIND NEXT GW ---
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

    # --- 3. PREPARE BATCH FEATURE DATA ---
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

    # --- 4. BATCH PREDICT PER POSITION (FAST) ---
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

    # --- 5. RESTRUCTURE RESULTS ---
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
             
             # *** FIX: Check if news is NaN and replace with an empty string ***
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

    # --- 6. SAVE THE FINAL FILE ---
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