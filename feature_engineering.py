# import pandas as pd
# import numpy as np

# def get_team_strength(teams_df):
#     """
#     Creates a dictionary mapping team ID to its strength ratings.
#     """
#     # Use 'id' and 'name' from teams.csv as the key
#     team_map_id = teams_df.set_index('id')[['name', 'strength_attack_home', 'strength_attack_away', 'strength_defence_home', 'strength_defence_away']].to_dict('index')
    
#     # Create a map based on team name as well, for the vaastav data
#     team_map_name = teams_df.set_index('name')[['id', 'strength_attack_home', 'strength_attack_away', 'strength_defence_home', 'strength_defence_away']].to_dict('index')

#     return team_map_id, team_map_name

# def create_features(df, team_strength_name_map):
#     """
#     Creates lag and opponent strength features for modeling.
#     We "lag" the data by 1 GW, meaning we use stats from GW(N-1) to predict points for GW(N).
#     """
    
#     # Sort data by player name and then gameweek, which is crucial for lag features
#     df = df.sort_values(by=['name', 'GW']).copy()
    
#     # Group by player
#     grouped = df.groupby('name')
    
#     # 1. Target Variable (y): The points in the *next* gameweek
#     df['target_points'] = grouped['total_points'].shift(-1)
    
#     # 2. Lagged Features (X): Stats from the *current* gameweek (to predict the next)
#     features_to_lag = [
#         'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
#         'bonus', 'bps', 'ict_index', 'influence', 'creativity', 'threat'
#     ]
    
#     for col in features_to_lag:
#         df[f'{col}_lag1'] = grouped[col].shift(1)

#     # 3. Rolling Form Features (5-week rolling average, lagged)
#     df['form_points_last_5'] = grouped['total_points'].shift(1).rolling(window=5, min_periods=1).mean()
#     df['form_ict_last_5'] = grouped['ict_index'].shift(1).rolling(window=5, min_periods=1).mean()
#     df['form_minutes_last_5'] = grouped['minutes'].shift(1).rolling(window=5, min_periods=1).mean()

#     # 4. NEW Opponent Strength Features
#     # 'opponent_team' in vaastav's data is the team name string (e.g., "Arsenal")
#     # We map this to the strength ratings from teams.csv
    
#     def get_opponent_strength(row, home_or_away):
#         team_name = row['opponent_team']
#         if team_name not in team_strength_name_map:
#             return 1000 # Default for unknown teams (e.g., promoted)
        
#         strength_type = f'strength_{home_or_away}' # e.g., 'strength_attack_home'
#         return team_strength_name_map[team_name][strength_type]

#     # If 'was_home' is True, the opponent was AWAY
#     df['opponent_attack_strength'] = df.apply(lambda row: get_opponent_strength(row, 'attack_away' if row['was_home'] else 'attack_home'), axis=1)
#     df['opponent_defence_strength'] = df.apply(lambda row: get_opponent_strength(row, 'defence_away' if row['was_home'] else 'defence_home'), axis=1)
#     df['was_home'] = df['was_home'].astype(int)

#     # Our final feature list
#     feature_cols = [
#         'total_points_lag1', 'minutes_lag1', 'goals_scored_lag1', 'assists_lag1', 
#         'clean_sheets_lag1', 'bonus_lag1', 'bps_lag1', 'ict_index_lag1', 
#         'influence_lag1', 'creativity_lag1', 'threat_lag1',
#         'form_points_last_5', 'form_ict_last_5', 'form_minutes_last_5',
#         'opponent_attack_strength', 'opponent_defence_strength', 'was_home'
#     ]

#     # Drop rows with missing values
#     df_clean = df.dropna(subset=['target_points'] + feature_cols)

#     X = df_clean[feature_cols]
#     y = df_clean['target_points']
    
#     return X, y, df


# import pandas as pd
# import numpy as np

# def get_team_strength(teams_df):
#     """
#     Creates a dictionary mapping team ID to its strength ratings and name/short_name.
#     """
#     # Use 'id' from teams.csv as the key
#     team_map_id = teams_df.set_index('id')[[
#         'name', 'short_name', 
#         'strength_attack_home', 'strength_attack_away', 
#         'strength_defence_home', 'strength_defence_away'
#     ]].to_dict('index')
    
#     # Create a map based on team name as well, for the vaastav data
#     team_map_name = teams_df.set_index('name')[[
#         'id', 'short_name', 
#         'strength_attack_home', 'strength_attack_away', 
#         'strength_defence_home', 'strength_defence_away'
#     ]].to_dict('index')

#     return team_map_id, team_map_name

# def create_features(df, team_strength_name_map):
#     """
#     Creates lag and opponent strength features for modeling.
#     We "lag" the data by 1 GW, meaning we use stats from GW(N-1) to predict points for GW(N).
#     """
    
#     # Sort data by player name and then gameweek, which is crucial for lag features
#     df = df.sort_values(by=['name', 'GW']).copy()
    
#     # Group by player
#     grouped = df.groupby('name')
    
#     # 1. Target Variable (y): The points in the *next* gameweek
#     df['target_points'] = grouped['total_points'].shift(-1)
    
#     # 2. Lagged Features (X): Stats from the *current* gameweek (to predict the next)
#     # *** UPDATED ***: Added all expected stats
#     features_to_lag = [
#         'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
#         'bonus', 'bps', 'ict_index', 'influence', 'creativity', 'threat',
#         'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded'
#     ]
    
#     for col in features_to_lag:
#         # Check if column exists, some older data might not have it
#         if col in df.columns:
#             df[f'{col}_lag1'] = grouped[col].shift(1)
#         else:
#             df[f'{col}_lag1'] = 0 # Default to 0 if column is missing

#     # 3. Rolling Form Features (5-week rolling average, lagged)
#     df['form_points_last_5'] = grouped['total_points'].shift(1).rolling(window=5, min_periods=1).mean()
#     df['form_ict_last_5'] = grouped['ict_index'].shift(1).rolling(window=5, min_periods=1).mean()
#     df['form_minutes_last_5'] = grouped['minutes'].shift(1).rolling(window=5, min_periods=1).mean()
    
#     # *** NEW ***: Add rolling form for expected stats
#     if 'expected_goals' in df.columns:
#         df['form_xg_last_5'] = grouped['expected_goals'].shift(1).rolling(window=5, min_periods=1).mean()
#     if 'expected_assists' in df.columns:
#         df['form_xa_last_5'] = grouped['expected_assists'].shift(1).rolling(window=5, min_periods=1).mean()
#     if 'expected_goal_involvements' in df.columns:
#         df['form_xgi_last_5'] = grouped['expected_goal_involvements'].shift(1).rolling(window=5, min_periods=1).mean()
#     if 'expected_goals_conceded' in df.columns:
#         df['form_xgc_last_5'] = grouped['expected_goals_conceded'].shift(1).rolling(window=5, min_periods=1).mean()


#     # 4. Opponent Strength Features based on historical opponent
#     def get_opponent_strength(row, home_or_away, strength_metric):
#         team_name = row['opponent_team']
#         # Handle cases where team name might be an ID or string
#         team_data = None
        
#         # Try to convert to int, as opponent_team might be an ID
#         try:
#             team_id = int(team_name)
#             # Create a reverse map from ID to Name for this lookup
#             id_to_name_map = {v['id']: k for k, v in team_strength_name_map.items()}
#             if team_id in id_to_name_map:
#                 team_name = id_to_name_map[team_id] # Convert ID to name
#         except (ValueError, TypeError):
#              pass # Was not an int, assume it's already a name

#         if isinstance(team_name, str) and team_name in team_strength_name_map:
#             team_data = team_strength_name_map[team_name]
        
#         if not team_data:
#             return 1000 # Default for unknown teams

#         strength_key = f'strength_{strength_metric}_{home_or_away}'
#         return team_data.get(strength_key, 1000)

#     df['opponent_attack_strength'] = df.apply(lambda row: get_opponent_strength(row, 'away' if row['was_home'] else 'home', 'attack'), axis=1)
#     df['opponent_defence_strength'] = df.apply(lambda row: get_opponent_strength(row, 'away' if row['was_home'] else 'home', 'defence'), axis=1)
#     df['was_home'] = df['was_home'].astype(int)

#     # *** UPDATED ***: Our final feature list now includes all new features
#     feature_cols = [
#         'total_points_lag1', 'minutes_lag1', 'goals_scored_lag1', 'assists_lag1', 
#         'clean_sheets_lag1', 'bonus_lag1', 'bps_lag1', 'ict_index_lag1', 
#         'influence_lag1', 'creativity_lag1', 'threat_lag1',
#         'expected_goals_lag1', 'expected_assists_lag1', 'expected_goal_involvements_lag1', 'expected_goals_conceded_lag1',
#         'form_points_last_5', 'form_ict_last_5', 'form_minutes_last_5',
#         'form_xg_last_5', 'form_xa_last_5', 'form_xgi_last_5', 'form_xgc_last_5',
#         'opponent_attack_strength', 'opponent_defence_strength', 'was_home'
#     ]
    
#     # Ensure all columns exist, fill missing ones with 0
#     for col in feature_cols:
#         if col not in df.columns:
#             df[col] = 0.0

#     # Drop rows with missing target or critical lags
#     df_clean = df.dropna(subset=['target_points', 'total_points_lag1', 'minutes_lag1']).copy()

#     X = df_clean[feature_cols]
#     y = df_clean['target_points']
    
#     # Return df_clean which has the 'position' column needed for training
#     return X, y, df_clean


import pandas as pd
import numpy as np

def get_team_strength(teams_df):
    """
    Creates a dictionary mapping team ID to its strength ratings and name/short_name.
    """
    # Use 'id' from teams.csv as the key
    team_map_id = teams_df.set_index('id')[[
        'name', 'short_name', 
        'strength_attack_home', 'strength_attack_away', 
        'strength_defence_home', 'strength_defence_away'
    ]].to_dict('index')
    
    # Create a map based on team name as well, for the vaastav data
    team_map_name = teams_df.set_index('name')[[
        'id', 'short_name', 
        'strength_attack_home', 'strength_attack_away', 
        'strength_defence_home', 'strength_defence_away'
    ]].to_dict('index')

    # Return the map keyed by ID, and the map keyed by name
    return team_map_id, team_map_name

def create_features(df, team_strength_name_map):
    """
    Creates lag and opponent strength features for modeling.
    We "lag" the data by 1 GW, meaning we use stats from GW(N-1) to predict points for GW(N).
    """
    
    # Sort data by player name and then gameweek, which is crucial for lag features
    df = df.sort_values(by=['name', 'GW']).copy()
    
    # Group by player
    grouped = df.groupby('name')
    
    # 1. Target Variable (y): The points in the *next* gameweek
    df['target_points'] = grouped['total_points'].shift(-1)
    
    # 2. Lagged Features (X): Stats from the *current* gameweek (to predict the next)
    features_to_lag = [
        'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bonus', 'bps', 'ict_index', 'influence', 'creativity', 'threat',
        'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded'
    ]
    
    for col in features_to_lag:
        if col in df.columns:
            df[f'{col}_lag1'] = grouped[col].shift(1)
        else:
            df[f'{col}_lag1'] = 0 

    # 3. Rolling Form Features (5-week rolling average, lagged)
    df['form_points_last_5'] = grouped['total_points'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    df['form_ict_last_5'] = grouped['ict_index'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    df['form_minutes_last_5'] = grouped['minutes'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    
    if 'expected_goals' in df.columns:
        df['form_xg_last_5'] = grouped['expected_goals'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    if 'expected_assists' in df.columns:
        df['form_xa_last_5'] = grouped['expected_assists'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    if 'expected_goal_involvements' in df.columns:
        df['form_xgi_last_5'] = grouped['expected_goal_involvements'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    if 'expected_goals_conceded' in df.columns:
        df['form_xgc_last_5'] = grouped['expected_goals_conceded'].shift(1).rolling(window=5, min_periods=1).mean().fillna(0)


    # 4. Opponent Strength Features based on historical opponent
    def get_opponent_strength(row, home_or_away, strength_metric):
        team_name = row['opponent_team']
        team_data = None
        
        try:
            team_id = int(team_name)
            id_to_name_map = {v['id']: k for k, v in team_strength_name_map.items()}
            if team_id in id_to_name_map:
                team_name = id_to_name_map[team_id]
        except (ValueError, TypeError):
             pass 

        if isinstance(team_name, str) and team_name in team_strength_name_map:
            team_data = team_strength_name_map[team_name]
        
        if not team_data:
            return 1000 

        strength_key = f'strength_{strength_metric}_{home_or_away}'
        return team_data.get(strength_key, 1000)

    df['opponent_attack_strength'] = df.apply(lambda row: get_opponent_strength(row, 'away' if row['was_home'] else 'home', 'attack'), axis=1)
    df['opponent_defence_strength'] = df.apply(lambda row: get_opponent_strength(row, 'away' if row['was_home'] else 'home', 'defence'), axis=1)
    df['was_home'] = df['was_home'].astype(int)

    feature_cols = [
        'total_points_lag1', 'minutes_lag1', 'goals_scored_lag1', 'assists_lag1', 
        'clean_sheets_lag1', 'bonus_lag1', 'bps_lag1', 'ict_index_lag1', 
        'influence_lag1', 'creativity_lag1', 'threat_lag1',
        'expected_goals_lag1', 'expected_assists_lag1', 'expected_goal_involvements_lag1', 'expected_goals_conceded_lag1',
        'form_points_last_5', 'form_ict_last_5', 'form_minutes_last_5',
        'form_xg_last_5', 'form_xa_last_5', 'form_xgi_last_5', 'form_xgc_last_5',
        'opponent_attack_strength', 'opponent_defence_strength', 'was_home'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df_clean = df.dropna(subset=['target_points', 'total_points_lag1', 'minutes_lag1']).copy()

    X = df_clean[feature_cols]
    y = df_clean['target_points']
    
    return X, y, df_clean