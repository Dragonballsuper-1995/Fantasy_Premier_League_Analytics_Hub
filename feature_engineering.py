import pandas as pd
import numpy as np

def get_team_strength(teams_df):
    team_map_id = teams_df.set_index('id')[[
        'name', 'short_name', 
        'strength_attack_home', 'strength_attack_away', 
        'strength_defence_home', 'strength_defence_away'
    ]].to_dict('index')
    
    team_map_name = teams_df.set_index('name')[[
        'id', 'short_name', 
        'strength_attack_home', 'strength_attack_away', 
        'strength_defence_home', 'strength_defence_away'
    ]].to_dict('index')

    return team_map_id, team_map_name

def create_features(df, team_strength_name_map):
    df = df.sort_values(by=['name', 'GW']).copy()
    
    grouped = df.groupby('name')
    
    df['target_points'] = grouped['total_points'].shift(-1)
    
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