import pandas as pd
import numpy as np
import os

def add_rolling_stats(input_csv_name='nba_features_ready_for_model.csv', output_csv_name='nba_features_with_rolling.csv'):
    base_dir = 'data'
    input_path = os.path.join(base_dir, input_csv_name)
    output_path = os.path.join(base_dir, output_csv_name)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find {input_path}. \n"
                                f"Make sure you ran the Elo module first.")

    df = pd.read_csv(input_path)
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Stack home and away games for correct analisys
    # This is done to ensure the rolling averages are calculated across both home and away games
    
    # Extract only relevant columns for both home and away
    home_df = df[['GAME_ID', 'GAME_DATE', 'TEAM_NAME_home', 'PTS_home', 'PACE_actual', 'WL_home']].copy()
    home_df.columns = ['GAME_ID', 'GAME_DATE', 'TEAM_NAME', 'PTS', 'PACE', 'WL']
    home_df['IS_HOME'] = 1

    away_df = df[['GAME_ID', 'GAME_DATE', 'TEAM_NAME_away', 'PTS_away', 'PACE_actual', 'WL_away']].copy()
    away_df.columns = ['GAME_ID', 'GAME_DATE', 'TEAM_NAME', 'PTS', 'PACE', 'WL']
    away_df['IS_HOME'] = 0

    team_logs = pd.concat([home_df, away_df], axis=0)
    team_logs = team_logs.sort_values(['TEAM_NAME', 'GAME_DATE']).reset_index(drop=True)

    # Convert W/L to numeric (1/0)
    team_logs['WIN_FLG'] = team_logs['WL'].apply(lambda x: 1 if x == 'W' else 0)

    # Calculate rolling stats
    # Group by team and shift(1) to ensure current game data is not used (which would be data leakage)
    grouped = team_logs.groupby('TEAM_NAME')

    team_logs['roll_pts_5'] = grouped['PTS'].transform(lambda x: x.shift(1).rolling(5).mean())
    team_logs['roll_pace_5'] = grouped['PACE'].transform(lambda x: x.shift(1).rolling(5).mean())
    team_logs['roll_win_pct_5'] = grouped['WIN_FLG'].transform(lambda x: x.shift(1).rolling(5).mean())

    # Fill NaN (first 5 games) with conservative estimates
    team_logs['roll_pts_5'] = team_logs['roll_pts_5'].fillna(112.0)
    team_logs['roll_pace_5'] = team_logs['roll_pace_5'].fillna(98.0)
    team_logs['roll_win_pct_5'] = team_logs['roll_win_pct_5'].fillna(0.50)

    # Merge it back to main df
    df = df.merge(
        team_logs[['GAME_ID', 'TEAM_NAME', 'roll_pts_5', 'roll_pace_5', 'roll_win_pct_5']],
        left_on=['GAME_ID', 'TEAM_NAME_home'],
        right_on=['GAME_ID', 'TEAM_NAME'],
        how='left'
    )
    df.rename(columns={
        'roll_pts_5': 'home_roll_pts', 
        'roll_pace_5': 'home_roll_pace', 
        'roll_win_pct_5': 'home_roll_win'
    }, inplace=True)
    df.drop(columns=['TEAM_NAME'], inplace=True)

    df = df.merge(
        team_logs[['GAME_ID', 'TEAM_NAME', 'roll_pts_5', 'roll_pace_5', 'roll_win_pct_5']],
        left_on=['GAME_ID', 'TEAM_NAME_away'],
        right_on=['GAME_ID', 'TEAM_NAME'],
        how='left'
    )
    df.rename(columns={
        'roll_pts_5': 'away_roll_pts', 
        'roll_pace_5': 'away_roll_pace', 
        'roll_win_pct_5': 'away_roll_win'
    }, inplace=True)
    df.drop(columns=['TEAM_NAME'], inplace=True)

    df.to_csv(output_path, index=False)    
    return df
