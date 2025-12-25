import pandas as pd
import numpy as np
import os

def load_and_process_data(input_file='nba_games_2019_2025.csv', output_file='nba_features.csv'):
    base_dir = 'data'
    
    input_path = os.path.join(base_dir, input_file)
    output_path = os.path.join(base_dir, output_file)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find {input_path}. \n"
                                f"Make sure you ran 'main.py' from the project root "
                                f"and that the file exists in the 'data' folder.")
    
    # Load and sort the games data to ensure no data leakage will occur 
    df = pd.read_csv(input_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)

    # Estimate possesions from the standard NBA formula
    df['POSS_home'] = df['FGA_home'] + 0.44 * df['FTA_home'] - df['OREB_home'] + df['TOV_home']
    df['POSS_away'] = df['FGA_away'] + 0.44 * df['FTA_away'] - df['OREB_away'] + df['TOV_away']

    # Estimate game pace 
    # Pace of the game is derived from the formula 48 * (pace of home and away) / mins of the game
    # Since low amount of games goes to OT I will approximate this dataset as if no game went into OT (so all games last standard 48 minutes)
    df['GAME_PACE'] = 48 * ((df['POSS_home'] + df['POSS_away']) / 2) / 48

    # Clean infinity / NaN
    df['POSS_home'] = df['POSS_home'].replace(0, np.nan)
    df['POSS_away'] = df['POSS_away'].replace(0, np.nan)

    # Calculate efficiency
    df['OFF_EFF_home_actual'] = (df['PTS_home'] / df['POSS_home']) * 1000
    df['OFF_EFF_away_actual'] = (df['PTS_away'] / df['POSS_away']) * 1000
    df['PACE_actual'] = df['GAME_PACE'] * 10 

    # Filter bad data to avoid logical errors (so if efficiencies and pace are abnormaly high due to bad data)
    cols_to_check = ['OFF_EFF_home_actual', 'OFF_EFF_away_actual', 'PACE_actual']
    df.dropna(subset=cols_to_check, inplace=True)

    mask_valid = (
        (df['OFF_EFF_home_actual'] < 3000) & (df['OFF_EFF_home_actual'] > 10) &
        (df['OFF_EFF_away_actual'] < 3000) & (df['OFF_EFF_away_actual'] > 10) &
        (df['PACE_actual'] < 2000) & (df['PACE_actual'] > 500)
    )
    df = df[mask_valid].reset_index(drop=True)

    # Calculate rest days
    last_game_date = {}
    home_rest_days = []
    away_rest_days = []

    for idx, row in df.iterrows():
        home, away, date = row['TEAM_NAME_home'], row['TEAM_NAME_away'], row['GAME_DATE']
        
        # Home Rest
        if home in last_game_date:
            delta = (date - last_game_date[home]).days
            home_rest_days.append(min(delta, 7))
        else:
            home_rest_days.append(3) # Set default to 3 rest days 
            
        # Away Rest
        if away in last_game_date:
            delta = (date - last_game_date[away]).days
            away_rest_days.append(min(delta, 7))
        else:
            away_rest_days.append(3)
            
        last_game_date[home] = date
        last_game_date[away] = date

    df['home_rest_days'] = home_rest_days
    df['away_rest_days'] = away_rest_days

    # Save the file. Return the dataframe so the next step (Elo) can use it directly in memory,
    # but  also save a copy for debugging.
    df.to_csv(output_path, index=False)
    return df