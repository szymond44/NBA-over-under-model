import pandas as pd
import numpy as np
import os

def elo_model(input_csv_name='nba_features.csv', output_csv_name='nba_features_ready_for_model.csv', k_factor = 0.15, reversion = 0.01, base_elo = 1000):
    # Setup paths
    base_dir = 'data'
    input_path = os.path.join(base_dir, input_csv_name)
    output_path = os.path.join(base_dir, output_csv_name)

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find {input_path}. \n"
                                f"Make sure the file exists in the '{base_dir}' folder.")

    df = pd.read_csv(input_path)

    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        # Sort by date to ensure Elo is calculated chronologically
        df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)

    # Elo configuration 
    K_FACTOR = k_factor
    REVERSION = reversion
    BASE_ELO = base_elo

    # State Dictionaries
    elo_off = {}
    elo_def = {}
    elo_pace = {}

    # Lists to store features
    home_off_pre, home_def_pre, home_pace_pre = [], [], []
    away_off_pre, away_def_pre, away_pace_pre = [], [], []

    # History for Reversion
    history_off = [BASE_ELO]
    history_def = [BASE_ELO]
    history_pace = [BASE_ELO]

    # Helper function to prevent explosion
    def safe_update(old_rating, actual, expected, k, reversion, league_avg):
        error = actual - expected
        new_rating = old_rating + (k * error)
        final_rating = (new_rating * (1 - reversion)) + (league_avg * reversion)

        # SAFETY CLAMP: If rating goes crazy, reset to baseline
        if np.isnan(final_rating) or final_rating > 3000 or final_rating < 0:
            return 1000.0
        return final_rating

    # Elo loop
    for idx, row in df.iterrows():
        home = row['TEAM_NAME_home']
        away = row['TEAM_NAME_away']
        
        # Get current ratings
        h_off = elo_off.get(home, BASE_ELO)
        h_def = elo_def.get(home, BASE_ELO)
        h_pace = elo_pace.get(home, BASE_ELO)
        
        a_off = elo_off.get(away, BASE_ELO)
        a_def = elo_def.get(away, BASE_ELO)
        a_pace = elo_pace.get(away, BASE_ELO)
        
        # Store features (pre-game state)
        home_off_pre.append(h_off)
        home_def_pre.append(h_def)
        home_pace_pre.append(h_pace)
        away_off_pre.append(a_off)
        away_def_pre.append(a_def)
        away_pace_pre.append(a_pace)
        
        # Prepare averages from last 1000 games (to keep things dynamic)
        avg_off = np.mean(history_off[-1000:])
        avg_def = np.mean(history_def[-1000:])
        avg_pace = np.mean(history_pace[-1000:]) 
        
        # Update
        h_perf = row['OFF_EFF_home_actual']
        a_perf = row['OFF_EFF_away_actual']
        game_pace = row['PACE_actual']
        
        # Logic A: Home Off / Away Def
        exp_h_off = h_off + (a_def - avg_def)
        elo_off[home] = safe_update(h_off, h_perf, exp_h_off, K_FACTOR, REVERSION, avg_off)
        elo_def[away] = safe_update(a_def, h_perf, exp_h_off, K_FACTOR, REVERSION, avg_def)

        # Logic B: Away Off / Home Def
        exp_a_off = a_off + (h_def - avg_def)
        elo_off[away] = safe_update(a_off, a_perf, exp_a_off, K_FACTOR, REVERSION, avg_off)
        elo_def[home] = safe_update(h_def, a_perf, exp_a_off, K_FACTOR, REVERSION, avg_def)
        
        # Logic C: Pace
        exp_pace = (h_pace + a_pace) / 2
        elo_pace[home] = safe_update(h_pace, game_pace, exp_pace, K_FACTOR, REVERSION, avg_pace)
        elo_pace[away] = safe_update(a_pace, game_pace, exp_pace, K_FACTOR, REVERSION, avg_pace)
        
        # Update history
        history_off.extend([elo_off[home], elo_off[away]])
        history_def.extend([elo_def[home], elo_def[away]])
        history_pace.extend([elo_pace[home], elo_pace[away]])

    # Save the data
    df['home_off_rating_pre'] = home_off_pre
    df['home_def_rating_pre'] = home_def_pre
    df['home_pace_rating_pre'] = home_pace_pre

    df['away_off_rating_pre'] = away_off_pre
    df['away_def_rating_pre'] = away_def_pre
    df['away_pace_rating_pre'] = away_pace_pre

    df.to_csv(output_path, index=False)

