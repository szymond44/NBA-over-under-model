import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os


# Define the 14 features used for both training and inference.
FEATURES = [
    'home_off_rating_pre', 'home_def_rating_pre', 'home_pace_rating_pre', 'home_rest_days',
    'away_off_rating_pre', 'away_def_rating_pre', 'away_pace_rating_pre', 'away_rest_days',
    'home_roll_pts', 'home_roll_pace', 'home_roll_win',
    'away_roll_pts', 'away_roll_pace', 'away_roll_win'
]

def load_data(input_file='nba_features_with_rolling.csv'):
    """Loads the final feature set from the data folder."""
    
    # Path setup: assumes file is in data/ folder relative to project root
    base_dir = 'data'
    input_path = os.path.join(base_dir, input_file)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find {input_path}. \n"
                                f"Please ensure the feature engineering step was completed.")

    df = pd.read_csv(input_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    # Sort is critical for chronological integrity
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    return df


def train_and_evaluate(df, start_date='2023-10-24'):
    """
    Performs Walk-Forward Validation (retrains monthly) to establish model accuracy.
    Returns the two final trained models.
    """
    
    # Define monthly windows for retraining
    start_date = pd.Timestamp(start_date)
    dates = pd.date_range(start=start_date, end=df['GAME_DATE'].max(), freq='MS')
    
    # Model Hyperparameters
    params = {
        'objective': 'reg:squarederror',
    
        'n_estimators': 1200,    
        'learning_rate': 0.02,    
        'max_depth': 4,            
        'min_child_weight': 3,    
        'gamma': 2,                   
        'subsample': 0.7,         
        'colsample_bytree': 0.7,   
        'n_jobs': -1
    }
    
    all_preds = []
    final_model_home = None
    final_model_away = None

    # Loop through each window (retrain on past, predict on next month)
    for i in range(len(dates) - 1):
        train_end = dates[i]
        
        # Split: ensure no future data leaks into training set
        train_df = df[df['GAME_DATE'] < train_end]
        test_df = df[(df['GAME_DATE'] >= train_end) & (df['GAME_DATE'] < dates[i+1])].copy()
        
        if len(test_df) == 0: continue
        
        # 1. Train Twin Regressors
        xgb_home = xgb.XGBRegressor(**params)
        xgb_home.fit(train_df[FEATURES], train_df['PTS_home'])
        
        xgb_away = xgb.XGBRegressor(**params)
        xgb_away.fit(train_df[FEATURES], train_df['PTS_away'])
        
        # 2. Predict and Store
        test_df['pred_home'] = xgb_home.predict(test_df[FEATURES])
        test_df['pred_away'] = xgb_away.predict(test_df[FEATURES])
        all_preds.append(test_df)
        
        # Keep the latest trained models for live prediction (the Oracle)
        final_model_home = xgb_home
        final_model_away = xgb_away
        
    # Final Metrics
    full_res = pd.concat(all_preds)
    full_res['pred_total'] = full_res['pred_home'] + full_res['pred_away']
    rmse_total = np.sqrt(mean_squared_error(full_res['PTS_home'] + full_res['PTS_away'], full_res['pred_total']))
    print(f'The model rmse is {rmse_total}')
    return final_model_home, final_model_away, rmse_total


class NBAOracle:
    """
    The Inference Engine: Holds the trained models and provides a live prediction API.
    """
    def __init__(self, df, model_home, model_away):
        self.df = df
        self.mh = model_home
        self.ma = model_away
        
        # Index is built only once at initialization
        self.index = self._build_index()
        
    def _build_index(self):
        """Builds a dictionary lookup of the most recent features for every team."""
        cols = ['GAME_DATE', 'TEAM', 'OFF', 'DEF', 'PACE', 'ROLL_PTS', 'ROLL_PACE', 'ROLL_WIN']
        
        # Stack stats (Home and Away) to get latest history for ALL teams
        h = self.df[['GAME_DATE', 'TEAM_NAME_home', 'home_off_rating_pre', 'home_def_rating_pre', 'home_pace_rating_pre', 'home_roll_pts', 'home_roll_pace', 'home_roll_win']].copy()
        h.columns = cols
        
        a = self.df[['GAME_DATE', 'TEAM_NAME_away', 'away_off_rating_pre', 'away_def_rating_pre', 'away_pace_rating_pre', 'away_roll_pts', 'away_roll_pace', 'away_roll_win']].copy()
        a.columns = cols
        
        return pd.concat([h, a]).sort_values('GAME_DATE').groupby('TEAM').last().to_dict('index')
    
    def predict(self, home, away):
        """Generates the total score prediction for a given matchup."""
        
        if home not in self.index or away not in self.index:
            return "Error: Team not found in historical data."
            
        h_stats, a_stats = self.index[home], self.index[away]
        today = pd.to_datetime('today')
        
        # Calculate Rest Days based on the last game date found in history
        h_rest = min((today - h_stats['GAME_DATE']).days, 7)
        a_rest = min((today - a_stats['GAME_DATE']).days, 7)
        
        # Construct Feature Vector (Must match the features list order)
        input_data = {
            'home_off_rating_pre': [h_stats['OFF']], 'home_def_rating_pre': [h_stats['DEF']],
            'home_pace_rating_pre': [h_stats['PACE']], 'home_rest_days': [h_rest],
            'away_off_rating_pre': [a_stats['OFF']], 'away_def_rating_pre': [a_stats['DEF']],
            'away_pace_rating_pre': [a_stats['PACE']], 'away_rest_days': [a_rest],
            'home_roll_pts': [h_stats['ROLL_PTS']], 'home_roll_pace': [h_stats['ROLL_PACE']], 'home_roll_win': [h_stats['ROLL_WIN']],
            'away_roll_pts': [a_stats['ROLL_PTS']], 'away_roll_pace': [a_stats['ROLL_PACE']], 'away_roll_win': [a_stats['ROLL_WIN']]
        }
        
        # 3. Predict and Output
        ph = self.mh.predict(pd.DataFrame(input_data))[0]
        pa = self.ma.predict(pd.DataFrame(input_data))[0]
        print(f"Projected: {home} {ph:.1f} - {pa:.1f} {away} (Total: {ph+pa:.1f})")