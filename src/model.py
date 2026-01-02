import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

FEATURES = [
    'home_off_rating_pre', 'home_def_rating_pre', 'home_pace_rating_pre', 'home_rest_days',
    'away_off_rating_pre', 'away_def_rating_pre', 'away_pace_rating_pre', 'away_rest_days',
    'home_roll_pts', 'home_roll_pace', 'home_roll_win',
    'away_roll_pts', 'away_roll_pace', 'away_roll_win'
]

def load_data(input_file='nba_features_with_rolling.csv'):
    base_dir = 'data'
    input_path = os.path.join(base_dir, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find {input_path}.")
    df = pd.read_csv(input_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    return df

def get_params(mode):
    if mode == 'conservative':
        # Conservative: safe, smoothed parameters
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 1200, 
            'learning_rate': 0.02, 
            'max_depth': 4,
            'min_child_weight': 3, 
            'gamma': 2, 
            'n_jobs': -1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
    elif mode == 'chaos':
        # Chaos: volatile, over-reactive parameters
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 2000, 
            'learning_rate': 0.05,  
            'max_depth': 8,         
            'min_child_weight': 0,  
            'gamma': 0,             
            'subsample': 1.0,      
            'n_jobs': -1
        }

def train_specific_model(df, dates, params, mode='normal'):
    all_preds = []
    final_mh = None
    final_ma = None

    for i in range(len(dates) - 1):
        train_end = dates[i]
        if mode == 'chaos':
            # Chaos only eats data from Jan 2024 onwards.
            cutoff_date = pd.Timestamp('2024-01-01')
            train_df = df[(df['GAME_DATE'] < train_end) & (df['GAME_DATE'] >= cutoff_date)].copy()
        else:
            # Anchor gets all the data
            train_df = df[df['GAME_DATE'] < train_end].copy()
            
        test_df = df[(df['GAME_DATE'] >= train_end) & (df['GAME_DATE'] < dates[i+1])].copy()
        
        if len(test_df) == 0: continue
        if len(train_df) < 50: continue # Skip if not enough recent data

        # Train
        mh = xgb.XGBRegressor(**params)
        mh.fit(train_df[FEATURES], train_df['PTS_home'])
        
        ma = xgb.XGBRegressor(**params)
        ma.fit(train_df[FEATURES], train_df['PTS_away'])
        
        # Predict
        test_df['pred_home'] = mh.predict(test_df[FEATURES])
        test_df['pred_away'] = ma.predict(test_df[FEATURES])
        all_preds.append(test_df)
        
        final_mh = mh
        final_ma = ma

    if not all_preds: return None, None, 0.0

    full_res = pd.concat(all_preds)
    full_res['pred_total'] = full_res['pred_home'] + full_res['pred_away']
    rmse = np.sqrt(mean_squared_error(full_res['PTS_home'] + full_res['PTS_away'], full_res['pred_total']))
    
    return final_mh, final_ma, rmse

def train_and_evaluate(df, start_date='2023-10-24'):
    dates = pd.date_range(start=pd.Timestamp(start_date), end=df['GAME_DATE'].max(), freq='W-SUN')
    
    # 1. Conservative (full history)
    cons_h, cons_a, cons_rmse = train_specific_model(df, dates, get_params('conservative'), mode='conservative')
    print(f'Conservative RMSE: {cons_rmse}')

    # 2. Chaos (modern era only)
    chaos_h, chaos_a, chaos_rmse = train_specific_model(df, dates, get_params('chaos'), mode='chaos')
    print(f'Chaos RMSE: {chaos_rmse}')
    
    return (cons_h, cons_a), (chaos_h, chaos_a), (cons_rmse, chaos_rmse)

class NBAOracle:
    def __init__(self, df, cons_models, chaos_models):
        self.df = df
        self.cons_h, self.cons_a = cons_models
        self.chaos_h, self.chaos_a = chaos_models
        self.index = self._build_index()
        
    def _build_index(self):
        cols = ['GAME_DATE', 'TEAM', 'OFF', 'DEF', 'PACE', 'ROLL_PTS', 'ROLL_PACE', 'ROLL_WIN']
        
        h = self.df[['GAME_DATE', 'TEAM_NAME_home', 'home_off_rating_pre', 'home_def_rating_pre', 
                     'home_pace_rating_pre', 'home_roll_pts', 'home_roll_pace', 'home_roll_win']].copy()
        h.columns = cols
        
        a = self.df[['GAME_DATE', 'TEAM_NAME_away', 'away_off_rating_pre', 'away_def_rating_pre', 
                     'away_pace_rating_pre', 'away_roll_pts', 'away_roll_pace', 'away_roll_win']].copy()
        a.columns = cols
        
        return pd.concat([h, a]).sort_values('GAME_DATE').groupby('TEAM').last().to_dict('index')
    
    def predict(self, home, away):
        if home not in self.index or away not in self.index:
            print(f"Error: Team not found ({home} or {away})")
            return
            
        h_stats, a_stats = self.index[home], self.index[away]
        today = pd.to_datetime('today')
        
        h_rest = min((today - h_stats['GAME_DATE']).days, 7)
        a_rest = min((today - a_stats['GAME_DATE']).days, 7)
        
        input_data = {
            'home_off_rating_pre': [h_stats['OFF']], 'home_def_rating_pre': [h_stats['DEF']],
            'home_pace_rating_pre': [h_stats['PACE']], 'home_rest_days': [h_rest],
            'away_off_rating_pre': [a_stats['OFF']], 'away_def_rating_pre': [a_stats['DEF']],
            'away_pace_rating_pre': [a_stats['PACE']], 'away_rest_days': [a_rest],
            'home_roll_pts': [h_stats['ROLL_PTS']], 'home_roll_pace': [h_stats['ROLL_PACE']], 'home_roll_win': [h_stats['ROLL_WIN']],
            'away_roll_pts': [a_stats['ROLL_PTS']], 'away_roll_pace': [a_stats['ROLL_PACE']], 'away_roll_win': [a_stats['ROLL_WIN']]
        }
        
        df_in = pd.DataFrame(input_data)
        
        # Predictions
        c_ph = self.cons_h.predict(df_in)[0]
        c_pa = self.cons_a.predict(df_in)[0]
        k_ph = self.chaos_h.predict(df_in)[0]
        k_pa = self.chaos_a.predict(df_in)[0]
        
        print(f"Conservative model: {home} {c_ph:.1f} - {c_pa:.1f} {away} (Total: {c_ph+c_pa:.1f})")
        print(f"Chaos model: {home} {k_ph:.1f} - {k_pa:.1f} {away} (Total: {k_ph+k_pa:.1f})")