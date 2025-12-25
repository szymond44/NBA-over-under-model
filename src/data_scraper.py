import pandas as pd
import os
import time
from nba_api.stats.endpoints import leaguegamelog

def scrape_raw_data():
    target_seasons = [
        '2019-20', 
        '2020-21', 
        '2021-22', 
        '2022-23', 
        '2023-24', 
        '2024-25',
        '2025-26'
    ]

    processed_team_dfs = []
    processed_player_dfs = []

    output_folder = 'data'

    for season in target_seasons:
        
        # Teams data (for version1 of the model)
        try:
            # Parsing and spliting data into home and away games
            team_log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='T')
            raw_team_df = team_log.get_data_frames()[0]
            home_teams = raw_team_df[raw_team_df['MATCHUP'].str.contains('vs.')].copy()
            away_teams = raw_team_df[raw_team_df['MATCHUP'].str.contains('@')].copy()
            
            # Set uqique columns that I am gona merge on 
            # these keyes wont have _home or _away suffixes, as they are uniform for both
            merge_keys = ['GAME_ID', 'GAME_DATE', 'SEASON_ID']
            
            # Optional: If 'VIDEO_AVAILABLE' exists and is always same, add it too
            if 'VIDEO_AVAILABLE' in raw_team_df.columns:
                merge_keys.append('VIDEO_AVAILABLE')

            games_merged = pd.merge(
                home_teams,
                away_teams,
                on=merge_keys, 
                suffixes=('_home', '_away')
            )
            
            processed_team_dfs.append(games_merged)
            
        except Exception as e:
            print(f"Error fetching Team data for {season}: {e}")

        # Create raw players dataset for now (for version2 of the model)
        try:
            player_log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P')
            raw_player_df = player_log.get_data_frames()[0]
            
            raw_player_df['IS_HOME'] = raw_player_df['MATCHUP'].str.contains('vs.').astype(int)
            processed_player_dfs.append(raw_player_df)
            
        except Exception as e:
            print(f"Error fetching Player data for {season}: {e}")

        time.sleep(1.5) # Set so that NBA API doesn't kick me out when im scarping the data
    
    #Save data to the csv
    # Teams
    master_team_df = pd.concat(processed_team_dfs, ignore_index=True)
    team_output_path = os.path.join(output_folder, 'nba_games_2019_2025.csv')
    master_team_df.to_csv(team_output_path, index=False)

    # Players
    master_player_df = pd.concat(processed_player_dfs, ignore_index=True)
    player_output_path = os.path.join(output_folder, 'nba_players_2019_2025.csv')
    master_player_df.to_csv(player_output_path, index=False)
    
    