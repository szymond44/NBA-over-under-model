import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from src.data_scraper import scrape_raw_data
from src.data_engineering import load_and_process_data
from src.elo_model import elo_model
from src.rolling_stats import add_rolling_stats
from src.model import load_data, train_and_evaluate, NBAOracle

def ensure_data_folder():
    if not os.path.exists('data'):
        os.makedirs('data')

#False to scrape data True to skip
def run_full_pipeline(skip_scraping=False):

    ensure_data_folder()
    
    if not skip_scraping:
        try:
            scrape_raw_data()
        except Exception as e:
            print(f"Error scraping: {e}")
            return None

    try:
        load_and_process_data()
        elo_model()
        add_rolling_stats()
        
        df = load_data()
        cons_models, chaos_models, rmses = train_and_evaluate(df)
        return df, cons_models, chaos_models
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return None

def interactive_prediction_loop(oracle):
    teams = sorted(oracle.index.keys())
    
    print("\nEnter matchups (or 'q' to exit)")
    
    while True:
        try:
            user_input = input("\nEnter matchup: ").strip()
            
            if user_input.lower() in ['q', 'exit']:
                break
            
            if ' vs ' in user_input.lower():
                parts = user_input.lower().split(' vs ')
            elif ' @ ' in user_input.lower():
                parts = user_input.lower().split(' @ ')
            else:
                print("Invalid format.")
                continue
            
            home_input = parts[0].strip()
            away_input = parts[1].strip()
            
            home_match = None
            away_match = None
            
            for team in teams:
                if home_input in team.lower():
                    home_match = team
                if away_input in team.lower():
                    away_match = team
            
            if not home_match or not away_match:
                print("Team not found.")
                continue
            
            print()
            oracle.predict(home_match, away_match)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    skip_scraping = False
    if len(sys.argv) > 1 and sys.argv[1] == '--skip-scraping':
        skip_scraping = True

    result = run_full_pipeline(skip_scraping=skip_scraping)
    
    if not result:
        return

    df, cons_models, chaos_models = result
    oracle = NBAOracle(df, cons_models, chaos_models)
    
    interactive_prediction_loop(oracle)

if __name__ == "__main__":
    main()