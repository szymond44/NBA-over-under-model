import os
import sys
# Ensure we're working from the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from src.data_scraper import scrape_raw_data
from src.data_engineering import load_and_process_data
from src.elo_model import elo_model
from src.rolling_stats import add_rolling_stats
from src.model import load_data, train_and_evaluate, NBAOracle


def ensure_data_folder():
    """Creates the data folder if it doesn't exist."""
    if not os.path.exists('data'):
        os.makedirs('data')


def run_full_pipeline(skip_scraping=False):
    """
    Executes the complete pipeline from data collection to model training.
    
    Args:
        skip_scraping: If True, skips data scraping (useful if data already exists)
    """
    ensure_data_folder()
    
    if not skip_scraping:
        try:
            scrape_raw_data()
        except Exception as e:
            print(f"Error during data scraping: {e}")
            return None, None, None, None
    

    try:
        load_and_process_data()
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return None, None, None, None
    
    try:
        elo_model()
    except Exception as e:
        print(f"Error during Elo calculation: {e}")
        return None, None, None, None

    try:
        add_rolling_stats()
    except Exception as e:
        print(f"Error during rolling stats computation: {e}")
        return None, None, None, None

    try:
        df = load_data()
        model_home, model_away, rmse = train_and_evaluate(df)
        return df, model_home, model_away, rmse
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None, None, None


def interactive_prediction_loop(oracle):
    """
    Interactive loop for making predictions on upcoming games.
    
    Args:
        oracle: Trained NBAOracle instance
    """
    teams = sorted(oracle.index.keys())
    if len(teams) % 3 != 0:
        print()
    
    print("\nEnter matchups (or 'quit' to exit)")
    print("Example: 'Los Angeles Lakers vs Boston Celtics'\n")
    
    while True:
        try:
            user_input = input("\nEnter matchup: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Parse input
            if ' vs ' in user_input.lower():
                parts = user_input.lower().split(' vs ')
            elif ' @ ' in user_input.lower():
                parts = user_input.lower().split(' @ ')
            else:
                print("Invalid format. Use 'Team1 vs Team2' or 'Team1 @ Team2'")
                continue
            
            if len(parts) != 2:
                print("Invalid format. Please enter exactly two teams.")
                continue
            
            home_input = parts[0].strip().title()
            away_input = parts[1].strip().title()
            
            # Find matching teams
            home_match = None
            away_match = None
            
            for team in teams:
                if home_input.lower() in team.lower():
                    home_match = team
                if away_input.lower() in team.lower():
                    away_match = team
            
            if not home_match:
                print(f"Could not find home team: '{home_input}'")
                continue
            if not away_match:
                print(f"Could not find away team: '{away_input}'")
                continue
            
            # Make prediction
            print()
            oracle.predict(home_match, away_match)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main execution function."""
    
    # Check if user wants to skip scraping
    skip_scraping = False
    if len(sys.argv) > 1 and sys.argv[1] == '--skip-scraping':
        skip_scraping = True

    df, model_home, model_away, rmse = run_full_pipeline(skip_scraping=skip_scraping)
    
    if df is None or model_home is None or model_away is None:
        print("Pipeline failed. Please check the errors above.")
        return

    oracle = NBAOracle(df, model_home, model_away)
    
    interactive_prediction_loop(oracle)

if __name__ == "__main__":
    main()

