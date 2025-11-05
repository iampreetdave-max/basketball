"""
Fetch DraftKings decimal odds for NBA.csv games and merge
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os

# Configuration
API_KEY = "QX0NQvDcyoOD1ezA00fte73Mp8EMDKNxpOZmhxod"
BASE_URL = "https://api.sportradar.com/oddscomparison-prematch/trial/v2"
LOCALE = "en"
SPORT_URN = "sr:sport:2"  # NBA

HEADERS = {"Accept": "application/json"}

# Team name to alias mapping
TEAM_ALIASES = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def get_sport_event_markets(date_str, debug=False):
    """Fetch sport event markets for a specific date"""
    endpoint = f"/{LOCALE}/sports/{SPORT_URN}/schedules/{date_str}/sport_event_markets"
    url = f"{BASE_URL}{endpoint}"
    params = {"api_key": API_KEY, "limit": 50}
    
    try:
        print(f"  {date_str}...", end=" ", flush=True)
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✓")
            
            # Debug: show available markets
            if debug and "sport_schedule_sport_event_markets" in data:
                events = data["sport_schedule_sport_event_markets"]
                if events:
                    event = events[0]
                    markets = event.get("markets", [])
                    market_names = set()
                    for market in markets:
                        books = market.get("books", [])
                        for book in books:
                            if book.get("name") == "DraftKings":
                                market_names.add(market.get("name", "Unknown"))
                    if market_names:
                        print(f"    Available DraftKings markets: {market_names}")
            
            return data
        elif response.status_code == 404:
            print("404")
            return None
        elif response.status_code == 429:
            print("RATE LIMIT - waiting 120s...")
            time.sleep(120)
            return get_sport_event_markets(date_str, debug)
        else:
            print(f"Error {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Exception: {e}")
        return None


def extract_draftkings_odds(markets_data, date_str):
    """Extract ONLY DraftKings decimal odds"""
    odds_dict = {}
    
    if not markets_data or "sport_schedule_sport_event_markets" not in markets_data:
        return odds_dict
    
    events_data = markets_data["sport_schedule_sport_event_markets"]
    
    for event_entry in events_data:
        sport_event = event_entry.get("sport_event", {})
        competitors = sport_event.get("competitors", [])
        
        if len(competitors) < 2:
            continue
        
        # Get team names
        away_name = competitors[1].get("name", "Unknown")
        home_name = competitors[0].get("name", "Unknown")
        
        # Get aliases from mapping
        away_alias = TEAM_ALIASES.get(away_name, away_name.replace(" ", "")[:3].upper())
        home_alias = TEAM_ALIASES.get(home_name, home_name.replace(" ", "")[:3].upper())
        
        # Create game identifier using aliases (to match NBA.csv format)
        game_id = f"{date_str}_{away_alias}@{home_alias}"
        
        # Initialize odds for this game
        game_odds = {
            'game_identifier': game_id,
        }
        
        # Parse all markets for DraftKings only
        markets = event_entry.get("markets", [])
        for market in markets:
            market_name = market.get("name", "Unknown")
            books = market.get("books", [])
            
            for book in books:
                # ONLY process DraftKings
                if book.get("name") != "DraftKings":
                    continue
                
                outcomes = book.get("outcomes", [])
                
                # Process moneyline (home/away winning)
                if "1x2" in market_name or "Moneyline" in market_name or "winner" in market_name.lower():
                    for outcome in outcomes:
                        outcome_type = outcome.get("type", "").lower()
                        odds_decimal = outcome.get("odds_decimal")
                        
                        if odds_decimal and "home" in outcome_type:
                            game_odds['home_winning_odds_decimal'] = odds_decimal
                        elif odds_decimal and "away" in outcome_type:
                            game_odds['away_winning_odds_decimal'] = odds_decimal
                
                # Process total (over/under)
                # Try multiple matching patterns
                if any(x in market_name.lower() for x in ["total", "over/under", "under/over", "ou ", "o/u"]):
                    for outcome in outcomes:
                        outcome_type = outcome.get("type", "").lower()
                        odds_decimal = outcome.get("odds_decimal")
                        total_line = outcome.get("total", "")
                        
                        if odds_decimal:
                            if "over" in outcome_type:
                                game_odds['over_odds_decimal'] = odds_decimal
                                if total_line:
                                    game_odds['total_line'] = total_line
                            elif "under" in outcome_type:
                                game_odds['under_odds_decimal'] = odds_decimal
        
        # Only add if we found DraftKings odds
        if len(game_odds) > 1:  # More than just game_identifier
            odds_dict[game_id] = game_odds
    
    return odds_dict


def main():
    """Load NBA CSV and fetch DraftKings odds"""
    
    print("=" * 80)
    print("FETCH DRAFTKINGS ODDS FOR NBA.CSV GAMES")
    print("=" * 80)
    print()
    
    # ========================================================================
    # STEP 1: Load NBA.csv
    # ========================================================================
    print("STEP 1: Loading NBA.csv...")
    print()
    
    nba_file ="NBA.csv"
    
    try:
        df_nba = pd.read_csv(nba_file)
        print(f"✓ Loaded {len(df_nba)} games")
        print(f"  Columns: {len(df_nba.columns)}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # ========================================================================
    # STEP 2: Extract dates from NBA.csv
    # ========================================================================
    print("\nSTEP 2: Extract dates from games...")
    print()
    
    if 'scheduled' not in df_nba.columns:
        print("✗ No 'scheduled' column")
        return
    
    # Extract unique dates
    df_nba['date'] = pd.to_datetime(df_nba['scheduled']).dt.strftime('%Y-%m-%d')
    unique_dates = sorted(df_nba['date'].unique())
    
    print(f"✓ Unique dates: {len(unique_dates)}")
    print(f"  From: {unique_dates[0]}")
    print(f"  To: {unique_dates[-1]}")
    print(f"  Sample: {unique_dates[:3]}")
    
    # ========================================================================
    # STEP 3: Fetch DraftKings odds for those dates
    # ========================================================================
    print(f"\nSTEP 3: Fetching DraftKings odds from Sportradar...")
    print()
    
    all_odds = {}
    
    for idx, date_str in enumerate(unique_dates):
        # Enable debug on first date to see available markets
        debug_mode = (idx == 0)
        markets_data = get_sport_event_markets(date_str, debug=debug_mode)
        odds_for_date = extract_draftkings_odds(markets_data, date_str)
        all_odds.update(odds_for_date)
        time.sleep(2)
    
    print(f"\n✓ Total games with DraftKings odds: {len(all_odds)}")
    
    if all_odds:
        print(f"\nSample game IDs from API:")
        for game_id in list(all_odds.keys())[:3]:
            print(f"  {game_id}")
    
    print(f"\nSample game IDs from NBA.csv:")
    for game_id in df_nba['game_identifier'].head(3).values:
        print(f"  {game_id}")
    
    # ========================================================================
    # STEP 4: Create odds dataframe
    # ========================================================================
    print(f"\nSTEP 4: Create odds dataframe...")
    print()
    
    odds_rows = []
    for game_id, odds in all_odds.items():
        odds_rows.append(odds)
    
    df_odds = pd.DataFrame(odds_rows)
    
    print(f"✓ Odds records: {len(df_odds)}")
    print(f"  Columns: {list(df_odds.columns)}")
    
    # Add missing columns with NaN
    if 'over_odds_decimal' not in df_odds.columns:
        df_odds['over_odds_decimal'] = None
    if 'under_odds_decimal' not in df_odds.columns:
        df_odds['under_odds_decimal'] = None
    
    # Check what we got
    print(f"\n  Home winning odds: {df_odds['home_winning_odds_decimal'].notna().sum()} games")
    print(f"  Away winning odds: {df_odds['away_winning_odds_decimal'].notna().sum()} games")
    print(f"  Over odds: {df_odds['over_odds_decimal'].notna().sum()} games")
    print(f"  Under odds: {df_odds['under_odds_decimal'].notna().sum()} games")
    
    if df_odds['over_odds_decimal'].notna().sum() == 0:
        print(f"\n  ⚠️  No over/under odds found - checking why...")
        print(f"  API may not have over/under for these dates")
        print(f"  Or market name format is different")
        # Show sample odds that were fetched
        if len(df_odds) > 0:
            print(f"\n  Sample fetched odds row:")
            print(f"  {df_odds.iloc[0]}")
    
    # ========================================================================
    # STEP 5: Merge with NBA data
    # ========================================================================
    print(f"\nSTEP 5: Merge with NBA data...")
    print()
    
    df_combined = df_nba.merge(df_odds, on='game_identifier', how='left')
    
    # Ensure all odds columns exist
    for col in ['home_winning_odds_decimal', 'away_winning_odds_decimal', 'over_odds_decimal', 'under_odds_decimal']:
        if col not in df_combined.columns:
            df_combined[col] = None
    
    matched = df_combined['home_winning_odds_decimal'].notna().sum()
    print(f"✓ Matched: {matched}/{len(df_nba)} games with DraftKings odds")
    print(f"  Coverage: {(matched/len(df_nba)*100):.1f}%")
    
    # ========================================================================
    # STEP 6: Save combined file
    # ========================================================================
    print(f"\nSTEP 6: Saving combined file...")
    print()
    
    current_dir = os.getcwd()
    output_file = os.path.join(current_dir, "NBA_perfect.csv")
    
    df_combined.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Total columns: {len(df_combined.columns)}")
    print(f"  Total rows: {len(df_combined)}")
    
    # ========================================================================
    # STEP 7: Show sample
    # ========================================================================
    print(f"\nSTEP 7: Sample data...")
    print()
    
    sample_cols = [
        'game_identifier', 'home_name', 'away_name',
        'home_winning_odds_decimal', 'away_winning_odds_decimal',
        'over_odds_decimal', 'under_odds_decimal',
        'home_recent_ppg', 'away_recent_ppg',
        'home_points', 'away_points'
    ]
    
    available_cols = [c for c in sample_cols if c in df_combined.columns]
    
    # Only show if we have data
    if available_cols:
        print(df_combined[available_cols].head(5).to_string(index=False))
    
    # Show which odds columns are populated
    print(f"\nOdds columns populated:")
    if 'home_winning_odds_decimal' in df_combined.columns:
        print(f"  ✓ home_winning_odds_decimal: {df_combined['home_winning_odds_decimal'].notna().sum()} games")
    if 'away_winning_odds_decimal' in df_combined.columns:
        print(f"  ✓ away_winning_odds_decimal: {df_combined['away_winning_odds_decimal'].notna().sum()} games")
    if 'over_odds_decimal' in df_combined.columns:
        print(f"  ✓ over_odds_decimal: {df_combined['over_odds_decimal'].notna().sum()} games")
    if 'under_odds_decimal' in df_combined.columns:
        print(f"  ✓ under_odds_decimal: {df_combined['under_odds_decimal'].notna().sum()} games")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Original games: {len(df_nba)}")
    print(f"Games with DraftKings odds: {matched}")
    print(f"Coverage: {(matched/len(df_nba)*100):.1f}%")
    
    print(f"\nNew columns added:")
    print(f"  • home_winning_odds_decimal (DraftKings)")
    print(f"  • away_winning_odds_decimal (DraftKings)")
    print(f"  • over_odds_decimal (DraftKings)")
    print(f"  • under_odds_decimal (DraftKings)")
    
    if matched > 0:
        print(f"\nOdds statistics (from {matched} games):")
        home_mean = df_combined['home_winning_odds_decimal'].mean()
        away_mean = df_combined['away_winning_odds_decimal'].mean()
        print(f"  Home winning odds - Mean: {home_mean:.3f}")
        print(f"  Away winning odds - Mean: {away_mean:.3f}")
        
        over_count = df_combined['over_odds_decimal'].notna().sum()
        under_count = df_combined['under_odds_decimal'].notna().sum()
        
        if over_count > 0:
            over_mean = df_combined['over_odds_decimal'].mean()
            under_mean = df_combined['under_odds_decimal'].mean()
            print(f"  Over odds - Mean: {over_mean:.3f} ({over_count} games)")
            print(f"  Under odds - Mean: {under_mean:.3f} ({under_count} games)")
        else:
            print(f"  ⚠️  Over/Under odds: Not available from API for these games")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
