"""
Fetch DraftKings Decimal Odds for UPCOMING NBA Matches
Uses Sportradar Odds Comparison API v2
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# ============================================================================
# SPORTRADAR API CONFIGURATION
# ============================================================================

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


def get_upcoming_games(days_ahead=7):
    """Fetch upcoming NBA games for next N days"""
    print(f"\n{'='*80}")
    print(f"FETCHING UPCOMING NBA GAMES (Next {days_ahead} days)")
    print(f"{'='*80}\n")
    
    upcoming_games = []
    today = datetime.now()
    
    for day_offset in range(days_ahead):
        date = today + timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")
        
        endpoint = f"/{LOCALE}/sports/{SPORT_URN}/schedules/{date_str}/sport_event_markets"
        url = f"{BASE_URL}{endpoint}"
        params = {"api_key": API_KEY, "limit": 50}
        
        try:
            print(f"Checking {date_str}...", end=" ", flush=True)
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "sport_schedule_sport_event_markets" in data:
                    events = data["sport_schedule_sport_event_markets"]
                    if events:
                        print(f"✓ Found {len(events)} games")
                        
                        for event_entry in events:
                            sport_event = event_entry.get("sport_event", {})
                            competitors = sport_event.get("competitors", [])
                            
                            if len(competitors) >= 2:
                                game_info = {
                                    'date': date_str,
                                    'sport_event_id': sport_event.get("id"),
                                    'start_time': sport_event.get("start_time", "N/A"),
                                    'status': sport_event.get("status", "N/A"),
                                    'home_team': competitors[0].get("name", "Unknown"),
                                    'away_team': competitors[1].get("name", "Unknown"),
                                    'markets': event_entry.get("markets", [])
                                }
                                upcoming_games.append(game_info)
                    else:
                        print("No games")
                else:
                    print("404")
            elif response.status_code == 429:
                print("RATE LIMIT - waiting 120s...")
                time.sleep(120)
                return get_upcoming_games(days_ahead)
            else:
                print(f"Error {response.status_code}")
            
            time.sleep(3)
            
        except Exception as e:
            print(f"Exception: {e}")
    
    print(f"\n✓ Total upcoming games found: {len(upcoming_games)}")
    return upcoming_games


def extract_draftkings_odds(games):
    """Extract ONLY DraftKings decimal odds from games"""
    print(f"\n{'='*80}")
    print(f"EXTRACTING DRAFTKINGS DECIMAL ODDS")
    print(f"{'='*80}\n")
    
    odds_list = []
    
    for game in games:
        game_id = f"{game['date']}_{TEAM_ALIASES.get(game['away_team'], game['away_team'][:3].upper())}@{TEAM_ALIASES.get(game['home_team'], game['home_team'][:3].upper())}"
        
        game_odds = {
            'game_identifier': game_id,
            'date': game['date'],
            'start_time': game['start_time'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'status': game['status'],
        }
        
        # Parse markets for DraftKings
        markets = game.get('markets', [])
        has_draftkings = False
        
        for market in markets:
            market_name = market.get("name", "Unknown")
            books = market.get("books", [])
            
            for book in books:
                # ONLY DraftKings
                if book.get("name") != "DraftKings":
                    continue
                
                has_draftkings = True
                outcomes = book.get("outcomes", [])
                
                # Moneyline (winner)
                if any(x in market_name.lower() for x in ["1x2", "moneyline", "winner"]):
                    for outcome in outcomes:
                        outcome_type = outcome.get("type", "").lower()
                        odds_decimal = outcome.get("odds_decimal")
                        
                        if odds_decimal:
                            if "home" in outcome_type:
                                game_odds['home_winning_odds_decimal'] = odds_decimal
                            elif "away" in outcome_type:
                                game_odds['away_winning_odds_decimal'] = odds_decimal
                
                # Total (over/under)
                if any(x in market_name.lower() for x in ["total", "over/under", "ou", "o/u"]):
                    for outcome in outcomes:
                        outcome_type = outcome.get("type", "").lower()
                        odds_decimal = outcome.get("odds_decimal")
                        total_line = outcome.get("total")
                        
                        if odds_decimal:
                            if "over" in outcome_type:
                                game_odds['over_odds_decimal'] = odds_decimal
                                if total_line:
                                    game_odds['total_line'] = total_line
                            elif "under" in outcome_type:
                                game_odds['under_odds_decimal'] = odds_decimal
        
        if has_draftkings and len(game_odds) > 6:  # More than just basic info
            odds_list.append(game_odds)
    
    return odds_list


def main():
    print("\n" + "="*80)
    print("SPORTRADAR - UPCOMING NBA DRAFTKINGS ODDS")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Get upcoming games
    # ========================================================================
    print(f"\nSTEP 1: Fetch upcoming NBA games")
    
    days_to_check = 14  # Check next 14 days
    upcoming_games = get_upcoming_games(days_to_check)
    
    if not upcoming_games:
        print("\n✗ No upcoming games found")
        return
    
    # ========================================================================
    # STEP 2: Extract DraftKings odds
    # ========================================================================
    print(f"\nSTEP 2: Extract DraftKings odds")
    
    odds_list = extract_draftkings_odds(upcoming_games)
    
    if not odds_list:
        print("\n✗ No DraftKings odds found")
        return
    
    print(f"\n✓ Games with DraftKings odds: {len(odds_list)}")
    
    # ========================================================================
    # STEP 3: Create DataFrame and save
    # ========================================================================
    print(f"\nSTEP 3: Save data")
    print()
    
    df_odds = pd.DataFrame(odds_list)
    
    current_dir = os.getcwd()
    output_file = os.path.join(current_dir, "upcoming_nba_draftkings_odds.csv")
    
    df_odds.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Games: {len(df_odds)}")
    print(f"  Columns: {len(df_odds.columns)}")
    
    # ========================================================================
    # STEP 4: Show sample
    # ========================================================================
    print(f"\nSTEP 5: Sample data")
    print()
    
    sample_cols = [
        'game_identifier', 'date', 'start_time',
        'home_team', 'away_team',
        'home_winning_odds_decimal', 'away_winning_odds_decimal',
        'total_line', 'over_odds_decimal', 'under_odds_decimal'
    ]
    
    available_cols = [c for c in sample_cols if c in df_odds.columns]
    
    if available_cols:
        print(df_odds[available_cols].to_string(index=False))
    
    # ========================================================================
    # STEP 6: Statistics
    # ========================================================================
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Total upcoming games with DraftKings odds: {len(df_odds)}")
    print(f"\nOdds availability:")
    print(f"  Home winning: {df_odds['home_winning_odds_decimal'].notna().sum()}")
    print(f"  Away winning: {df_odds['away_winning_odds_decimal'].notna().sum()}")
    print(f"  Over: {df_odds['over_odds_decimal'].notna().sum()}")
    print(f"  Under: {df_odds['under_odds_decimal'].notna().sum()}")
    
    if df_odds['home_winning_odds_decimal'].notna().sum() > 0:
        print(f"\nOdds ranges:")
        print(f"  Home: {df_odds['home_winning_odds_decimal'].min():.2f} - {df_odds['home_winning_odds_decimal'].max():.2f}")
        print(f"  Away: {df_odds['away_winning_odds_decimal'].min():.2f} - {df_odds['away_winning_odds_decimal'].max():.2f}")
    
    if df_odds['over_odds_decimal'].notna().sum() > 0:
        print(f"  Over: {df_odds['over_odds_decimal'].min():.2f} - {df_odds['over_odds_decimal'].max():.2f}")
        print(f"  Under: {df_odds['under_odds_decimal'].min():.2f} - {df_odds['under_odds_decimal'].max():.2f}")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
