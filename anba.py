"""
NBA Historical Games + Odds Integration (FIXED VERSION)
FIX: Now properly extracts actual game scores from game summaries

The issue was that extract_game_result() was looking at schedule data,
which doesn't contain final scores. Now we get scores from game summaries.
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ============================================================================
# API CONFIGURATION
# ============================================================================

NBA_API_KEY = "ihQYMqeJjtkej2xx6LzFYy8ZqLzw0z2irwMR7NaC"
NBA_BASE_URL = "https://api.sportradar.us/nba/trial/v8/en"

ODDS_API_KEY = "ihQYMqeJjtkej2xx6LzFYy8ZqLzw0z2irwMR7NaC"
ODDS_BASE_URL = "https://api.sportradar.com/oddscomparison-usp1"

NBA_REQUEST_DELAY = 1.1
ODDS_REQUEST_DELAY = 1.1


class HistoricalFeatureEngine:
    """Fetches historical games and reconstructs pre-match features"""
    
    def __init__(self, api_key=NBA_API_KEY):
        self.api_key = api_key
        self.base_url = NBA_BASE_URL
        self.request_count = 0
        self.start_time = time.time()
        self.game_summaries = {}
        self.season_games = None
        
    def _make_request(self, endpoint: str, retries: int = 3) -> Optional[Dict]:
        """Make API request with retry logic"""
        url = f"{self.base_url}/{endpoint}?api_key={self.api_key}"
        
        for attempt in range(retries):
            try:
                elapsed = time.time() - self.start_time
                print(f"  [NBA API - {self.request_count+1} reqs, {elapsed:.0f}s] {endpoint[:50]}...")
                
                response = requests.get(url, timeout=30)
                self.request_count += 1
                
                if response.status_code == 200:
                    time.sleep(NBA_REQUEST_DELAY)
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    print(f"  ⚠️  Rate limit! Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    print(f"  ℹ️  Not found (404)")
                    return None
                else:
                    print(f"  ⚠️  Error {response.status_code}")
                    if attempt < retries - 1:
                        time.sleep(5)
                        continue
                    return None
                    
            except Exception as e:
                print(f"  ❌ Request failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                return None
        
        return None
    
    def get_season_schedule(self, year: int = 2024, season_type: str = "REG") -> Optional[List[Dict]]:
        """Get full season schedule"""
        if self.season_games is not None:
            return self.season_games
        
        endpoint = f"games/{year}/{season_type}/schedule.json"
        schedule = self._make_request(endpoint)
        
        if schedule and 'games' in schedule:
            self.season_games = schedule['games']
            return self.season_games
        
        return []
    
    def get_game_summary(self, game_id: str) -> Optional[Dict]:
        """Get detailed game statistics"""
        if game_id in self.game_summaries:
            return self.game_summaries[game_id]
        
        endpoint = f"games/{game_id}/summary.json"
        summary = self._make_request(endpoint)
        
        if summary:
            self.game_summaries[game_id] = summary
        
        return summary
    
    def filter_completed_games(self, games: List[Dict], limit: int = 100) -> List[Dict]:
        """Filter to only completed games"""
        completed = [g for g in games if g.get('status') == 'closed']
        completed_sorted = sorted(completed, key=lambda x: x.get('scheduled', ''), reverse=True)
        return completed_sorted[:limit]
    
    def get_games_before_date(self, all_games: List[Dict], before_date: str, 
                             team_id: str, limit: int = 5) -> List[Dict]:
        """Get N games for a team that occurred BEFORE a specific date"""
        team_games_before = []
        
        for game in all_games:
            game_date = game.get('scheduled', '')
            if game_date >= before_date:
                continue
            if game.get('status') != 'closed':
                continue
            
            home_id = game.get('home', {}).get('id')
            away_id = game.get('away', {}).get('id')
            
            if team_id not in [home_id, away_id]:
                continue
            
            team_games_before.append(game)
        
        team_games_sorted = sorted(team_games_before, key=lambda x: x.get('scheduled', ''), reverse=True)
        return team_games_sorted[:limit]
    
    def calculate_team_stats_from_games(self, team_id: str, games: List[Dict]) -> Dict:
        """Calculate statistics from a list of games"""
        if not games:
            return {'games_played': 0, 'insufficient_data': True}
        
        stats_list = []
        results = []
        
        for game in games:
            game_summary = self.get_game_summary(game['id'])
            if not game_summary:
                continue
            
            home_id = game_summary.get('home', {}).get('id')
            
            if team_id == home_id:
                team_stats = game_summary.get('home', {}).get('statistics', {})
                opp_stats = game_summary.get('away', {}).get('statistics', {})
                team_points = game_summary.get('home', {}).get('points', 0)
                opp_points = game_summary.get('away', {}).get('points', 0)
                is_home = True
            else:
                team_stats = game_summary.get('away', {}).get('statistics', {})
                opp_stats = game_summary.get('home', {}).get('statistics', {})
                team_points = game_summary.get('away', {}).get('points', 0)
                opp_points = game_summary.get('home', {}).get('points', 0)
                is_home = False
            
            if not team_stats:
                continue
            
            stats_list.append(team_stats)
            results.append({
                'won': team_points > opp_points,
                'points_for': team_points,
                'points_against': opp_points,
                'is_home': is_home
            })
        
        if not stats_list:
            return {'games_played': 0, 'insufficient_data': True}
        
        avg_stats = self._average_stats(stats_list)
        form_stats = self._calculate_form(results)
        
        recent_stats = {
            'games_played': len(stats_list),
            'insufficient_data': False,
            **{f'recent_{k}': v for k, v in avg_stats.items()},
            **form_stats
        }
        
        return recent_stats
    
    def _average_stats(self, stats_list: List[Dict]) -> Dict:
        """Calculate average statistics"""
        if not stats_list:
            return {}
        
        key_metrics = [
            'points', 'field_goals_pct', 'three_points_pct', 'free_throws_pct',
            'rebounds', 'assists', 'turnovers', 'steals', 'blocks',
            'offensive_rebounds', 'defensive_rebounds'
        ]
        
        avg_stats = {}
        for metric in key_metrics:
            values = [s.get(metric, 0) for s in stats_list if metric in s]
            if values:
                avg_stats[metric] = round(np.mean(values), 2)
        
        return avg_stats
    
    def _calculate_form(self, results: List[Dict]) -> Dict:
        """Calculate form metrics"""
        if not results:
            return {}
        
        wins = sum(1 for r in results if r['won'])
        point_diffs = [r['points_for'] - r['points_against'] for r in results]
        
        form = {
            'recent_wins': wins,
            'recent_losses': len(results) - wins,
            'recent_win_pct': round(wins / len(results), 3) if results else 0,
            'recent_ppg': round(np.mean([r['points_for'] for r in results]), 2),
            'recent_opp_ppg': round(np.mean([r['points_against'] for r in results]), 2),
            'recent_point_diff': round(np.mean(point_diffs), 2),
            'recent_scoring_trend': self._calculate_trend([r['points_for'] for r in results])
        }
        
        return form
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if values are trending up, down, or stable"""
        if len(values) < 2:
            return 'stable'
        
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if slope > 1:
            return 'up'
        elif slope < -1:
            return 'down'
        else:
            return 'stable'
    
    def create_game_identifier(self, game: Dict) -> str:
        """Create unique game identifier for matching"""
        scheduled = game.get('scheduled', '')
        if scheduled:
            game_date = scheduled.split('T')[0]
        else:
            game_date = 'UNKNOWN'
        
        home_alias = game.get('home', {}).get('alias', 'UNK')
        away_alias = game.get('away', {}).get('alias', 'UNK')
        
        return f"{game_date}_{away_alias}@{home_alias}"
    
    def extract_game_result_from_summary(self, game_summary: Dict) -> Dict:
        """
        🔧 FIXED: Extract actual game result from game SUMMARY (not schedule)
        
        This is the key fix - we need to get scores from the summary endpoint,
        not the schedule endpoint which doesn't have final scores.
        """
        home_points = game_summary.get('home', {}).get('points', 0)
        away_points = game_summary.get('away', {}).get('points', 0)
        
        return {
            'home_points': home_points,
            'away_points': away_points,
            'total_points': home_points + away_points,
            'point_differential': home_points - away_points,
            'home_won': 1 if home_points > away_points else 0,
            'away_won': 1 if away_points > home_points else 0,
        }
    
    def enrich_historical_game(self, game: Dict, all_games: List[Dict], 
                              recent_games_count: int = 5) -> Dict:
        """
        🔧 FIXED: Now properly extracts actual scores
        
        Enrich a historical game with pre-match features AND actual results
        """
        game_date = game.get('scheduled', '')
        home_id = game.get('home', {}).get('id')
        away_id = game.get('away', {}).get('id')
        home_alias = game.get('home', {}).get('alias', 'UNK')
        away_alias = game.get('away', {}).get('alias', 'UNK')
        
        print(f"  Processing: {away_alias} @ {home_alias} ({game_date[:10]})")
        
        enriched_data = {
            'match_id': game.get('id', ''),
            'game_identifier': self.create_game_identifier(game),
            'scheduled': game_date,
            'venue_name': game.get('venue', {}).get('name', ''),
            'venue_city': game.get('venue', {}).get('city', ''),
            'home_id': home_id,
            'home_name': game.get('home', {}).get('name', ''),
            'home_alias': home_alias,
            'home_market': game.get('home', {}).get('market', ''),
            'away_id': away_id,
            'away_name': game.get('away', {}).get('name', ''),
            'away_alias': away_alias,
            'away_market': game.get('away', {}).get('market', ''),
        }
        
        # Get pre-match features for home team
        home_recent_games = self.get_games_before_date(all_games, game_date, home_id, recent_games_count)
        home_stats = self.calculate_team_stats_from_games(home_id, home_recent_games)
        
        for key, value in home_stats.items():
            enriched_data[f'home_{key}'] = value
        
        # Get pre-match features for away team
        away_recent_games = self.get_games_before_date(all_games, game_date, away_id, recent_games_count)
        away_stats = self.calculate_team_stats_from_games(away_id, away_recent_games)
        
        for key, value in away_stats.items():
            enriched_data[f'away_{key}'] = value
        
        # Calculate comparative features
        if not home_stats.get('insufficient_data') and not away_stats.get('insufficient_data'):
            enriched_data.update(self._calculate_comparative_features(home_stats, away_stats))
        
        # 🔧 FIX: Get actual game result from SUMMARY, not schedule
        print(f"    Fetching final score...")
        game_summary = self.get_game_summary(game['id'])
        
        if game_summary:
            result = self.extract_game_result_from_summary(game_summary)
            enriched_data.update(result)
            print(f"    Score: {away_alias} {result['away_points']} @ {home_alias} {result['home_points']}")
        else:
            # If summary unavailable, set to None instead of 0
            print(f"    ⚠️  Could not fetch final score")
            enriched_data.update({
                'home_points': None,
                'away_points': None,
                'total_points': None,
                'point_differential': None,
                'home_won': None,
                'away_won': None,
            })
        
        return enriched_data
    
    def _calculate_comparative_features(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Calculate comparative features"""
        comparative = {}
        
        home_ppg = home_stats.get('recent_ppg', 0)
        away_ppg = away_stats.get('recent_ppg', 0)
        comparative['scoring_advantage_home'] = round(home_ppg - away_ppg, 2)
        
        home_win_pct = home_stats.get('recent_win_pct', 0)
        away_win_pct = away_stats.get('recent_win_pct', 0)
        comparative['form_advantage_home'] = round(home_win_pct - away_win_pct, 3)
        
        home_opp_ppg = home_stats.get('recent_opp_ppg', 0)
        away_opp_ppg = away_stats.get('recent_opp_ppg', 0)
        comparative['defensive_advantage_home'] = round(away_opp_ppg - home_opp_ppg, 2)
        
        home_assists = home_stats.get('recent_assists', 0)
        home_turnovers = home_stats.get('recent_turnovers', 1)
        away_assists = away_stats.get('recent_assists', 0)
        away_turnovers = away_stats.get('recent_turnovers', 1)
        
        home_ratio = round(home_assists / home_turnovers, 2) if home_turnovers > 0 else 0
        away_ratio = round(away_assists / away_turnovers, 2) if away_turnovers > 0 else 0
        comparative['ball_control_advantage_home'] = round(home_ratio - away_ratio, 2)
        
        return comparative


class OddsFetcher:
    """Fetches betting odds from Sportradar Odds API"""
    
    def __init__(self, api_key=ODDS_API_KEY):
        self.api_key = api_key
        self.base_url = ODDS_BASE_URL
        self.request_count = 0
        
    def _make_request(self, endpoint: str, retries: int = 3) -> Optional[Dict]:
        """Make API request with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        params = {'api_key': self.api_key}
        
        for attempt in range(retries):
            try:
                print(f"  [Odds API - {self.request_count+1} reqs] {endpoint[:50]}...")
                
                response = requests.get(url, params=params, timeout=30)
                self.request_count += 1
                
                if response.status_code == 200:
                    time.sleep(ODDS_REQUEST_DELAY)
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    print(f"  ⚠️  Rate limit! Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  ⚠️  Error {response.status_code}")
                    if attempt < retries - 1:
                        time.sleep(5)
                        continue
                    return None
                    
            except Exception as e:
                print(f"  ❌ Request failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                return None
        
        return None
    
    def get_schedule_with_odds(self, start_date: str, end_date: str) -> List[Dict]:
        """Get NBA schedule with odds for date range"""
        endpoint = f"us/schedules/{start_date}/schedule.json"
        
        schedule = self._make_request(endpoint)
        
        if not schedule:
            return []
        
        sport_events = schedule.get('sport_events', [])
        nba_events = [e for e in sport_events if e.get('sport', {}).get('name') == 'Basketball']
        
        return nba_events
    
    def extract_odds_data(self, sport_event: Dict) -> Dict:
        """Extract key betting odds from sport event"""
        odds_data = {
            'has_odds': False,
            'moneyline_home': None,
            'moneyline_away': None,
            'spread_home': None,
            'spread_away': None,
            'spread_points': None,
            'total_over': None,
            'total_under': None,
            'total_points': None,
            'num_bookmakers': 0
        }
        
        markets = sport_event.get('markets', [])
        if not markets:
            return odds_data
        
        odds_data['has_odds'] = True
        odds_data['num_bookmakers'] = len(markets)
        
        for market in markets:
            books = market.get('books', [])
            if not books:
                continue
            
            book = books[0]
            outcomes = book.get('outcomes', [])
            
            market_type = market.get('name', '')
            
            if market_type == '2way':
                for outcome in outcomes:
                    competitor_id = outcome.get('competitor_id')
                    odds = outcome.get('odds')
                    
                    if competitor_id == sport_event.get('competitors', [{}])[0].get('id'):
                        odds_data['moneyline_home'] = odds
                    else:
                        odds_data['moneyline_away'] = odds
            
            elif market_type == 'spread':
                for outcome in outcomes:
                    competitor_id = outcome.get('competitor_id')
                    spread = outcome.get('spread')
                    odds = outcome.get('odds')
                    
                    if competitor_id == sport_event.get('competitors', [{}])[0].get('id'):
                        odds_data['spread_home'] = odds
                        odds_data['spread_points'] = spread
                    else:
                        odds_data['spread_away'] = odds
            
            elif market_type == 'total':
                for outcome in outcomes:
                    outcome_type = outcome.get('type')
                    total = outcome.get('total')
                    odds = outcome.get('odds')
                    
                    if outcome_type == 'over':
                        odds_data['total_over'] = odds
                        odds_data['total_points'] = total
                    else:
                        odds_data['total_under'] = odds
        
        return odds_data
    
    def create_game_identifier_from_odds(self, sport_event: Dict) -> str:
        """Create game identifier from odds data"""
        scheduled = sport_event.get('scheduled', '')
        if scheduled:
            game_date = scheduled.split('T')[0]
        else:
            return 'UNKNOWN'
        
        competitors = sport_event.get('competitors', [])
        if len(competitors) < 2:
            return 'UNKNOWN'
        
        home_team = None
        away_team = None
        
        for comp in competitors:
            if comp.get('qualifier') == 'home':
                home_team = comp.get('abbreviation', 'UNK')
            elif comp.get('qualifier') == 'away':
                away_team = comp.get('abbreviation', 'UNK')
        
        if not home_team or not away_team:
            return 'UNKNOWN'
        
        return f"{game_date}_{away_team}@{home_team}"


class NBAHistoricalWithOdds:
    """Main pipeline combining historical games with odds"""
    
    def __init__(self, nba_api_key=NBA_API_KEY, odds_api_key=ODDS_API_KEY):
        self.nba_engine = HistoricalFeatureEngine(nba_api_key)
        self.odds_fetcher = OddsFetcher(odds_api_key)
        
    def fetch_historical_games(self, num_games: int = 50, year: int = 2024,
                              season_type: str = "REG", recent_games_count: int = 5) -> pd.DataFrame:
        """Fetch historical games with features"""
        print(f"\n{'='*70}")
        print(f"STEP 1: FETCHING HISTORICAL NBA GAMES")
        print(f"{'='*70}\n")
        
        all_games = self.nba_engine.get_season_schedule(year, season_type)
        
        if not all_games:
            print("❌ Failed to fetch season schedule")
            return pd.DataFrame()
        
        print(f"✓ Found {len(all_games)} total games\n")
        
        completed_games = self.nba_engine.filter_completed_games(all_games, num_games)
        
        if not completed_games:
            print("❌ No completed games found")
            return pd.DataFrame()
        
        print(f"✓ Found {len(completed_games)} completed games")
        print(f"Processing {min(num_games, len(completed_games))} games...\n")
        
        enriched_games = []
        
        for idx, game in enumerate(completed_games[:num_games], 1):
            print(f"[{idx}/{num_games}]")
            
            try:
                enriched = self.nba_engine.enrich_historical_game(
                    game, all_games, recent_games_count
                )
                enriched_games.append(enriched)
                print(f"  ✓ Completed\n")
                
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                continue
        
        df = pd.DataFrame(enriched_games)
        print(f"✓ Historical data fetched: {len(df)} games\n")
        
        return df
    
    def fetch_odds_for_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch odds for a date range"""
        print(f"\n{'='*70}")
        print(f"STEP 2: FETCHING BETTING ODDS")
        print(f"{'='*70}\n")
        print(f"Date range: {start_date} to {end_date}\n")
        
        all_odds = []
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            print(f"Fetching odds for {date_str}...")
            
            sport_events = self.odds_fetcher.get_schedule_with_odds(date_str, date_str)
            
            for event in sport_events:
                game_identifier = self.odds_fetcher.create_game_identifier_from_odds(event)
                odds_data = self.odds_fetcher.extract_odds_data(event)
                
                odds_record = {
                    'game_identifier': game_identifier,
                    'odds_scheduled': event.get('scheduled', ''),
                    **odds_data
                }
                
                all_odds.append(odds_record)
            
            print(f"  ✓ Found {len(sport_events)} games\n")
            
            current += timedelta(days=1)
        
        df_odds = pd.DataFrame(all_odds)
        print(f"✓ Odds data fetched: {len(df_odds)} games\n")
        
        return df_odds
    
    def merge_data(self, df_games: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
        """Merge historical games with odds data"""
        print(f"\n{'='*70}")
        print(f"STEP 3: MERGING DATA")
        print(f"{'='*70}\n")
        
        print(f"Games with features: {len(df_games)}")
        print(f"Games with odds: {len(df_odds)}")
        
        df_merged = df_games.merge(
            df_odds,
            on='game_identifier',
            how='left'
        )
        
        print(f"\n✓ Merged dataset: {len(df_merged)} games")
        
        if 'has_odds' in df_merged.columns:
            print(f"  Games with odds: {df_merged['has_odds'].sum()}")
            print(f"  Games without odds: {(~df_merged['has_odds']).sum()}")
        
        print()
        
        return df_merged
    
    def run_full_pipeline(self, num_games: int = 50, year: int = 2024,
                         season_type: str = "REG", recent_games_count: int = 5,
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Run full pipeline: fetch games + odds + merge"""
        
        print(f"\n{'='*70}")
        print(f"NBA HISTORICAL GAMES + ODDS INTEGRATION (FIXED)")
        print(f"{'='*70}")
        print(f"\nConfiguration:")
        print(f"  Games to fetch: {num_games}")
        print(f"  Season: {year} {season_type}")
        print(f"  Recent games for features: {recent_games_count}")
        if start_date and end_date:
            print(f"  Odds date range: {start_date} to {end_date}")
        print(f"{'='*70}\n")
        
        # Note: We now need 1 more API call per game for the final score
        nba_calls = 1 + (num_games * 12)  # Changed from 11 to 12
        
        if start_date and end_date:
            days = (datetime.strptime(end_date, '%Y-%m-%d') - 
                   datetime.strptime(start_date, '%Y-%m-%d')).days + 1
            odds_calls = days
        else:
            odds_calls = 0
        
        total_calls = nba_calls + odds_calls
        estimated_time = total_calls * 1.1 / 60
        
        print(f"ESTIMATES:")
        print(f"  NBA API calls: ~{nba_calls}")
        print(f"  Odds API calls: ~{odds_calls}")
        print(f"  Total API calls: ~{total_calls}")
        print(f"  Estimated time: ~{estimated_time:.1f} minutes")
        print(f"{'='*70}\n")
        
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return pd.DataFrame()
        
        df_games = self.fetch_historical_games(num_games, year, season_type, recent_games_count)
        
        if df_games.empty:
            print("❌ No games fetched")
            return pd.DataFrame()
        
        # Determine date range if not provided
        if not start_date or not end_date:
            dates = pd.to_datetime(df_games['scheduled'])
            start_date = dates.min().strftime('%Y-%m-%d')
            end_date = dates.max().strftime('%Y-%m-%d')
            print(f"Auto-detected date range: {start_date} to {end_date}")
        
        # Fetch odds
        df_odds = self.fetch_odds_for_date_range(start_date, end_date)
        
        if df_odds.empty:
            print("⚠️  No odds fetched - continuing with games only")
            return df_games
        
        # Merge
        df_final = self.merge_data(df_games, df_odds)
        
        return df_final


def main():
    """Main execution function"""
    
    # Check if odds API key is set
    if ODDS_API_KEY == "YOUR_ODDS_API_KEY_HERE":
        print("\n" + "="*70)
        print("⚠️  WARNING: Odds API key not configured!")
        print("="*70)
        print("\nTo get odds data, you need a Sportradar Odds Comparison API key.")
        print("Get one at: https://developer.sportradar.com/")
        print("\nOptions:")
        print("  1. Set ODDS_API_KEY in the script and run again")
        print("  2. Continue without odds (games + features only)")
        print("="*70 + "\n")
        
        response = input("Continue without odds? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled. Please add your Odds API key and run again.")
            return
        
        print("\n✓ Continuing with NBA games only (no odds)\n")
        use_odds = False
    else:
        use_odds = True
    
    # Configuration
    CONFIG = {
        'num_games': 50,           # Number of historical games
        'year': 2024,              # Season year
        'season_type': 'REG',      # Regular season
        'recent_games_count': 5,   # Last N games for features
        'start_date': None,        # Auto-detect from games
        'end_date': None,          # Auto-detect from games
    }
    
    # Initialize pipeline
    pipeline = NBAHistoricalWithOdds()
    
    # Run pipeline
    start_time = time.time()
    
    if use_odds:
        df_final = pipeline.run_full_pipeline(**CONFIG)
    else:
        # Games only
        df_final = pipeline.fetch_historical_games(
            CONFIG['num_games'],
            CONFIG['year'],
            CONFIG['season_type'],
            CONFIG['recent_games_count']
        )
    
    elapsed = time.time() - start_time
    
    if df_final.empty:
        print("\n❌ No data to save")
        return
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total games: {len(df_final)}")
    print(f"Total columns: {len(df_final.columns)}")
    print(f"NBA API requests: {pipeline.nba_engine.request_count}")
    if use_odds:
        print(f"Odds API requests: {pipeline.odds_fetcher.request_count}")
        if 'has_odds' in df_final.columns:
            print(f"Games with odds: {df_final['has_odds'].sum()}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Avg per game: {elapsed/len(df_final):.1f} seconds")
    
    # 🔧 VERIFICATION: Check if scores are populated
    print(f"\n{'='*70}")
    print(f"🔧 SCORE DATA VERIFICATION")
    print(f"{'='*70}")
    
    if 'home_points' in df_final.columns:
        null_scores = df_final['home_points'].isnull().sum()
        zero_scores = (df_final['home_points'] == 0).sum()
        valid_scores = len(df_final) - null_scores - zero_scores
        
        print(f"✓ Valid scores: {valid_scores}/{len(df_final)} games")
        if null_scores > 0:
            print(f"⚠️  Null scores: {null_scores} games")
        if zero_scores > 0:
            print(f"⚠️  Zero scores: {zero_scores} games (possible shutouts or errors)")
        
        if valid_scores > 0:
            print(f"\nSample scores:")
            sample = df_final[df_final['home_points'].notna() & (df_final['home_points'] > 0)].head(3)
            for _, row in sample.iterrows():
                print(f"  {row['away_alias']} {int(row['away_points'])} @ {row['home_alias']} {int(row['home_points'])}")
    else:
        print("❌ home_points column not found!")
    
    # Column breakdown
    print(f"\n{'='*70}")
    print(f"COLUMN CATEGORIES")
    print(f"{'='*70}")
    
    basic_cols = [c for c in df_final.columns if not c.startswith(('home_', 'away_', 'recent_'))]
    home_cols = [c for c in df_final.columns if c.startswith('home_')]
    away_cols = [c for c in df_final.columns if c.startswith('away_')]
    odds_cols = [c for c in df_final.columns if c.startswith(('moneyline', 'spread', 'total', 'odds_', 'has_odds', 'num_book'))]
    
    print(f"Basic info: {len(basic_cols)} columns")
    print(f"Home team: {len(home_cols)} columns")
    print(f"Away team: {len(away_cols)} columns")
    if use_odds and len(odds_cols) > 0:
        print(f"Odds data: {len(odds_cols)} columns")
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"nba_historical_FIXED_{timestamp}.csv"
    df_final.to_csv(filename, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Data saved to: {filename}")
    print(f"{'='*70}")
    
    # Display sample
    print(f"\n{'='*70}")
    print(f"SAMPLE DATA (first 3 games)")
    print(f"{'='*70}\n")
    
    sample_cols = [
        'game_identifier', 'home_alias', 'away_alias',
        'home_recent_ppg', 'away_recent_ppg',
        'home_recent_win_pct', 'away_recent_win_pct',
    ]
    
    if use_odds and 'moneyline_home' in df_final.columns:
        sample_cols.extend(['moneyline_home', 'moneyline_away', 'spread_points', 'total_points'])
    
    sample_cols.extend(['home_points', 'away_points', 'home_won'])
    
    available_cols = [col for col in sample_cols if col in df_final.columns]
    
    if available_cols:
        print(df_final[available_cols].head(3).to_string(index=False))
    
    # Quick stats
    if 'home_won' in df_final.columns and 'home_points' in df_final.columns:
        print(f"\n{'='*70}")
        print(f"QUICK STATS")
        print(f"{'='*70}")
        
        # Filter to games with valid scores
        valid_games = df_final[df_final['home_points'].notna() & (df_final['home_points'] > 0)]
        
        if len(valid_games) > 0:
            home_wins = valid_games['home_won'].sum()
            total_games = len(valid_games)
            home_win_pct = (home_wins / total_games * 100) if total_games > 0 else 0
            
            print(f"Home team wins: {home_wins}/{total_games} ({home_win_pct:.1f}%)")
            
            if 'total_points' in valid_games.columns:
                avg_total = valid_games['total_points'].mean()
                print(f"Average total points: {avg_total:.1f}")
            
            if 'point_differential' in valid_games.columns:
                avg_diff = valid_games['point_differential'].abs().mean()
                print(f"Average point differential: {avg_diff:.1f}")
            
            if use_odds and 'has_odds' in valid_games.columns:
                pct_with_odds = (valid_games['has_odds'].sum() / total_games * 100)
                print(f"Games with odds data: {pct_with_odds:.1f}%")
        else:
            print("⚠️  No valid game scores found")
        
        print(f"{'='*70}\n")
    
    # Analysis suggestions
    print(f"\n{'='*70}")
    print(f"NEXT STEPS")
    print(f"{'='*70}")
    print("\n📊 Suggested analyses:")
    print("  1. Correlation analysis: features vs actual results")
    print("  2. Feature importance: which stats predict wins best?")
    if use_odds:
        print("  3. Odds accuracy: compare implied probabilities to outcomes")
        print("  4. Value betting: find games where odds don't match form")
    print("  5. Train ML models: predict winners, spreads, totals")
    print("  6. Backtest strategies: simulate betting based on features")
    
    print(f"\n📁 Key columns for analysis:")
    key_features = [
        'home_recent_win_pct', 'away_recent_win_pct',
        'scoring_advantage_home', 'form_advantage_home',
        'defensive_advantage_home', 'ball_control_advantage_home'
    ]
    
    if use_odds:
        key_features.extend(['moneyline_home', 'spread_points', 'total_points'])
    
    key_features.extend(['home_points', 'away_points', 'home_won', 'point_differential'])
    
    for col in key_features:
        if col in df_final.columns:
            print(f"  ✓ {col}")
    
    print(f"\n{'='*70}")
    print(f"NOTES")
    print(f"{'='*70}")
    print("✓ Pre-match features are from BEFORE each game")
    print("✓ Actual results are included (home_points, away_points, home_won)")
    print("✓ 🔧 FIXED: Scores now properly extracted from game summaries")
    if use_odds:
        print("✓ Betting odds included (moneyline, spread, total)")
        print("✓ game_identifier used to match games across APIs")
    print("✓ Ready for ML training and backtesting!")
    print(f"{'='*70}\n")


def quick_analysis(csv_file: str):
    """Quick analysis of saved CSV file"""
    print(f"\n{'='*70}")
    print(f"QUICK ANALYSIS: {csv_file}")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(csv_file)
    
    print(f"Total games: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    if 'scheduled' in df.columns:
        print(f"Date range: {df['scheduled'].min()[:10]} to {df['scheduled'].max()[:10]}")
    
    # Check for missing data
    print(f"\n{'='*70}")
    print(f"DATA COMPLETENESS")
    print(f"{'='*70}\n")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    
    high_missing = missing_pct[missing_pct > 10].sort_values(ascending=False)
    
    if len(high_missing) > 0:
        print("Columns with >10% missing data:")
        for col, pct in high_missing.items():
            print(f"  {col}: {pct}% missing")
    else:
        print("✓ All columns have <10% missing data")
    
    # Score verification
    if 'home_points' in df.columns:
        print(f"\n{'='*70}")
        print(f"SCORE VERIFICATION")
        print(f"{'='*70}\n")
        
        null_scores = df['home_points'].isnull().sum()
        zero_scores = (df['home_points'] == 0).sum()
        valid_scores = len(df) - null_scores
        
        print(f"Valid scores: {valid_scores}/{len(df)} games")
        if null_scores > 0:
            print(f"Missing scores: {null_scores} games")
        
        if valid_scores > 0:
            avg_home = df['home_points'].mean()
            avg_away = df['away_points'].mean()
            print(f"Average home score: {avg_home:.1f}")
            print(f"Average away score: {avg_away:.1f}")
    
    # Home advantage
    if 'home_won' in df.columns:
        print(f"\n{'='*70}")
        print(f"HOME COURT ADVANTAGE")
        print(f"{'='*70}\n")
        
        valid_games = df[df['home_points'].notna()]
        
        if len(valid_games) > 0:
            home_wins = valid_games['home_won'].sum()
            total = len(valid_games)
            home_pct = (home_wins / total * 100)
            
            print(f"Home wins: {home_wins}/{total} ({home_pct:.1f}%)")
            print(f"Expected (no advantage): 50%")
            print(f"Actual advantage: {home_pct - 50:+.1f}%")
    
    # Scoring analysis
    if 'home_recent_ppg' in df.columns and 'home_points' in df.columns:
        print(f"\n{'='*70}")
        print(f"PREDICTION ACCURACY")
        print(f"{'='*70}\n")
        
        valid = df[df['home_points'].notna() & (df['home_points'] > 0)]
        
        if len(valid) > 0:
            home_pred_error = abs(valid['home_recent_ppg'] - valid['home_points']).mean()
            away_pred_error = abs(valid['away_recent_ppg'] - valid['away_points']).mean()
            
            print(f"Home team PPG prediction error: ±{home_pred_error:.1f} points")
            print(f"Away team PPG prediction error: ±{away_pred_error:.1f} points")
    
    # Odds analysis (if available)
    if 'has_odds' in df.columns:
        df_with_odds = df[df['has_odds'] == True]
        
        if len(df_with_odds) > 0:
            print(f"\n{'='*70}")
            print(f"ODDS ANALYSIS ({len(df_with_odds)} games)")
            print(f"{'='*70}\n")
            
            def moneyline_to_prob(odds):
                if pd.isna(odds):
                    return None
                if odds > 0:
                    return 100 / (odds + 100)
                else:
                    return abs(odds) / (abs(odds) + 100)
            
            df_with_odds['home_implied_prob'] = df_with_odds['moneyline_home'].apply(moneyline_to_prob)
            df_with_odds['away_implied_prob'] = df_with_odds['moneyline_away'].apply(moneyline_to_prob)
            
            correct_favorites = 0
            total_with_probs = 0
            
            for _, row in df_with_odds.iterrows():
                if pd.notna(row['home_implied_prob']) and pd.notna(row['away_implied_prob']) and pd.notna(row['home_won']):
                    total_with_probs += 1
                    
                    favorite_home = row['home_implied_prob'] > row['away_implied_prob']
                    home_won = row['home_won'] == 1
                    
                    if favorite_home == home_won:
                        correct_favorites += 1
            
            if total_with_probs > 0:
                accuracy = (correct_favorites / total_with_probs * 100)
                print(f"Favorite wins: {correct_favorites}/{total_with_probs} ({accuracy:.1f}%)")
                print(f"Upset rate: {100 - accuracy:.1f}%")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
  main()