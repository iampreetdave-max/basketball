"""
NBA Historical Games Feature Engineering
Fetches historical NBA games and constructs pre-match features for ML training.
Properly extracts actual game scores from game summaries.
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ============================================================================
# API CONFIGURATION
# ============================================================================

NBA_API_KEY = "QX0NQvDcyoOD1ezA00fte73Mp8EMDKNxpOZmhxod"
NBA_BASE_URL = "https://api.sportradar.us/nba/trial/v8/en"

NBA_REQUEST_DELAY = 1.1
# ============================================================================
# DOWNLOAD CONFIGURATION
# ============================================================================

NUM_MATCHES = 350  # Change this value to download a different number of matches


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


def main():
    """Main execution function with deduplication"""
    
    # Configuration
    CONFIG = {
        'num_games': NUM_MATCHES,  # Uses the NUM_MATCHES variable from top
        'year': 2023,              # Season year
        'season_type': 'REG',      # Regular season
        'recent_games_count': 5,   # Last N games for features
    }
    
    # Initialize engine
    engine = HistoricalFeatureEngine(NBA_API_KEY)
    
    print(f"\n{'='*70}")
    print(f"FETCHING HISTORICAL NBA GAMES")
    print(f"{'='*70}\n")
    
    # Load existing data if it exists
    existing_ids = set()
    df_existing = None
    
    if os.path.exists('ball_data.csv'):
        try:
            df_existing = pd.read_csv('ball_data.csv')
            existing_ids = set(df_existing['match_id'].astype(str).unique())
            print(f"✓ Found existing data with {len(existing_ids)} games\n")
        except Exception as e:
            print(f"⚠️  Error loading existing data: {e}\n")
    else:
        print(f"📁 Creating new ball_data.csv file\n")
    
    start_time = time.time()
    
    # Get season schedule
    all_games = engine.get_season_schedule(CONFIG['year'], CONFIG['season_type'])
    
    if not all_games:
        print("❌ Failed to fetch season schedule")
        return
    
    print(f"✓ Found {len(all_games)} total games\n")
    
    # Filter to completed games
    completed_games = engine.filter_completed_games(all_games, CONFIG['num_games'])
    
    if not completed_games:
        print("❌ No completed games found")
        return
    
    print(f"✓ Found {len(completed_games)} completed games")
    
    # Filter out already fetched games
    games_to_process = [g for g in completed_games if str(g.get('id', '')) not in existing_ids]
    
    if not games_to_process:
        print(f"✓ All {len(completed_games)} games already in ball_data.csv")
        print(f"No new games to fetch\n")
        return
    
    print(f"Processing {len(games_to_process)} new games (skipping {len(completed_games) - len(games_to_process)} duplicates)...\n")
    
    # Process each game
    games_data = []
    for game in games_to_process:
        try:
            enriched_game = engine.enrich_historical_game(
                game, 
                all_games, 
                CONFIG['recent_games_count']
            )
            games_data.append(enriched_game)
        except Exception as e:
            print(f"    ⚠️  Error processing game: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    if not games_data:
        print("\n❌ No valid games processed")
        return
    
    # Create DataFrame for new games
    df_new = pd.DataFrame(games_data)
    
    # Combine with existing data
    if df_existing is not None:
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"\n✓ Combined {len(df_existing)} existing + {len(df_new)} new = {len(df_final)} total games")
    else:
        df_final = df_new
        print(f"\n✓ Created {len(df_final)} new games")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total games in ball_data.csv: {len(df_final)}")
    print(f"Total columns: {len(df_final.columns)}")
    print(f"NBA API requests: {engine.request_count}")
    print(f"Fetch time: {elapsed/60:.1f} minutes")
    if len(df_new) > 0:
        print(f"Avg per game: {elapsed/len(df_new):.1f} seconds")
    
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
            print(f"⚠️  Zero scores: {zero_scores} games")
        
        if valid_scores > 0:
            print(f"\nLatest sample scores:")
            sample = df_final[df_final['home_points'].notna() & (df_final['home_points'] > 0)].tail(3)
            for _, row in sample.iterrows():
                print(f"  {row['away_alias']} {int(row['away_points'])} @ {row['home_alias']} {int(row['home_points'])}")
    
    # Column breakdown
    print(f"\n{'='*70}")
    print(f"COLUMN CATEGORIES")
    print(f"{'='*70}")
    
    basic_cols = [c for c in df_final.columns if not c.startswith(('home_', 'away_'))]
    home_cols = [c for c in df_final.columns if c.startswith('home_')]
    away_cols = [c for c in df_final.columns if c.startswith('away_')]
    
    print(f"Basic info: {len(basic_cols)} columns")
    print(f"Home team: {len(home_cols)} columns")
    print(f"Away team: {len(away_cols)} columns")
    
    # Save to ball_data.csv
    df_final.to_csv('ball_data.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Saved to: ball_data.csv")
    print(f"{'='*70}\n")
    
    available_cols = [
        'home_alias', 'away_alias', 'scheduled',
        'home_recent_win_pct', 'away_recent_win_pct',
        'home_points', 'away_points', 'home_won'
    ]
    
    available_cols = [c for c in available_cols if c in df_final.columns]
    
    if available_cols:
        print("Latest games in ball_data.csv:")
        print(df_final[available_cols].tail(5).to_string(index=False))
    
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
            
            if 'point_differential' in valid_games.columns:
                avg_diff = valid_games['point_differential'].abs().mean()
                print(f"Average point differential: {avg_diff:.1f}")
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
    print("  3. Train ML models: predict winners")
    print("  4. Backtest strategies: simulate betting based on features")
    
    print(f"\n📁 Key columns for analysis:")
    key_features = [
        'home_recent_win_pct', 'away_recent_win_pct',
        'scoring_advantage_home', 'form_advantage_home',
        'defensive_advantage_home', 'ball_control_advantage_home',
        'home_points', 'away_points', 'home_won', 'point_differential'
    ]
    
    for col in key_features:
        if col in df_final.columns:
            print(f"  ✓ {col}")
    
    print(f"\n{'='*70}")
    print(f"NOTES")
    print(f"{'='*70}")
    print("✓ Pre-match features are from BEFORE each game")
    print("✓ Actual results are included (home_points, away_points, home_won)")
    print("✓ Data is accumulated in ball_data.csv (no duplicates)")
    print("✓ Ready for ML training and backtesting!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
  main()
