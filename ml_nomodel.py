import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('nba_split.csv')

# Filter rows with necessary data
df = df[
    df['home_recent_ppg'].notna() &
    df['away_recent_ppg'].notna() &
    df['total_line'].notna() &
    df['home_winning_odds_decimal'].notna() &
    df['away_winning_odds_decimal'].notna() &
    df['home_points'].notna() &
    df['away_points'].notna()
].copy()

print(f"Working with {len(df)} games with complete data\n")

# Convert decimal odds to implied probability
def decimal_odds_to_probability(odds):
    """Convert decimal odds to implied probability"""
    if pd.isna(odds) or odds == 0:
        return 0.5
    return 1 / odds

# Main prediction function
def predict_scores(row):
    """
    Predict home and away scores based on:
    1. Total line as the base
    2. Odds as probability distribution
    3. Features for adjustments
    """

    total_line = row['total_line']

    # Step 1: Convert odds to implied probabilities
    home_prob = decimal_odds_to_probability(row['home_winning_odds_decimal'])
    away_prob = decimal_odds_to_probability(row['away_winning_odds_decimal'])

    # Normalize probabilities
    total_prob = home_prob + away_prob
    home_prob_norm = home_prob / total_prob
    away_prob_norm = away_prob / total_prob

    # Step 2: Distribute total_line based on odds probabilities
    predicted_home = total_line * home_prob_norm
    predicted_away = total_line * away_prob_norm

    # Step 3: Adjust based on recent PPG (relative performance)
    home_ppg = row['home_recent_ppg']
    away_ppg = row['away_recent_ppg']
    ppg_total = home_ppg + away_ppg

    if ppg_total > 0:
        # Weight by recent performance
        home_ppg_ratio = home_ppg / ppg_total
        away_ppg_ratio = away_ppg / ppg_total

        # Blend with odds-based distribution (60% odds, 40% recent form)
        predicted_home = predicted_home * 0.6 + (total_line * home_ppg_ratio) * 0.4
        predicted_away = predicted_away * 0.6 + (total_line * away_ppg_ratio) * 0.4

    # Step 4: Adjust based on defensive quality (opponent PPG)
    if pd.notna(row['home_recent_opp_ppg']):
        # Opponent PPG reflects defense quality (higher = worse defense)
        # If away team's defense is weak, home should score more
        defense_adjustment = (row['home_recent_opp_ppg'] - 110) * 0.05
        predicted_home += defense_adjustment

    if pd.notna(row['away_recent_opp_ppg']):
        # Opponent PPG reflects defense quality
        # If home team's defense is weak, away should score more
        defense_adjustment = (row['away_recent_opp_ppg'] - 110) * 0.05
        predicted_away += defense_adjustment

    # Step 5: Adjust based on point differential (momentum/performance trend)
    if pd.notna(row['home_recent_point_diff']):
        predicted_home += row['home_recent_point_diff'] * 0.08

    if pd.notna(row['away_recent_point_diff']):
        predicted_away += row['away_recent_point_diff'] * 0.08

    # Step 6: Apply scoring advantage metric
    if pd.notna(row['scoring_advantage_home']):
        predicted_home += row['scoring_advantage_home'] * 0.15
        predicted_away -= row['scoring_advantage_home'] * 0.15

    # Step 7: Ensure predictions sum to approximately total_line
    final_total = predicted_home + predicted_away
    if final_total > 0:
        scale = total_line / final_total
        predicted_home = predicted_home * scale
        predicted_away = predicted_away * scale

    # Ensure no negative values
    predicted_home = max(0, predicted_home)
    predicted_away = max(0, predicted_away)

    return predicted_home, predicted_away

# Generate predictions
print("Generating predictions...")
df[['pred_home_points', 'pred_away_points']] = df.apply(
    lambda row: pd.Series(predict_scores(row)), axis=1
)

# Calculate prediction errors
df['home_error'] = abs(df['pred_home_points'] - df['home_points'])
df['away_error'] = abs(df['pred_away_points'] - df['away_points'])

# Accuracy check functions
def check_within_threshold(pred, actual, threshold):
    return abs(pred - actual) <= threshold

# Points accuracy at ±1.0 threshold
df['home_acc_1'] = df.apply(
    lambda row: check_within_threshold(row['pred_home_points'], row['home_points'], 1.0), axis=1
)
df['away_acc_1'] = df.apply(
    lambda row: check_within_threshold(row['pred_away_points'], row['away_points'], 1.0), axis=1
)
df['both_acc_1'] = df['home_acc_1'] & df['away_acc_1']

# Points accuracy at ±0.5 threshold
df['home_acc_05'] = df.apply(
    lambda row: check_within_threshold(row['pred_home_points'], row['home_points'], 0.5), axis=1
)
df['away_acc_05'] = df.apply(
    lambda row: check_within_threshold(row['pred_away_points'], row['away_points'], 0.5), axis=1
)
df['both_acc_05'] = df['home_acc_05'] & df['away_acc_05']

# Winner prediction
df['pred_winner'] = df.apply(
    lambda row: 'HOME' if row['pred_home_points'] > row['pred_away_points'] else 'AWAY', axis=1
)

# Actual winner
df['actual_winner'] = df.apply(
    lambda row: 'HOME' if row['home_won'] == 1 else 'AWAY', axis=1
)

# Winner accuracy
df['winner_correct'] = df['pred_winner'] == df['actual_winner']

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("POINTS PREDICTION ACCURACY")
print("="*60)

print(f"\n±1.0 POINT THRESHOLD:")
print(f"  Home Points:    {df['home_acc_1'].sum():3d}/{len(df)} ({df['home_acc_1'].mean()*100:5.2f}%)")
print(f"  Away Points:    {df['away_acc_1'].sum():3d}/{len(df)} ({df['away_acc_1'].mean()*100:5.2f}%)")
print(f"  Both Correct:   {df['both_acc_1'].sum():3d}/{len(df)} ({df['both_acc_1'].mean()*100:5.2f}%)")

print(f"\n±0.5 POINT THRESHOLD:")
print(f"  Home Points:    {df['home_acc_05'].sum():3d}/{len(df)} ({df['home_acc_05'].mean()*100:5.2f}%)")
print(f"  Away Points:    {df['away_acc_05'].sum():3d}/{len(df)} ({df['away_acc_05'].mean()*100:5.2f}%)")
print(f"  Both Correct:   {df['both_acc_05'].sum():3d}/{len(df)} ({df['both_acc_05'].mean()*100:5.2f}%)")

print(f"\nPOINT PREDICTION ERROR STATS:")
print(f"  Home Avg Error: {df['home_error'].mean():.2f} points")
print(f"  Away Avg Error: {df['away_error'].mean():.2f} points")

print("\n" + "="*60)
print("WINNER PREDICTION ACCURACY")
print("="*60)
print(f"  Correct: {df['winner_correct'].sum():3d}/{len(df)} ({df['winner_correct'].mean()*100:5.2f}%)")

# ============================================
# DETAILED OUTPUT SAMPLE
# ============================================

print("\n" + "="*60)
print("SAMPLE PREDICTIONS (First 15 games)")
print("="*60)

sample_cols = [
    'home_name', 'away_name', 'total_line',
    'pred_home_points', 'pred_away_points',
    'home_points', 'away_points',
    'home_error', 'away_error',
    'pred_winner', 'actual_winner', 'winner_correct'
]

sample_df = df[sample_cols].head(15).copy()
sample_df['pred_home_points'] = sample_df['pred_home_points'].round(1)
sample_df['pred_away_points'] = sample_df['pred_away_points'].round(1)
sample_df['home_error'] = sample_df['home_error'].round(1)
sample_df['away_error'] = sample_df['away_error'].round(1)

print(sample_df.to_string(index=False))

# ============================================
# SAVE DETAILED RESULTS
# ============================================

output_df = df[[
    'match_id', 'home_name', 'away_name', 'date',
    'total_line',
    'pred_home_points', 'pred_away_points',
    'home_points', 'away_points',
    'home_error', 'away_error',
    'home_acc_1', 'home_acc_05', 'away_acc_1', 'away_acc_05',
    'both_acc_1', 'both_acc_05',
    'pred_winner', 'actual_winner', 'winner_correct',
    'home_winning_odds_decimal', 'away_winning_odds_decimal'
]].copy()

output_df = output_df.round(2)
output_df.to_csv('nba_predictions_results.csv', index=False)
print(f"\n✓ Detailed results saved to: nba_predictions_results.csv")

print("\n" + "="*60)
