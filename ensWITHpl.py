import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NBA PREDICTIONS - OPTIMIZED GRADIENT BOOSTING ENSEMBLE")
print("="*80)

# ============================================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================================
print("\n[1/5] LOADING DATA...")
df = pd.read_csv('NBA_perfect.csv')
df = df[df['home_insufficient_data'] == False].copy()
df = df[df['away_insufficient_data'] == False].copy()
print(f"Dataset: {df.shape[0]} samples")

# Feature Engineering
df_fe = df.copy()

home_stats = ['home_recent_points', 'home_recent_field_goals_pct', 'home_recent_three_points_pct',
              'home_recent_free_throws_pct', 'home_recent_assists', 'home_recent_steals',
              'home_recent_blocks', 'home_recent_offensive_rebounds', 'home_recent_defensive_rebounds',
              'home_recent_wins', 'home_recent_losses', 'home_recent_win_pct', 'home_recent_ppg',
              'home_recent_opp_ppg', 'home_recent_point_diff']

away_stats = ['away_recent_points', 'away_recent_field_goals_pct', 'away_recent_three_points_pct',
              'away_recent_free_throws_pct', 'away_recent_assists', 'away_recent_steals',
              'away_recent_blocks', 'away_recent_offensive_rebounds', 'away_recent_defensive_rebounds',
              'away_recent_wins', 'away_recent_losses', 'away_recent_win_pct', 'away_recent_ppg',
              'away_recent_opp_ppg', 'away_recent_point_diff']

df_fe['home_vs_away_ppg_ratio'] = df_fe['home_recent_ppg'] / (df_fe['away_recent_ppg'] + 1)
df_fe['home_vs_away_defense_ratio'] = df_fe['home_recent_opp_ppg'] / (df_fe['away_recent_opp_ppg'] + 1)
df_fe['home_assist_steal_ratio'] = df_fe['home_recent_assists'] / (df_fe['home_recent_steals'] + 1)
df_fe['away_assist_steal_ratio'] = df_fe['away_recent_assists'] / (df_fe['away_recent_steals'] + 1)
df_fe['home_rebound_efficiency'] = (df_fe['home_recent_offensive_rebounds'] + df_fe['home_recent_defensive_rebounds']) / (df_fe['home_recent_points'] + 1)
df_fe['away_rebound_efficiency'] = (df_fe['away_recent_offensive_rebounds'] + df_fe['away_recent_defensive_rebounds']) / (df_fe['away_recent_points'] + 1)
df_fe['home_recent_momentum'] = df_fe['home_recent_wins'] - df_fe['home_recent_losses']
df_fe['away_recent_momentum'] = df_fe['away_recent_wins'] - df_fe['away_recent_losses']
df_fe['home_scoring_trend_num'] = (df_fe['home_recent_scoring_trend'] == 'up').astype(int)
df_fe['away_scoring_trend_num'] = (df_fe['away_recent_scoring_trend'] == 'up').astype(int)
df_fe['home_defensive_rating'] = df_fe['home_recent_opp_ppg']
df_fe['away_defensive_rating'] = df_fe['away_recent_opp_ppg']
df_fe['home_avg_blocks_steals'] = df_fe['home_recent_blocks'] + df_fe['home_recent_steals']
df_fe['away_avg_blocks_steals'] = df_fe['away_recent_blocks'] + df_fe['away_recent_steals']

for col in ['scoring_advantage_home', 'form_advantage_home', 'defensive_advantage_home', 'ball_control_advantage_home']:
    df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce').fillna(0)

df_fe['ppg_diff_squared'] = (df_fe['home_recent_ppg'] - df_fe['away_recent_ppg']) ** 2
df_fe['home_ft_fg_product'] = df_fe['home_recent_free_throws_pct'] * df_fe['home_recent_field_goals_pct']
df_fe['away_ft_fg_product'] = df_fe['away_recent_free_throws_pct'] * df_fe['away_recent_field_goals_pct']
df_fe['home_better_ft_pct'] = (df_fe['home_recent_free_throws_pct'] > df_fe['away_recent_free_throws_pct']).astype(int)
df_fe['home_better_fg_pct'] = (df_fe['home_recent_field_goals_pct'] > df_fe['away_recent_field_goals_pct']).astype(int)
df_fe['home_better_3p_pct'] = (df_fe['home_recent_three_points_pct'] > df_fe['away_recent_three_points_pct']).astype(int)

df_fe = df_fe.fillna(0)

exclude_cols = ['match_id', 'game_identifier', 'scheduled', 'venue_name', 'venue_city',
                'home_id', 'home_name', 'home_alias', 'home_market', 'away_id', 'away_name',
                'away_alias', 'away_market', 'home_insufficient_data', 'away_insufficient_data',
                'home_points', 'away_points', 'total_points', 'point_differential', 'home_won',
                'away_won', 'date', 'home_games_played', 'away_games_played', 
                'home_recent_scoring_trend', 'away_recent_scoring_trend']

all_features = [col for col in df_fe.columns if col not in exclude_cols]
best_50_features = all_features[:50]

X = df_fe[best_50_features].values
y_home = df_fe['home_points'].values
y_away = df_fe['away_points'].values
y_winner = df_fe['home_won'].astype(int).values

print(f"Features: {len(best_50_features)}")

# ============================================================================
# 2. SCALE & SPLIT DATA
# ============================================================================
print("[2/5] PREPROCESSING DATA...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_h_train, y_h_test, y_a_train, y_a_test, y_w_train, y_w_test = train_test_split(
    X_scaled, y_home, y_away, y_winner, test_size=0.2, random_state=42
)

n_features = X_train.shape[1]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {n_features}")

# ============================================================================
# 3. TRAIN OPTIMIZED MODELS
# ============================================================================
print("[3/5] TRAINING OPTIMIZED MODELS...")

# HOME POINTS - 3 Model Ensemble (GB, XGB, RF)
print("\n  Training Home Points...")
gb_home = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=8,
                                     random_state=42, subsample=0.85, min_samples_leaf=1)
gb_home.fit(X_train, y_h_train)

xgb_home = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=8,
                         random_state=42, subsample=0.85, colsample_bytree=0.8)
xgb_home.fit(X_train, y_h_train, verbose=False)

rf_home = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42,
                                min_samples_leaf=1, n_jobs=-1)
rf_home.fit(X_train, y_h_train)

y_h_pred_gb = gb_home.predict(X_test)
y_h_pred_xgb = xgb_home.predict(X_test)
y_h_pred_rf = rf_home.predict(X_test)

# Ensemble: Weighted average
y_h_pred = (y_h_pred_gb * 0.4 + y_h_pred_xgb * 0.4 + y_h_pred_rf * 0.2)
h_r2 = r2_score(y_h_test, y_h_pred)
h_mae = mean_absolute_error(y_h_test, y_h_pred)

# Confidence from model agreement
h_pred_std = np.std([y_h_pred_gb, y_h_pred_xgb, y_h_pred_rf], axis=0)
h_confidence = 100 / (1 + h_pred_std)
h_confidence = np.clip(h_confidence, 10, 90)

# AWAY POINTS - 3 Model Ensemble
print("  Training Away Points...")
gb_away = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=8,
                                     random_state=42, subsample=0.85, min_samples_leaf=1)
gb_away.fit(X_train, y_a_train)

xgb_away = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=8,
                         random_state=42, subsample=0.85, colsample_bytree=0.8)
xgb_away.fit(X_train, y_a_train, verbose=False)

rf_away = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42,
                                min_samples_leaf=1, n_jobs=-1)
rf_away.fit(X_train, y_a_train)

y_a_pred_gb = gb_away.predict(X_test)
y_a_pred_xgb = xgb_away.predict(X_test)
y_a_pred_rf = rf_away.predict(X_test)

y_a_pred = (y_a_pred_gb * 0.4 + y_a_pred_xgb * 0.4 + y_a_pred_rf * 0.2)
a_r2 = r2_score(y_a_test, y_a_pred)
a_mae = mean_absolute_error(y_a_test, y_a_pred)

a_pred_std = np.std([y_a_pred_gb, y_a_pred_xgb, y_a_pred_rf], axis=0)
a_confidence = 100 / (1 + a_pred_std)
a_confidence = np.clip(a_confidence, 10, 90)

def classify_confidence(confidence_scores):
    return ['Low' if c < 50 else 'Medium' if c < 75 else 'High' for c in confidence_scores]

h_conf_class = classify_confidence(h_confidence)
a_conf_class = classify_confidence(a_confidence)

# MONEYLINE - Gradient Boosting Classifier
print("  Training Moneyline...")
gb_winner = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7,
                                       random_state=42, subsample=0.85)
gb_winner.fit(X_train, y_w_train)

xgb_winner = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=7,
                            random_state=42, subsample=0.85)
xgb_winner.fit(X_train, y_w_train, verbose=False)

y_w_pred_gb = gb_winner.predict_proba(X_test)[:, 1]
y_w_pred_xgb = xgb_winner.predict_proba(X_test)[:, 1]
y_w_pred_prob = (y_w_pred_gb * 0.6 + y_w_pred_xgb * 0.4)
y_w_pred = (y_w_pred_prob > 0.5).astype(int)
w_acc = accuracy_score(y_w_test, y_w_pred)

# Confidence from prediction margin (distance from 0.5)
w_confidence_per_sample = 100 * np.abs(y_w_pred_prob - 0.5) * 2
w_confidence_per_sample = np.clip(w_confidence_per_sample, 15, 85)
w_conf_class = classify_confidence(w_confidence_per_sample)

# ============================================================================
# 4. OVER/UNDER PREDICTION
# ============================================================================
print("[4/5] COMPUTING OVER/UNDER & CONFIDENCE...")

OVER_UNDER_THRESHOLD = 227.4
total_pred = y_h_pred + y_a_pred
ou_pred = (total_pred >= OVER_UNDER_THRESHOLD).astype(int)
y_ou_actual = (y_h_test + y_a_test >= OVER_UNDER_THRESHOLD).astype(int)
ou_acc = accuracy_score(y_ou_actual, ou_pred)

# O/U confidence from combined home+away confidence
ou_confidence = (h_confidence + a_confidence) / 2
ou_conf_class = classify_confidence(ou_confidence)

# ============================================================================
# 5. SAVE PREDICTIONS
# ============================================================================
print("[5/5] SAVING RESULTS...\n")

# Calculate P/L columns
winner_pl = np.where(y_w_pred == y_w_test, 1, -1)
ou_pl = np.where(ou_pred == y_ou_actual, 1, -1)

predictions_df = pd.DataFrame({
    'home_points_actual': y_h_test,
    'home_points_predicted': y_h_pred.astype(int),
    'home_confidence': h_confidence.astype(int),
    'home_confidence_class': h_conf_class,
    'away_points_actual': y_a_test,
    'away_points_predicted': y_a_pred.astype(int),
    'away_confidence': a_confidence.astype(int),
    'away_confidence_class': a_conf_class,
    'total_points_actual': y_h_test + y_a_test,
    'total_points_predicted': total_pred.astype(int),
    'ou_actual': y_ou_actual,
    'ou_predicted': ou_pred,
    'ou_confidence': ou_confidence.astype(int),
    'ou_confidence_class': ou_conf_class,
    'ou_pl': ou_pl,
    'winner_actual': y_w_test,
    'winner_predicted': y_w_pred,
    'winner_confidence': w_confidence_per_sample.astype(int),
    'winner_confidence_class': w_conf_class,
    'winner_pl': winner_pl
})

predictions_df.to_csv('PREDICTIONS.csv', index=False)

print("="*80)
print("RESULTS - OPTIMIZED ENSEMBLE")
print("="*80)

def calc_pl_by_category(pl_array, conf_class_array):
    low_pl = pl_array[np.array(conf_class_array) == 'Low']
    med_pl = pl_array[np.array(conf_class_array) == 'Medium']
    high_pl = pl_array[np.array(conf_class_array) == 'High']
    
    low_profit = np.sum(low_pl) if len(low_pl) > 0 else 0
    med_profit = np.sum(med_pl) if len(med_pl) > 0 else 0
    high_profit = np.sum(high_pl) if len(high_pl) > 0 else 0
    
    low_acc = (np.sum(low_pl == 1) / len(low_pl)) if len(low_pl) > 0 else 0
    med_acc = (np.sum(med_pl == 1) / len(med_pl)) if len(med_pl) > 0 else 0
    high_acc = (np.sum(high_pl == 1) / len(high_pl)) if len(high_pl) > 0 else 0
    
    return (low_profit, med_profit, high_profit), (low_acc, med_acc, high_acc)

h_within_5 = np.mean(np.abs(y_h_pred - y_h_test) <= 5)
h_within_10 = np.mean(np.abs(y_h_pred - y_h_test) <= 10)
a_within_5 = np.mean(np.abs(y_a_pred - y_a_test) <= 5)
a_within_10 = np.mean(np.abs(y_a_pred - y_a_test) <= 10)

h_low = np.sum(np.array(h_conf_class) == 'Low')
h_med = np.sum(np.array(h_conf_class) == 'Medium')
h_high = np.sum(np.array(h_conf_class) == 'High')

a_low = np.sum(np.array(a_conf_class) == 'Low')
a_med = np.sum(np.array(a_conf_class) == 'Medium')
a_high = np.sum(np.array(a_conf_class) == 'High')

ou_low = np.sum(np.array(ou_conf_class) == 'Low')
ou_med = np.sum(np.array(ou_conf_class) == 'Medium')
ou_high = np.sum(np.array(ou_conf_class) == 'High')

w_low = np.sum(np.array(w_conf_class) == 'Low')
w_med = np.sum(np.array(w_conf_class) == 'Medium')
w_high = np.sum(np.array(w_conf_class) == 'High')

# Calculate P/L by confidence category
ou_pl_by_cat, ou_acc_by_cat = calc_pl_by_category(ou_pl, ou_conf_class)
w_pl_by_cat, w_acc_by_cat = calc_pl_by_category(winner_pl, w_conf_class)

print(f"\nHOME POINTS (GB 40% + XGB 40% + RF 20%)")
print(f"  R² Score: {h_r2:.4f}")
print(f"  MAE:      {h_mae:.2f} points")
print(f"  Within ±5:  {h_within_5*100:.1f}%")
print(f"  Within ±10: {h_within_10*100:.1f}%")
print(f"  Confidence: Low={h_low} | Medium={h_med} | High={h_high}")
print(f"  Avg Conf:   {h_confidence.mean():.1f}%")

print(f"\nAWAY POINTS (GB 40% + XGB 40% + RF 20%)")
print(f"  R² Score: {a_r2:.4f}")
print(f"  MAE:      {a_mae:.2f} points")
print(f"  Within ±5:  {a_within_5*100:.1f}%")
print(f"  Within ±10: {a_within_10*100:.1f}%")
print(f"  Confidence: Low={a_low} | Medium={a_med} | High={a_high}")
print(f"  Avg Conf:   {a_confidence.mean():.1f}%")

print(f"\nOVER/UNDER {OVER_UNDER_THRESHOLD} (Combined Confidence)")
print(f"  Accuracy: {ou_acc:.4f} ({ou_acc*100:.1f}%)")
ou_over = np.sum(ou_pred)
ou_under = len(ou_pred) - ou_over
ou_actual_over = np.sum(y_ou_actual)
ou_actual_under = len(y_ou_actual) - ou_actual_over
print(f"  Predicted: {ou_over} Over, {ou_under} Under")
print(f"  Actual:    {ou_actual_over} Over, {ou_actual_under} Under")
print(f"  Total Profit/Loss: {np.sum(ou_pl)}")
print(f"  By Confidence:")
print(f"    Low (n={ou_low}):    {ou_pl_by_cat[0]:+3d} units | Accuracy: {ou_acc_by_cat[0]*100:5.1f}%")
print(f"    Medium (n={ou_med}): {ou_pl_by_cat[1]:+3d} units | Accuracy: {ou_acc_by_cat[1]*100:5.1f}%")
print(f"    High (n={ou_high}):  {ou_pl_by_cat[2]:+3d} units | Accuracy: {ou_acc_by_cat[2]*100:5.1f}%")



print(f"\nMONEYLINE - WINNER (GB 60% + XGB 40%)")
print(f"  Accuracy: {w_acc:.4f} ({w_acc*100:.1f}%)")
print(f"  Total Profit/Loss: {np.sum(winner_pl)}")
print(f"  By Confidence:")
print(f"    Low (n={w_low}):    {w_pl_by_cat[0]:+3d} units | Accuracy: {w_acc_by_cat[0]*100:5.1f}%")
print(f"    Medium (n={w_med}):  {w_pl_by_cat[1]:+3d} units | Accuracy: {w_acc_by_cat[1]*100:5.1f}%")
print(f"    High (n={w_high}):   {w_pl_by_cat[2]:+3d} units | Accuracy: {w_acc_by_cat[2]*100:5.1f}%")

print(f"\nPREDICTIONS SAVED: PREDICTIONS.csv ({len(predictions_df)} samples)")
print("="*80)
print("\nMODELS & APPROACH")
print("="*80)
print("""
REGRESSION (Points):
  - Gradient Boosting (GB): 300 estimators, depth=8, lr=0.03
  - XGBoost (XGB): 300 estimators, depth=8, lr=0.03
  - Random Forest (RF): 200 estimators, depth=15
  - Blend: 40% GB + 40% XGB + 20% RF

CLASSIFICATION (Winner/Moneyline):
  - GB Classifier: 200 estimators, depth=7, lr=0.05
  - XGB Classifier: 200 estimators, depth=7, lr=0.05
  - Blend: 60% GB + 40% XGB

CONFIDENCE CALIBRATION:
  - Points: Model disagreement (std across ensemble)
  - Moneyline: Prediction margin from 0.5
  - Over/Under: Combined home+away confidence
""")
print("="*80)