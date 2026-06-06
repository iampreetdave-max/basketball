# NBA Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

A data pipeline that reconstructs pre-match NBA team features from the Sportradar API, merges them with betting odds, and trains models to study and predict game outcomes.

## Overview

This project assembles a machine-learning dataset for NBA games. For each completed game it looks back at each team's recent form *before* that game, derives comparative features, and attaches the actual final score — producing a leak-free training set. It can also pull betting odds (moneyline, spread, total) from the Sportradar Odds Comparison API and join them to the games by a date/teams identifier, enabling odds-versus-outcome analysis and backtesting.

## Key Features

- **Pre-match feature engineering** — for each game, computes each team's recent averages (points, shooting percentages, rebounds, assists, turnovers, etc.), win/loss form, scoring trend, and head-to-head comparative advantages from the prior N games only
- **Historical dataset builder** — walks a season schedule, enriches completed games with pre-match features and the actual result pulled from each game summary, and exports to timestamped CSV
- **Betting odds integration** — fetches moneyline, spread, and total markets from the Sportradar Odds API and merges them onto games via a shared game identifier
- **Built-in analysis** — score-completeness checks, home-court advantage, PPG prediction error, and favorite-vs-outcome accuracy from saved CSVs
- **Rate-limit-aware API client** — retry with backoff on HTTP 429/5xx and request pacing for the Sportradar trial tier
- **CSV utilities** — combine and reshape exported datasets for downstream modeling

## How It Works

1. **Fetch schedule** — pull a season's schedule from the Sportradar NBA API and filter to completed games.
2. **Reconstruct features** — for each game, gather each team's games *before* that date, fetch their summaries, and average the box-score stats into recent-form and comparative features.
3. **Attach results** — read the final score from each game summary (the schedule endpoint omits scores) to label every row.
4. **Merge odds (optional)** — fetch odds for the relevant date range and join on the game identifier.
5. **Export & analyze** — write a timestamped CSV and run the included completeness/accuracy summaries.

## Tech Stack

- **Language:** Python 3
- **Data:** pandas, NumPy
- **HTTP:** requests
- **Data sources:** Sportradar NBA API, Sportradar Odds Comparison API

## Getting Started

### Prerequisites

- Python 3.8+
- A Sportradar API key (NBA, and optionally Odds Comparison)

### Installation

```bash
git clone https://github.com/iampreetdave-max/basketball.git
cd basketball
pip install pandas numpy requests
```

### Configuration

The scripts read Sportradar API keys defined near the top of each file (`NBA_API_KEY`, `ODDS_API_KEY`). Replace these with your own key before running. **Do not commit real API keys** — prefer supplying them via environment variables or a local, gitignored config.

### Run

```bash
python anba.py             # Build the historical games + odds dataset (interactive)
python databaseBuilder.py  # Reconstruct the historical feature database
python PreMatch.py         # Generate pre-match features for upcoming games
```

`anba.py` prints an API-call/time estimate and prompts for confirmation before it begins, then writes a timestamped CSV.

## Project Structure

```
basketball/
├── anba.py                 # Historical games + odds integration pipeline
├── databaseBuilder.py      # Historical pre-match feature database builder
├── PreMatch.py             # Pre-match feature engineering for upcoming games
├── ensWITHpl.py            # Ensemble modeling with profit/loss tracking
├── ml_nomodel.py           # ML pipeline without a pre-trained model
├── scrapeODDS.py           # Odds scraping
├── upcoming odds.py        # Upcoming-game odds fetcher
├── combineCSVs.py          # CSV combiner/reshaper
├── NBA_perfect.csv         # Exported dataset
├── NBA (2).csv             # Game data
├── nba_prematch_features_*.csv  # Sample feature export
├── LICENSE
└── README.md
```

## License

Licensed under the [Apache License 2.0](LICENSE).
