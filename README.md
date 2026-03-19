# Basketball Predictions

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NBA](https://img.shields.io/badge/NBA-Stats-orange?style=flat)

> An NBA prediction pipeline with pre-match feature engineering, historical database building, odds scraping, and ensemble ML models.

## About

A comprehensive NBA basketball prediction system that fetches game data from the SportsRadar API, builds historical feature databases, scrapes betting odds, and trains ensemble ML models to predict game outcomes. Features include pre-match feature engineering (team form, comparative stats, recent performance), historical data reconstruction, and profit/loss tracking.

## Tech Stack

- **Language:** Python 3
- **ML:** scikit-learn, XGBoost
- **Data:** Pandas, NumPy
- **API:** SportsRadar NBA API
- **Web Scraping:** Selenium, BeautifulSoup

## Features

- **Pre-match feature engineering** — team form, recent stats, comparative metrics from SportsRadar API
- **Historical database builder** — reconstructs pre-match features for training from past games
- **Odds scraping** — fetches betting odds from sportsbook websites via Selenium
- **Ensemble ML models** — combines multiple models for game outcome predictions
- **Profit/loss tracking** — evaluates prediction performance against odds
- **CSV data pipeline** — exports structured datasets for analysis

## Getting Started

### Prerequisites

- Python 3.8+
- SportsRadar API key

### Installation

```bash
git clone https://github.com/iampreetdave-max/basketball.git
cd basketball
pip install pandas numpy scikit-learn xgboost selenium beautifulsoup4 requests
```

### Run

**Build historical database:**

```bash
python databaseBuilder.py
```

**Generate pre-match features for upcoming games:**

```bash
python PreMatch.py
```

**Run predictions with ensemble models:**

```bash
python ensWITHpl.py
```

## Project Structure

```
basketball/
├── PreMatch.py          # Pre-match feature engineering
├── databaseBuilder.py   # Historical data builder
├── ensWITHpl.py         # Ensemble model with P/L tracking
├── anba.py              # Advanced NBA analysis
├── ml_nomodel.py        # ML pipeline without pre-trained models
├── scrapeODDS.py        # Odds scraping via Selenium
├── upcoming odds.py     # Upcoming game odds fetcher
├── combineCSVs.py       # CSV data combiner
├── NBA_perfect.csv      # Historical dataset
├── NBA (2).csv          # Game data
├── LICENSE
└── README.md
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
