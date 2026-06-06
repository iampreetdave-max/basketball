"""
Unit tests for deterministic, pure-logic helpers in scrapeODDS.py.

Scope (intentionally narrow):
  * Only ``scrapeODDS.py`` is exercised. It is import-safe: at module level it
    defines constants and functions only, and the network-driven workflow lives
    under ``main()`` guarded by ``if __name__ == "__main__"``. No Sportradar HTTP
    call or CSV/DB access happens on import.
  * ``ml_nomodel.py`` is deliberately NOT imported here: it runs
    ``pd.read_csv('nba_split.csv')`` at module top level, so importing it would
    perform file I/O and crash without that data file. Its helpers therefore
    cannot be honestly unit-tested without first refactoring the module to be
    import-safe.

Everything below feeds hand-built dict fixtures into the real
``extract_draftkings_odds`` function and asserts its actual return values. No
network is touched.
"""

import os
import sys

import pytest

# Make the repo root importable regardless of where pytest is invoked from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

scrapeODDS = pytest.importorskip("scrapeODDS")

extract_draftkings_odds = scrapeODDS.extract_draftkings_odds
TEAM_ALIASES = scrapeODDS.TEAM_ALIASES

DK_BOOK_ID = "sr:book:18149"
WINNER_MARKET = "sr:market:219"
TOTAL_MARKET = "sr:market:225"


def _winner_market(home_odds, away_odds, book_id=DK_BOOK_ID):
    return {
        "id": WINNER_MARKET,
        "books": [
            {
                "id": book_id,
                "outcomes": [
                    {"type": "home", "odds_decimal": home_odds},
                    {"type": "away", "odds_decimal": away_odds},
                ],
            }
        ],
    }


def _total_market(over_odds, under_odds, total_line, book_id=DK_BOOK_ID):
    return {
        "id": TOTAL_MARKET,
        "books": [
            {
                "id": book_id,
                "outcomes": [
                    {"type": "over", "odds_decimal": over_odds, "total": total_line},
                    {"type": "under", "odds_decimal": under_odds},
                ],
            }
        ],
    }


def _event(home_name, away_name, markets):
    # competitors[0] == home, competitors[1] == away (per source contract).
    return {
        "sport_event": {
            "competitors": [
                {"name": home_name},
                {"name": away_name},
            ]
        },
        "markets": markets,
    }


def _payload(events):
    return {"sport_schedule_sport_event_markets": events}


# --------------------------------------------------------------------------
# TEAM_ALIASES: static mapping sanity
# --------------------------------------------------------------------------

def test_team_aliases_known_mappings():
    assert TEAM_ALIASES["Boston Celtics"] == "BOS"
    assert TEAM_ALIASES["LA Lakers"] == "LAL"
    assert TEAM_ALIASES["Golden State Warriors"] == "GSW"


# --------------------------------------------------------------------------
# Empty / malformed input handling
# --------------------------------------------------------------------------

def test_returns_empty_for_none():
    assert extract_draftkings_odds(None, "2024-01-01") == {}


def test_returns_empty_when_key_missing():
    assert extract_draftkings_odds({"something_else": []}, "2024-01-01") == {}


def test_returns_empty_for_no_events():
    assert extract_draftkings_odds(_payload([]), "2024-01-01") == {}


def test_event_with_fewer_than_two_competitors_is_skipped():
    bad_event = {
        "sport_event": {"competitors": [{"name": "Boston Celtics"}]},
        "markets": [_winner_market(1.5, 2.5)],
    }
    assert extract_draftkings_odds(_payload([bad_event]), "2024-01-01") == {}


# --------------------------------------------------------------------------
# Winner market (sr:market:219)
# --------------------------------------------------------------------------

def test_winner_odds_extracted_with_correct_game_id():
    payload = _payload(
        [_event("Boston Celtics", "Los Angeles", [_winner_market(1.5, 2.6)])]
    )
    result = extract_draftkings_odds(payload, "2024-01-01")

    # Away alias is built via fallback ("Los"[:3] -> "LOS"), home via mapping.
    away_alias = TEAM_ALIASES.get("Los Angeles", "Los Angeles".replace(" ", "")[:3].upper())
    expected_id = f"2024-01-01_{away_alias}@BOS"

    assert set(result.keys()) == {expected_id}
    game = result[expected_id]
    assert game["game_identifier"] == expected_id
    assert game["home_winning_odds_decimal"] == 1.5
    assert game["away_winning_odds_decimal"] == 2.6


def test_known_team_aliases_drive_game_id():
    payload = _payload(
        [_event("Boston Celtics", "Miami Heat", [_winner_market(1.4, 3.0)])]
    )
    result = extract_draftkings_odds(payload, "2025-02-15")
    assert "2025-02-15_MIA@BOS" in result


# --------------------------------------------------------------------------
# Total market (sr:market:225)
# --------------------------------------------------------------------------

def test_total_odds_and_line_extracted():
    payload = _payload(
        [_event("Miami Heat", "Chicago Bulls", [_total_market(1.91, 1.89, 213.5)])]
    )
    result = extract_draftkings_odds(payload, "2024-03-03")
    game = result["2024-03-03_CHI@MIA"]
    assert game["over_odds_decimal"] == 1.91
    assert game["under_odds_decimal"] == 1.89
    assert game["total_line"] == 213.5


def test_winner_and_total_combined():
    payload = _payload(
        [
            _event(
                "Denver Nuggets",
                "Phoenix Suns",
                [_winner_market(1.7, 2.2), _total_market(1.95, 1.85, 228.0)],
            )
        ]
    )
    game = extract_draftkings_odds(payload, "2024-04-04")["2024-04-04_PHX@DEN"]
    assert game["home_winning_odds_decimal"] == 1.7
    assert game["away_winning_odds_decimal"] == 2.2
    assert game["over_odds_decimal"] == 1.95
    assert game["under_odds_decimal"] == 1.85
    assert game["total_line"] == 228.0


# --------------------------------------------------------------------------
# Filtering: only DraftKings book + only the two relevant markets
# --------------------------------------------------------------------------

def test_non_draftkings_book_is_ignored():
    payload = _payload(
        [_event("Boston Celtics", "Miami Heat",
                [_winner_market(1.5, 2.6, book_id="sr:book:99999")])]
    )
    # No DK odds -> game has only game_identifier -> excluded entirely.
    assert extract_draftkings_odds(payload, "2024-01-01") == {}


def test_irrelevant_market_is_ignored():
    irrelevant = {
        "id": "sr:market:1",
        "books": [
            {
                "id": DK_BOOK_ID,
                "outcomes": [{"type": "home", "odds_decimal": 1.5}],
            }
        ],
    }
    payload = _payload([_event("Boston Celtics", "Miami Heat", [irrelevant])])
    assert extract_draftkings_odds(payload, "2024-01-01") == {}


def test_multiple_games_in_one_payload():
    payload = _payload(
        [
            _event("Boston Celtics", "Miami Heat", [_winner_market(1.4, 3.0)]),
            _event("Denver Nuggets", "Phoenix Suns", [_winner_market(1.7, 2.2)]),
        ]
    )
    result = extract_draftkings_odds(payload, "2024-05-05")
    assert set(result.keys()) == {
        "2024-05-05_MIA@BOS",
        "2024-05-05_PHX@DEN",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
