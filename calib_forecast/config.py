"""Paths + spec for the calibrated-forecasting demo."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "box_scores.parquet"
ASSETS = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# Stats to forecast (column -> display name).
STATS = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes"}
# Pre-game features the mean model uses (trailing rollups, derived leakage-safe).
ROLL_COLS = ["min", "pts", "reb", "ast", "fg3m", "fga", "fta", "fg3a"]
TEST_SEASON = "2025-26"   # train on everything before; evaluate on this
