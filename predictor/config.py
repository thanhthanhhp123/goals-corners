"""Static configuration: competitions, data window and model hyperparameters."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WEB_DIR = ROOT / "web"
WEB_DATA_DIR = WEB_DIR / "data"

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"

# Competitions we publish predictions for.
TARGET_COMPETITIONS: dict[str, dict] = {
    "eng.1": {"name": "Premier League", "short": "EPL", "country": "England"},
    "esp.1": {"name": "LaLiga", "short": "LaLiga", "country": "Spain"},
    "ita.1": {"name": "Serie A", "short": "Serie A", "country": "Italy"},
    "ger.1": {"name": "Bundesliga", "short": "Bundesliga", "country": "Germany"},
    "fra.1": {"name": "Ligue 1", "short": "Ligue 1", "country": "France"},
    "uefa.champions": {"name": "Champions League", "short": "UCL", "country": "Europe"},
}

# Competitions used only to train the model. Second divisions rate newly promoted
# clubs; feeder leagues and the other UEFA competitions put clubs from smaller
# leagues on the same scale as the big five (needed for Champions League games).
SUPPORT_COMPETITIONS: dict[str, dict] = {
    "eng.2": {"name": "Championship", "country": "England"},
    "esp.2": {"name": "LaLiga 2", "country": "Spain"},
    "ita.2": {"name": "Serie B", "country": "Italy"},
    "ger.2": {"name": "2. Bundesliga", "country": "Germany"},
    "fra.2": {"name": "Ligue 2", "country": "France"},
    "por.1": {"name": "Liga Portugal", "country": "Portugal"},
    "ned.1": {"name": "Eredivisie", "country": "Netherlands"},
    "bel.1": {"name": "Belgian Pro League", "country": "Belgium"},
    "sco.1": {"name": "Scottish Premiership", "country": "Scotland"},
    "tur.1": {"name": "Super Lig", "country": "Turkey"},
    "aut.1": {"name": "Austrian Bundesliga", "country": "Austria"},
    "gre.1": {"name": "Greek Super League", "country": "Greece"},
    "den.1": {"name": "Danish Superliga", "country": "Denmark"},
    "nor.1": {"name": "Eliteserien", "country": "Norway"},
    "swe.1": {"name": "Allsvenskan", "country": "Sweden"},
    "cyp.1": {"name": "Cypriot First Division", "country": "Cyprus"},
    "uefa.europa": {"name": "Europa League", "country": "Europe"},
    "uefa.europa.conf": {"name": "Conference League", "country": "Europe"},
    "uefa.champions_qual": {"name": "UCL Qualifying", "country": "Europe"},
    "uefa.europa_qual": {"name": "UEL Qualifying", "country": "Europe"},
    "uefa.europa.conf_qual": {"name": "UECL Qualifying", "country": "Europe"},
}

ALL_COMPETITIONS = {**TARGET_COMPETITIONS, **SUPPORT_COMPETITIONS}


def is_domestic(slug: str) -> bool:
    return not slug.startswith("uefa.")


FIRST_YEAR = 2023  # calendar years of history to download (ESPN queries by calendar year)

# Statuses that count as a finished match. Extra-time games are kept out of
# training because their totals include 120 minutes of play.
FINISHED_STATUSES = {"STATUS_FULL_TIME", "STATUS_FINAL"}
EXTRA_TIME_STATUSES = {"STATUS_FINAL_AET", "STATUS_FINAL_PEN"}

# Model hyperparameters, chosen by `python -m predictor tune` on the 2025-26 season
# (fortnightly refits, scored by mean log score of the match total). The optimum is
# flat: goals lam_team 20-30 and corners 64-100 are within 0.0005 of each other.
GOALS_PARAMS = {
    "half_life_days": 270.0,
    "lam_team": 25.0,
    "lam_group": 2.0,
    "lam_comp": 50.0,
    "lam_home": 20.0,
}
CORNERS_PARAMS = {
    "half_life_days": 270.0,
    "lam_team": 80.0,
    "lam_group": 2.0,
    "lam_comp": 50.0,
    "lam_home": 20.0,
}

GOAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
CORNER_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
TEAM_CORNER_LINES = [2.5, 3.5, 4.5, 5.5, 6.5]
MAX_GOALS = 10
MAX_CORNERS = 30
