# Goals & Corners Predictor

A web app that predicts **total goals** and **total corners** for every match in the
Premier League, LaLiga, Serie A, Bundesliga, Ligue 1 and the UEFA Champions League.

For each fixture you get:

- expected goals and corners for each team and for the match
- the full probability distribution of total goals and total corners, with over/under
  probabilities and fair odds for any line (goals 0.5–5.5, corners 6.5–13.5)
- 1X2, both-teams-to-score, clean sheets and the most likely scorelines
- team corner lines and the chance each side wins more corners
- recent form (goals and corners, last 6 matches) for both clubs

Other views:

- **Results** shows every finished match this season next to the prediction made *before*
  kick-off, with running hit rates.
- **Match tool** predicts any pairing of clubs, including clubs from different leagues.
- **Teams** lists attack, defence and corner ratings for every club.
- **Model** reports out-of-sample accuracy against a baseline, calibration charts and the
  method.

## Quick start

```bash
pip install -r requirements.txt
python -m predictor update     # download data, fit, backtest, write web/data/*.json (~3 min the first time)
python -m predictor serve      # http://127.0.0.1:8000
```

On Windows you can double-click `start.bat` instead; it builds the data on the first run
and opens the browser.

`serve` refreshes the data automatically when it is more than 6 hours old
(`--refresh-hours 0` turns that off), and the page has a **Refresh data** button.
A refresh takes about 40 seconds because finished backtest weeks are cached.
If you edit `predictor/config.py`, restart `serve` so it picks up the change.

The `web/` folder is a static site, so once `web/data/` is built any static host can serve it.
Without the Python server, the refresh button is hidden.
`.github/workflows/pages.yml` does this on GitHub for free: it rebuilds the data every 6 hours
and publishes `web/` to GitHub Pages (set Settings → Pages → Source to "GitHub Actions").
`python -m predictor export` writes `dist/goals-corners-predictor.html`, a single file with the
data and team logos inlined. You can open it anywhere or upload it to any host.
`--artifact` writes the same page without `<html>`/`<head>`, for hosts that add their own.

Other commands:

```bash
python -m predictor update --offline          # rebuild from cached data, no download
python -m predictor backtest --start 2026-07-01
python -m predictor tune --kind corners       # grid-search hyperparameters on 2025-26
python -m pytest -q                           # tests (includes a Python/JS parity check)
```

## How it works

**Data.** ESPN's public scoreboard API returns fixtures, results and box-score stats
(including corners won) for each competition. The app downloads 27 competitions from 2023
onward. The six target competitions are covered, and so are:

- the five second divisions, so promoted clubs have a rating;
- 11 feeder leagues (Portugal, Netherlands, Belgium, Scotland, Turkey, Austria, Greece,
  Denmark, Norway, Sweden, Cyprus);
- the other UEFA competitions and their qualifiers.

The extra leagues and cups link every Champions League club to one shared rating scale.
Raw responses are cached in `data/raw/`; past years are downloaded once.
ESPN reports a missing box score as 0–0 corners, so those games are treated as missing.

**Model** (`predictor/model.py`). Two penalised Poisson regressions, one for goals and one
for corners:

```
log E[count for a side] = mu + competition + home advantage + attack(team) − defence(opponent)
```

- Matches are weighted by recency (exponential half-life).
- Each club's ratings are shrunk toward its league's average, which keeps estimates stable
  for clubs with little data.
- Goals: the two expected values give a scoreline grid with a Dixon–Coles low-score
  correction. The total-goals distribution, 1X2, BTTS and over/unders all come from that grid.
- Corners: a negative-binomial distribution with a fitted dispersion, because corner totals
  vary more than a Poisson count does.
- Hyperparameters live in `predictor/config.py` and were chosen by `tune` on the 2025-26
  season only.

**Evaluation** (`predictor/backtest.py`). Rolling origin: each week the model is refitted
using only matches played before that week, then scored on that week's games. The baseline
is the competition's empirical distribution of totals over the previous year. 2026-27 is
held out from tuning. Results are cached in `data/backtest_*.pkl`, so later builds only
compute new weeks.

## Layout

```
predictor/            Python package
  config.py           competitions, data window, hyperparameters
  espn.py             download + cache + normalise ESPN scoreboards
  model.py            rating models and match predictions
  dist.py             Poisson / negative-binomial / Dixon–Coles helpers
  backtest.py         rolling evaluation, metrics, tuning
  build.py            end-to-end build → web/data/*.json
  server.py           static server + /api/refresh, /api/status
  export.py           single-file HTML export
web/                  static front end (no build step)
  index.html, styles.css, app.js, charts.js
  model.js            JS port of the prediction maths for the match tool
tests/
```

## Accuracy (out of sample)

Every number below comes from a weekly backtest. 2026-27 was never used to choose settings.

| | 2025-26 (tuning season) | 2026-27 (held out, to 18 Oct) |
|---|---|---|
| Matches | 1,937 | 373 |
| Goals O/U 2.5 calls correct | 57.9% (baseline 55.6%) | 62.7% (baseline 58.2%) |
| Avg error, total goals | 1.32 (baseline 1.33) | 1.46 (baseline 1.51) |
| Corners O/U 9.5 calls correct | 56.5% (baseline 53.2%) | 57.1% (baseline 52.0%) |
| Avg error, total corners | 2.67 (baseline 2.70) | 2.81 (baseline 2.86) |

The baseline is each competition's own distribution of totals over the previous 365 days.
Match totals are noisy, so expect an edge of a few percentage points, not a large one.
Scoring in 2026-27 is running well above earlier seasons (3.18 goals per game against 2.87
predicted). The model's league levels are catching up week by week.

## Limits

Predictions cover regulation time (90 minutes plus stoppage time). The model does not know
about line-ups, injuries, suspensions, weather, referee tendencies or motivation, and corner
totals are noisy by nature. Treat the numbers as an informed starting point, not a certainty.
