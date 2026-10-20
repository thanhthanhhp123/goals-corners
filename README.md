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
- when bookmaker odds are out (league games, a few days before kick-off): the bookmakers'
  probabilities next to ours, average and best prices, and the expected return at each price
- a price checker: type any bookmaker's odds for a goals, corners or both-teams-to-score line
  and see our probability and the expected return

Other views:

- **Lên thuyền** ("get on the boat") suggests picks: the model's strongest over/under calls
  for the coming days, with our chance, the competition's usual rate, break-even odds and the
  best bookmaker price where we have one. Picks can be added to a boat (a combo) to see the
  chance that they all win. Below the list, the same rule's record over past matches.
- **Results** shows every finished match this season next to the prediction made *before*
  kick-off, with running hit rates.
- **Match tool** predicts any pairing of clubs, including clubs from different leagues.
- **Teams** lists attack, defence and corner ratings for every club.
- **Model** reports out-of-sample accuracy against a baseline and against the bookmakers,
  calibration charts and the method.

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

**Odds** (`predictor/market.py`). [football-data.co.uk](https://www.football-data.co.uk)
publishes free CSVs for 16 of these leagues (not the UEFA cups) with average, best and
single-bookmaker prices for 1X2 and over/under 2.5, in two snapshots: pre-closing (Tuesday
and Friday before a round) and closing. Rows are matched to ESPN events by date, score and
club names (about 20,000 matches, 99–100% coverage). From each match's 1X2 and over/under
prices, with the margin removed, we solve for the two expected-goal numbers whose
Dixon–Coles score grid reproduces them. The same files carry corner counts, which fill
ESPN's gaps (3,000+ extra matches, mostly 2023–24).
Files are cached in `data/market/`.

**Model** (`predictor/model.py`). Two penalised Poisson regressions, one for goals and one
for corners:

```
log E[count for a side] = mu + competition + home advantage + attack(team) − defence(opponent)
```

- Goals: wherever a past match had odds, the regression is fitted to the expected goals
  implied by its closing prices instead of the scoreline, and such a match counts 30 times
  a scoreline. A price measures team strength with far less luck than a result does, so the
  ratings can follow form with a 60-day half-life. UEFA cup games and leagues without odds
  are fitted to their scores.
- Matches are weighted by recency (exponential half-life).
- Each club's ratings are shrunk toward its league's average, which keeps estimates stable
  for clubs with little data.
- Goals: the two expected values give a scoreline grid with a Dixon–Coles low-score
  correction. The total-goals distribution, 1X2, BTTS and over/unders all come from that grid.
- When a fixture has pre-closing odds, its expected goals are a geometric blend: 60% the
  bookmakers', 40% the model's (`MARKET_BLEND`).
- Corners: a negative-binomial distribution with a fitted dispersion, because corner totals
  vary more than a Poisson count does. Nudging corners by the odds and adding shot-volume
  ratings were both tested; neither improved the held-out season reliably, so neither is used.
- Hyperparameters live in `predictor/config.py` and were chosen by `tune` on the 2025-26
  season only.

**Evaluation** (`predictor/backtest.py`). Rolling origin: each week the model is refitted
using only matches played before that week, then scored on that week's games. The baseline
is the competition's empirical distribution of totals over the previous year. Blending uses
only the pre-closing odds, which were public a day or more before kick-off. 2026-27 is
held out from tuning. Results are cached in `data/backtest_*.pkl`, so later builds only
compute new weeks.

**Picks** (`predictor/tips.py`). For each match and market (total goals at 1.5/2.5/3.5,
total corners at 8.5–11.5) the pick is the side whose chance is furthest above the
competition's own rate at that line over the past year, if the chance is between 60% and 85%
and at least 8 points above that rate. Taking the most likely side instead would keep
listing calls such as "under 11.5 corners" that hold in most matches. The thresholds were set
for a manageable number of picks (about 20–25 a week), not tuned for results. The same rule
runs over the backtest to give the record on the page.

## Layout

```
predictor/            Python package
  config.py           competitions, data window, hyperparameters
  espn.py             download + cache + normalise ESPN scoreboards
  model.py            rating models and match predictions
  dist.py             Poisson / negative-binomial / Dixon–Coles helpers
  backtest.py         rolling evaluation, metrics, tuning
  tips.py             pick suggestions (Lên thuyền) and their track record
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

| | 2025-26 (tuning season) | 2026-27 (held out, to 19 Oct) |
|---|---|---|
| Matches | 1,937 | 377 |
| Goals O/U 2.5 calls correct | 58.7% (baseline 55.6%) | 63.9% (baseline 58.4%) |
| Avg error, total goals | 1.31 (baseline 1.33) | 1.43 (baseline 1.51) |
| Corners O/U 9.5 calls correct | 56.8% (baseline 53.2%) | 56.2% (baseline 51.7%) |
| Avg error, total corners | 2.67 (baseline 2.70) | 2.81 (baseline 2.86) |

The baseline is each competition's own distribution of totals over the previous 365 days.

**Against the bookmakers.** Over/under 2.5 goals in big-five league matches that had odds
(log loss, lower is better):

| | 2025-26 (1,751 matches) | 2026-27 (341 matches) |
|---|---|---|
| Earlier score-only model | 0.681 | 0.658 (297 matches to 18 Oct) |
| Model alone | 0.6761 | 0.6491 |
| Model + odds (what the site shows) | **0.6758** | **0.6455** |
| Bookmakers, margin removed | 0.6760 | 0.6432 |

On the tuning season the blend matches the bookmakers; on the held-out season it trails them
by 0.002 (the score-only model trailed by 0.017). Matching the bookmakers is not the same as
beating them: their margin averaged 6.8%, and backing every over/under 2.5 "edge" above 3% at
the best price across bookmakers returned −5.5% over 113 bets in 2025-26 and −36% over 67 bets
so far in 2026-27. The site shows edges, but treat them as hints. Corner odds are not in the
free data, so the corner forecasts are only tested against results.

**Picks.** The Lên thuyền picks win about as often as they say. Of 1,090 picks in 2025-26
(stated 70% on average) 68% won; of 191 in 2026-27 to 20 Oct (stated 70%) 72% won. That makes them
likely, not profitable: the 233 over/under 2.5 picks of 2025-26 that had odds returned −10%
at the average price (±4%) and −8% at the best price, because the bookmakers usually rate
the same side as likely and price it short. 2026-27 to 20 Oct: +1.5% over 46 picks (±9%), which
is within noise.

## Limits

Predictions cover regulation time (90 minutes plus stoppage time). Line-ups, injuries,
suspensions and motivation reach the forecast only through bookmaker odds, so only for league
games whose odds are out. The Champions League has no odds in the free data. Corner totals are
noisy by nature. Treat the numbers as an informed starting point, not a certainty.
