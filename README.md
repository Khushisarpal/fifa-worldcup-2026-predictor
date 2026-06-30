# fifa-worldcup-2026-predictor
Ensemble model (Elo + Poisson + XGBoost) that simulates the FIFA World Cup 2026 ten thousand times to predict each team's odds of reaching the semis, final, and winning it all. Outputs feed a Power BI dashboard.
# FIFA World Cup 2026 Predictor
 
I got curious whether I could actually predict the World Cup with real models instead of just vibes, so I built this. It pulls historical international football results going back decades, builds a rating system for every national team, and runs the whole 2026 tournament thousands of times in a simulation to see who wins most often.
 
Short version: it's not just "who has the best Elo rating." It's three different models arguing with each other and then averaging out their opinion.
 
## What it actually does
 
1. **Loads match data** — historical results from Kaggle plus live 2026 results as they come in, so predictions update as the tournament actually unfolds.
2. **Builds Elo ratings** — the classic chess-style rating system, adapted for football, where recent results count more than old ones.
3. **Fits a Poisson model** — predicts how many goals each team is likely to score based on their attack and the opponent's defence. This is what lets the sim generate actual scorelines instead of just win/loss.
4. **Trains an XGBoost classifier** — a machine learning model that picks up on patterns the other two miss (form, squad strength, home advantage interactions).
5. **Blends all three** — weights are tuned against real World Cup results, not just guessed.
6. **Runs 10,000 Monte Carlo simulations** of the entire tournament, groups through final, to get a probability for every team reaching the semis, final, and winning it all.
7. **Outputs everything to CSV** so it can be dropped straight into Power BI.
## Why three models instead of one
 
Elo alone only tells you who's more likely to win — it doesn't know what scoreline that win looks like, which matters a lot for group stage tiebreakers (goal difference, goals scored). Poisson fixes that by giving full scoreline probabilities. XGBoost adds in the non-linear stuff, like how recent form and squad strength interact in ways a simple rating can't capture. None of them are "the best" on their own — together they're more reliable than any single one.
 
## Running it
 
```bash
pip install pandas numpy scipy scikit-learn xgboost openpyxl
python wc2026_predictor.py
```
 
You'll need:
- `data/results.csv` — historical match results (Kaggle international football dataset works)
- `data/live data.xlsx` — actual 2026 World Cup results as they're played
- `data/squad_strength.csv` *(optional)* — squad value/caps data, improves predictions if you have it
Outputs land in `/output`:
- `wc2026_predictions.csv` — every team's probability of reaching semis/final/winning
- `group_stage_predictions.csv` — match-by-match predictions for the group stage
- `elo_ratings.csv` — full Elo leaderboard
Set `N_SIM=20000` as an environment variable if you want more simulations (slower, slightly more stable numbers). Set `FORCE_REFIT=1` if you've changed the data and want the models to retrain from scratch instead of using cached versions.
 
## What's next
 
- Better bracket logic for the third-place qualifier slots (right now it's simplified)
- Player-level injury/suspension data feeding into match-day predictions
- Auto-refresh the live results so the dashboard updates itself during the tournament
Built as a personal project to learn ensemble modelling and Monte Carlo simulation properly, and partly because I wanted to know if Brazil really has the numbers behind them or if it's just hype.
