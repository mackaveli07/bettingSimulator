import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_team_score_model(df, team_col_score, elo_col_pre, opponent_elo_col):
    X = df[[elo_col_pre, opponent_elo_col]]
    y = df[team_col_score]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def convert_winprob_to_moneyline(p):
    if p == 0: return float('-inf')
    if p == 1: return float('inf')
    return round(-100 * p / (1 - p)) if p > 0.5 else round(100 * (1 - p) / p)

def simulate_game(team_elo, opp_elo, model_team, model_opp, n_simulations=1000):
    team_scores = []
    opp_scores = []
    for _ in range(n_simulations):
        team_score = model_team.predict([[team_elo, opp_elo]])[0]
        opp_score = model_opp.predict([[opp_elo, team_elo]])[0]
        team_scores.append(team_score)
        opp_scores.append(opp_score)
    team_scores = np.array(team_scores)
    opp_scores = np.array(opp_scores)
    spread = team_scores - opp_scores
    total_points = team_scores + opp_scores
    win_prob = np.mean(team_scores > opp_scores)
    return {
        "team_avg_score": float(np.mean(team_scores)),
        "opp_avg_score": float(np.mean(opp_scores)),
        "spread_mean": float(np.mean(spread)),
        "total_points_mean": float(np.mean(total_points)),
        "win_prob": float(win_prob),
        "moneyline_odds": convert_winprob_to_moneyline(win_prob)
    }