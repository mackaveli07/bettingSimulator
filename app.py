import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor

# --- Cache and scrape FiveThirtyEight Elo ---
@st.cache_data(show_spinner=False)
def scrape_fivethirtyeight_elo(league):
    urls = {
        "nba": "https://projects.fivethirtyeight.com/2024-nba-predictions/",
        "nfl": "https://projects.fivethirtyeight.com/nfl-predictions-2024/",
        "mlb": "https://projects.fivethirtyeight.com/mlb-predictions-2024/",
        "nhl": "https://projects.fivethirtyeight.com/nhl-predictions-2024/",
        "wnba": "https://projects.fivethirtyeight.com/wnba-predictions-2024/"
    }
    url = urls.get(league.lower())
    if not url:
        st.error(f"No URL configured for league '{league}'")
        return pd.DataFrame()

    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")

    for idx, table in enumerate(tables):
        try:
            dfs = pd.read_html(str(table))
            for df in dfs:
                # Ensure the DataFrame isn't empty and has the necessary columns
                if "Team" in df.columns and any("Elo" in col for col in df.columns):
                    elo_col = next(col for col in df.columns if "Elo" in col)
                    return df[["Team", elo_col]].rename(columns={"Team": "team", elo_col: "elo"})
        except Exception as e:
            # Optionally log the error
            print(f"Table {idx} skipped due to error: {e}")
            continue

    st.error(f"Elo table not found for league '{league}'")
    return pd.DataFrame()

# --- Name mapping ESPN team -> FiveThirtyEight team ---
# You should customize this for your league and data sources
TEAM_NAME_MAP = {
    # NBA example
    "Boston Celtics": "Boston Celtics",
    "Miami Heat": "Miami Heat",
    "Golden State Warriors": "Golden State Warriors",
    "LA Lakers": "Los Angeles Lakers",  # Example where ESPN uses "LA Lakers"
    "Los Angeles Lakers": "Los Angeles Lakers",
    # Add mappings for all teams for your league here
}

def map_team_name(espn_name):
    return TEAM_NAME_MAP.get(espn_name, espn_name)

# --- Realistic fetch_live_matchups_from_espn placeholder ---
@st.cache_data(show_spinner=False)
def fetch_live_matchups_from_espn(league):
    # TODO: Replace with your real ESPN scraping or API fetching code
    # Here is example static data, team names use ESPN style
    if league == "nba":
        return [
            {"team": "Boston Celtics", "opponent": "Miami Heat",
             "team_logo": "https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg",
             "opponent_logo": "https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.svg"},
            {"team": "Golden State Warriors", "opponent": "LA Lakers",
             "team_logo": "https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg",
             "opponent_logo": "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg"},
        ]
    else:
        return []

# --- Train simple RandomForestRegressor models on ELO data ---
@st.cache_data(show_spinner=False)
def train_score_models(elo_df):
    # Create synthetic historical data for demonstration:
    # Elo rating for team and opponent, target is score (mocked as elo-based)
    n_samples = 200
    np.random.seed(42)

    df_train = pd.DataFrame({
        "team_elo": np.random.uniform(1300, 1800, n_samples),
        "opp_elo": np.random.uniform(1300, 1800, n_samples),
    })
    # Synthetic scores roughly proportional to Elo difference + noise
    df_train["team_score"] = 100 + 0.1 * (df_train.team_elo - df_train.opp_elo) + np.random.normal(0, 10, n_samples)
    df_train["opp_score"] = 100 + 0.1 * (df_train.opp_elo - df_train.team_elo) + np.random.normal(0, 10, n_samples)

    model_team = RandomForestRegressor(n_estimators=50, random_state=42)
    model_opp = RandomForestRegressor(n_estimators=50, random_state=42)

    model_team.fit(df_train[["team_elo", "opp_elo"]], df_train["team_score"])
    model_opp.fit(df_train[["opp_elo", "team_elo"]], df_train["opp_score"])

    return model_team, model_opp

# --- Convert win probability to moneyline odds ---
def convert_winprob_to_moneyline(win_prob):
    if win_prob == 0:
        return None
    elif win_prob >= 0.5:
        odds = -100 * win_prob / (1 - win_prob)
    else:
        odds = 100 * (1 - win_prob) / win_prob
    return int(round(odds))

# --- Simulate games ---
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

# --- Main app ---
st.set_page_config(page_title="Betting Odds Simulator", layout="wide")
st.title("🕹️ Betting Odds Simulator")

league = st.selectbox("Select League", ["nba", "nfl", "mlb", "nhl", "wnba"])

matchups = fetch_live_matchups_from_espn(league)
if not matchups:
    st.warning("No matchups found for this league.")
    st.stop()

elo_df = scrape_fivethirtyeight_elo(league)
if elo_df.empty:
    st.warning("Could not fetch Elo data for this league.")
    st.stop()

# Map ESPN team names to FiveThirtyEight names for merging
for m in matchups:
    m["team"] = map_team_name(m["team"])
    m["opponent"] = map_team_name(m["opponent"])

df = pd.DataFrame(matchups)
# Merge Elo ratings on team and opponent
df = df.merge(elo_df, left_on="team", right_on="team", how="left").rename(columns={"elo": "team_elo"})
df = df.merge(elo_df, left_on="opponent", right_on="team", how="left", suffixes=("", "_opp")).rename(columns={"elo": "opp_elo"})
df = df.drop(columns=["team_opp"])

df = df.dropna(subset=["team_elo", "opp_elo"])

model_team, model_opp = train_score_models(elo_df)

results = []
for _, row in df.iterrows():
    res = simulate_game(row["team_elo"], row["opp_elo"], model_team, model_opp)
    res.update({
        "team": row["team"],
        "opponent": row["opponent"],
        "team_logo": row["team_logo"],
        "opponent_logo": row["opponent_logo"]
    })
    results.append(res)

results_df = pd.DataFrame(results)

for _, row in results_df.iterrows():
    cols = st.columns([1, 2, 1])
    with cols[0]:
        st.image(row["team_logo"], width=80)
        st.markdown(f"**{row['team']}**")
        st.markdown(f"Avg Score: {row['team_avg_score']:.1f}")
    with cols[1]:
        st.markdown("### VS")
        st.markdown(f"Spread: {row['spread_mean']:.1f}")
        st.markdown(f"Moneyline: {row['moneyline_odds']}")
        st.markdown(f"Total Points: {row['total_points_mean']:.1f}")
    with cols[2]:
        st.image(row["opponent_logo"], width=80)
        st.markdown(f"**{row['opponent']}**")
        st.markdown(f"Avg Score: {row['opp_avg_score']:.1f}")

    # Win probability chart
    fig, ax = plt.subplots(figsize=(6, 2))
    labels = [row['team'], row['opponent']]
    probs = [row['win_prob'], 1 - row['win_prob']]
    ax.barh(labels, probs, color=["#4caf50", "#f44336"])
    ax.set_xlim(0, 1)
    ax.set_title("Win Probability")
    st.pyplot(fig)
    plt.close(fig)
