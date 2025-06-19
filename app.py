import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from simulator import simulate_game, train_team_score_model
from data_updater import fetch_live_matchups_from_espn

st.set_page_config(page_title="Betting Odds Simulator", layout="wide")
st.title("🕹️ Betting Odds Simulator")

# League selector
league = st.selectbox("Select League", ["nba", "nfl", "mlb", "nhl", "wnba"])

# Load real historical Elo data from FiveThirtyEight
@st.cache_data(show_spinner=False)
def load_elo_data(league):
    urls = {
        "nba": "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv",
        "nfl": "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv",
        "mlb": "https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv",
        "nhl": "https://projects.fivethirtyeight.com/nhl-api/nhl_elo.csv",
        "wnba": "https://projects.fivethirtyeight.com/wnba-model/wnba_elo.csv"
    }
    url = urls.get(league)
    if not url:
        st.error(f"No Elo data URL configured for league {league}")
        return pd.DataFrame()

    try:
        # Read CSV with python engine to handle irregular lines
        df = pd.read_csv(url, engine="python", on_bad_lines='skip')
    except Exception as e:
        st.error(f"Failed to load Elo data: {e}")
        return pd.DataFrame()

    return df

# Auto-fetch matchups using cache
@st.cache_data(show_spinner=False)
def get_matchups(league):
    return fetch_live_matchups_from_espn(league)

matchups = get_matchups(league)
elo_df = load_elo_data(league)

if not matchups:
    st.warning("No matchups found for this league.")
else:
    df = pd.DataFrame(matchups)
    st.subheader("Fetched Matchups")
    st.dataframe(df)

    # Extract latest Elo per team
    latest_elo = elo_df.sort_values("date").groupby("team1")["elo1_pre"].last().reset_index()
    latest_elo.columns = ["team", "elo"]

    # Assign Elo ratings to matchups
    df = df.merge(latest_elo, left_on="team", right_on="team", how="left")
    df = df.rename(columns={"elo": "team_elo"})
    df = df.merge(latest_elo, left_on="opponent", right_on="team", how="left", suffixes=("", "_opp"))
    df = df.rename(columns={"elo": "opp_elo"})

    # Drop extra team columns
    df = df.drop(columns=["team_opp"])

    # Train models using real historical data
    model_team = train_team_score_model(elo_df, 'score1', 'elo1_pre', 'elo2_pre')
    model_opp = train_team_score_model(elo_df, 'score2', 'elo2_pre', 'elo1_pre')

    results = []
    for _, row in df.iterrows():
        if pd.isna(row['team_elo']) or pd.isna(row['opp_elo']):
            continue
        res = simulate_game(row['team_elo'], row['opp_elo'], model_team, model_opp)
        res.update({
            "team": row["team"],
            "opponent": row["opponent"],
            "team_logo": row["team_logo"],
            "opponent_logo": row["opponent_logo"]
        })
        results.append(res)

    results_df = pd.DataFrame(results)

    st.subheader("Simulation Results")
    st.dataframe(results_df)

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

        # Chart: Win Probability
        fig, ax = plt.subplots(figsize=(6, 2))
        labels = [row['team'], row['opponent']]
        probs = [row['win_prob'], 1 - row['win_prob']]
        ax.barh(labels, probs, color=["#4caf50", "#f44336"])
        ax.set_xlim(0, 1)
        ax.set_title("Win Probability")
        st.pyplot(fig)
        plt.close(fig)

        # Mock live odds comparison
        live_spread = row['spread_mean'] + random.uniform(-5, 5)
        live_total = row['total_points_mean'] + random.uniform(-10, 10)
        spread_color = "green" if abs(row['spread_mean']) < abs(live_spread) else "red"
        total_color = "green" if row['total_points_mean'] > live_total else "red"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Model Spread:** `{row['spread_mean']:.1f}`")
            st.markdown(f"**Live Spread:** :{spread_color}[{live_spread:.1f}]")
        with col2:
            st.markdown(f"**Model Total:** `{row['total_points_mean']:.1f}`")
            st.markdown(f"**Live Total:** :{total_color}[{live_total:.1f}]")
        with col3:
            st.markdown(f"**Moneyline Odds:** `{row['moneyline_odds']}`")
