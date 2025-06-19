import streamlit as st
import pandas as pd
from simulator import simulate_game, train_team_score_model
from data_updater import fetch_live_matchups_from_espn

st.set_page_config(page_title="Betting Odds Simulator", layout="wide")
st.title("🕹️ Betting Odds Simulator")

league = st.selectbox("Select League", ["nba", "nfl", "mlb", "nhl", "wnba"])

if st.button("Fetch & Simulate Matchups"):
    matchups = fetch_live_matchups_from_espn(league)
    df = pd.DataFrame(matchups)
    st.write("Fetched Matchups:")

    # Dummy training data
    import numpy as np
    dummy_df = pd.DataFrame({
        'elo1_pre': np.random.randint(1400, 1700, size=100),
        'elo2_pre': np.random.randint(1400, 1700, size=100),
        'score1': np.random.randint(80, 130, size=100),
        'score2': np.random.randint(80, 130, size=100)
    })
    model_team = train_team_score_model(dummy_df, 'score1', 'elo1_pre', 'elo2_pre')
    model_opp = train_team_score_model(dummy_df, 'score2', 'elo2_pre', 'elo1_pre')

    results = []
    for _, row in df.iterrows():
        res = simulate_game(row['team_elo'], row['opp_elo'], model_team, model_opp)
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

    st.subheader("📊 Odds Visualization")
    for _, row in results_df.iterrows():
        fig, ax = plt.subplots(figsize=(6, 2))
        labels = [row['team'], row['opponent']]
        probs = [row['win_prob'], 1 - row['win_prob']]
        ax.barh(labels, probs, color=["#4caf50", "#f44336"])
        ax.set_xlim(0, 1)
        ax.set_title("Win Probability")
        st.pyplot(fig)