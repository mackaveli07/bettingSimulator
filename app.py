import streamlit as st
import pandas as pd
from data_scrapers import (
    scrape_nba, scrape_nfl, scrape_mlb, scrape_nhl, scrape_wnba,
    scrape_league_multi_season
)
from elo_calculator import calculate_elo_ratings  # Your Elo functions
from simulator import simulate_game, train_team_score_model, convert_winprob_to_moneyline
import numpy as np

LEAGUE_SCRAPERS = {
    "nba": scrape_nba,
    "nfl": scrape_nfl,
    "mlb": scrape_mlb,
    "nhl": scrape_nhl,
    "wnba": scrape_wnba
}

st.set_page_config(page_title="Betting Odds Simulator", layout="wide")
st.title("🕹️ Betting Odds Simulator")

league = st.selectbox("Select League", list(LEAGUE_SCRAPERS.keys()))
start_year = st.number_input("Start Season Year", min_value=2000, max_value=2025, value=2021)
end_year = st.number_input("End Season Year", min_value=2000, max_value=2025, value=2023)

@st.cache_data(ttl=60*60*24*2)
def load_games(league, start_year, end_year):
    scraper = LEAGUE_SCRAPERS[league]
    df = scrape_league_multi_season(scraper, start_year, end_year)
    return df

games_df = load_games(league, start_year, end_year)

if games_df.empty:
    st.warning("No games loaded!")
else:
    st.write(f"Loaded {len(games_df)} {league.upper()} games")

    # Calculate Elo ratings based on historical games
    # You need to implement calculate_elo_ratings to process your df format
    elo_df = calculate_elo_ratings(games_df)

    st.dataframe(elo_df.head())

    # Now simulate upcoming or hypothetical matchups
    # For example, pick some matchups (or live upcoming games you fetch)
    # Here, just simulate the last 5 games as example
    last_games = games_df.tail(5)

    # Dummy models — train your ML models based on Elo or other stats as you want
    dummy_df = pd.DataFrame({
        'elo1_pre': np.random.randint(1400, 1700, 100),
        'elo2_pre': np.random.randint(1400, 1700, 100),
        'score1': np.random.randint(80, 130, 100),
        'score2': np.random.randint(80, 130, 100)
    })
    model_team = train_team_score_model(dummy_df, 'score1', 'elo1_pre', 'elo2_pre')
    model_opp = train_team_score_model(dummy_df, 'score2', 'elo2_pre', 'elo1_pre')

    results = []
    for _, row in last_games.iterrows():
        team_elo = elo_df.loc[elo_df['team'] == row['home_team'], 'elo'].values
        opp_elo = elo_df.loc[elo_df['team'] == row['visitor_team'], 'elo'].values
        if len(team_elo) == 0 or len(opp_elo) == 0:
            continue
        res = simulate_game(team_elo[0], opp_elo[0], model_team, model_opp)
        res.update({
            "team": row['home_team'],
            "opponent": row['visitor_team']
        })
        results.append(res)

    if results:
        st.subheader("Simulation Results")
        for r in results:
            st.write(r)
    else:
        st.info("No simulation results available.")
