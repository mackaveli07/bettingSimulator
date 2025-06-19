import requests

def fetch_live_matchups_from_espn(league):
    endpoint = {
        'nba': 'basketball/nba',
        'nfl': 'football/nfl',
        'mlb': 'baseball/mlb',
        'nhl': 'hockey/nhl',
        'wnba': 'basketball/wnba'
    }.get(league)
    if not endpoint:
        return []
    url = f"https://site.api.espn.com/apis/site/v2/sports/{endpoint}/scoreboard"
    data = requests.get(url).json()
    games = data.get('events', [])
    matchups = []
    for g in games:
        competitors = g['competitions'][0]['competitors']
        home = next(c for c in competitors if c['homeAway'] == 'home')
        away = next(c for c in competitors if c['homeAway'] == 'away')
        matchups.append({
            'team': home['team']['abbreviation'],
            'opponent': away['team']['abbreviation'],
            'team_elo': 1500,
            'opp_elo': 1500,
            'team_logo': home['team']['logo'],
            'opponent_logo': away['team']['logo']
        })
    return matchups