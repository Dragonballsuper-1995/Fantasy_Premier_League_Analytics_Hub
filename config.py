# config.py

# --- Data URLs ---
# We point directly to the 'raw' files in Vaastav's GitHub repository.

# Training data (2024-25 season)
TRAIN_DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"

# Current season data (2025-26)
# NOTE: Vaastav's repo names this 'merged_gw.csv' inside the folder.
HISTORY_DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv"
FIXTURES_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/fixtures.csv"
TEAMS_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/teams.csv"
PLAYER_RAW_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/players_raw.csv"

# --- Output ---
# This is the local file our main.py will create, which the GitHub Action will then commit.
OUTPUT_JSON_FILE = "predictions.json"