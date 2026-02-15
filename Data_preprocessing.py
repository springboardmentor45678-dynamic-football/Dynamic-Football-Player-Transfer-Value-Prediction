import pandas as pd
import os

path = "/content/transfermarkt_datasets"  # change based on your PC path

files = os.listdir(path)
files
appearances = pd.read_excel(f"{path}/appearances.xlsx")
club_games = pd.read_excel(f"{path}/club_games.xlsx")
clubs = pd.read_excel(f"{path}/clubs.xlsx")
competitions = pd.read_excel(f"{path}/competitions.xlsx")
game_events = pd.read_excel(f"{path}/game_events.xlsx")
game_lineups = pd.read_excel(f"{path}/game_lineups.xlsx")
games = pd.read_excel(f"{path}/games.xlsx")
player_valuations = pd.read_excel(f"{path}/player_valuations.xlsx")
players = pd.read_excel(f"{path}/players.xlsx")
transfers = pd.read_excel(f"{path}/transfers.xlsx")
for name, df in [
    ("appearances", appearances),
    ("players", players),
    ("player_valuations", player_valuations),
    ("transfers", transfers)
]:
    print("\n====", name, "====")
    print(df.shape)
    print(df.isnull().sum())
    print(df.head())
players.columns = players.columns.str.lower().str.replace(" ", "_")

players.drop_duplicates(subset="player_id", keep="first", inplace=True)

players["name"] = players["name"].str.strip().str.title()
player_valuations['date'] = pd.to_datetime(player_valuations['date'], errors='coerce')

player_valuations = player_valuations.dropna(subset=['value'])

market = player_valuations.merge(players, on="player_id", how="left")
