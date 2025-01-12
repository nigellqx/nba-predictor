import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytz
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamdashboardbygeneralsplits
from nba_api.stats.endpoints import teamgamelog
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

__model = None
__scaler = None

# This function is obtained from ChatGPT
def convert_to_native_types(data):
    """Recursively convert NumPy and PyTorch types to native Python types."""
    if isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, np.float32) or isinstance(data, np.float64):
        return float(data)  # Convert numpy floats to native Python float
    elif isinstance(data, np.int32) or isinstance(data, np.int64):
        return int(data)  # Convert numpy ints to native Python int
    elif isinstance(data, torch.Tensor):
        return data.tolist()  # Convert PyTorch tensors to lists
    else:
        return data

class Model(nn.Module):
    def __init__(self, in_features=36, h1=16, h2=16, h3=16, out_features=2):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,h3)
        self.out = nn.Linear(h3,out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

def nba_predict(nba_team, nba_opponent, playing):
    result = []

    teams_dict = teams.get_teams()
    season = "2024-25"
    overall_nba_stats = pd.DataFrame()

    for team in teams_dict:
        teams_name = team['full_name']
        teams_id = team['id']
        team_log = teamgamelog.TeamGameLog(team_id=teams_id,season=season)
        team_game_log = team_log.get_data_frames()[0]
        if (teams_name == nba_team):
            team_id = teams_id
        if (teams_name == nba_opponent):
            opponent_id = teams_id
        team_game_log['Team'] = teams_name
        overall_nba_stats = pd.concat([overall_nba_stats, team_game_log], ignore_index=True)

    team_stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(team_id=team_id,season=season,season_type_all_star="Regular Season")
    team_data = team_stats.overall_team_dashboard.get_data_frame()
    team_games_played = team_data.iloc[0]['GP']

    opponent_stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(team_id=opponent_id,season=season,season_type_all_star="Regular Season")
    opponent_data = opponent_stats.overall_team_dashboard.get_data_frame()
    opponent_games_played = opponent_data.iloc[0]['GP']

    w_pct = team_data.iloc[0]['W_PCT']
    fg_pct = team_data.iloc[0]['FG_PCT']
    fg3_pct = team_data.iloc[0]['FG3_PCT']
    ft_pct = team_data.iloc[0]['FT_PCT']
    oreb = team_data.iloc[0]['OREB']/team_games_played
    reb = team_data.iloc[0]['REB']/team_games_played
    ast = team_data.iloc[0]['AST']/team_games_played
    stl = team_data.iloc[0]['STL']/team_games_played
    blk = team_data.iloc[0]['BLK']/team_games_played
    tov = team_data.iloc[0]['TOV']/team_games_played
    pf = team_data.iloc[0]['PF']/team_games_played
    pts = team_data.iloc[0]['PTS']/team_games_played
    
    opponent_w_pct = opponent_data.iloc[0]['W_PCT']
    opponent_fg_pct = opponent_data.iloc[0]['FG_PCT']
    opponent_fg3_pct = opponent_data.iloc[0]['FG3_PCT']
    opponent_ft_pct = opponent_data.iloc[0]['FT_PCT']
    opponent_oreb = opponent_data.iloc[0]['OREB']/opponent_games_played
    opponent_reb = opponent_data.iloc[0]['REB']/opponent_games_played
    opponent_ast = opponent_data.iloc[0]['AST']/opponent_games_played
    opponent_stl = opponent_data.iloc[0]['STL']/opponent_games_played
    opponent_blk = opponent_data.iloc[0]['BLK']/opponent_games_played
    opponent_tov = opponent_data.iloc[0]['TOV']/opponent_games_played
    opponent_pf = opponent_data.iloc[0]['PF']/opponent_games_played
    opponent_pts = opponent_data.iloc[0]['PTS']/opponent_games_played


    # Add Opponent Details to row
    duplicate_nba_stats = overall_nba_stats.copy()
    new_team_stats = pd.merge(overall_nba_stats, duplicate_nba_stats,on='Game_ID',suffixes=('','_Opponent'))
    new_team_stats = new_team_stats[new_team_stats['Team_ID'] != new_team_stats['Team_ID_Opponent']]
    new_team_stats = new_team_stats.drop(['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','DREB','GAME_DATE_Opponent','MATCHUP_Opponent','WL_Opponent','W_Opponent','L_Opponent','MIN_Opponent','FGM_Opponent','FGA_Opponent','FG3M_Opponent','FG3A_Opponent','FTM_Opponent','FTA_Opponent','DREB_Opponent'],axis=1)
    new_team_stats = new_team_stats.dropna()
    new_team_stats = new_team_stats[new_team_stats['Team'] == nba_team]
    
    win = 0
    total = 0 
    for i in range(10):
        if (new_team_stats.iloc[i]['WL'] == 'W'):
            win = win+1
        total = total + 1
    last_10_games = win/total if total>0 else 0
    
    win = 0
    total = 0
    for i in range(len(new_team_stats)):
        if (new_team_stats.iloc[i]['Team_Opponent']==nba_opponent):
            total = total + 1
            if (new_team_stats.iloc[i]['WL']=='W'):
                win = win + 1
    head_to_head = win/total if total>0 else 0            

    new_team_stats['GAME_DATE'] = pd.to_datetime(new_team_stats['GAME_DATE'],format="%b %d, %Y")
    new_team_stats['GAME_DATE'] = new_team_stats['GAME_DATE'].dt.tz_localize('UTC')
    today = datetime.now().astimezone(pytz.utc)
    hours_difference = (today - new_team_stats.iloc[0]['GAME_DATE']).total_seconds() / 3600
    if hours_difference < 52:
        back_to_back = 1
    else:
        back_to_back = 0
    
    home_away = 1

    one_playing = 0
    two_playing = 0
    three_playing = 0
    opponent_one_playing = 0
    opponent_two_playing = 0
    opponent_three_playing = 0
    if 1 in playing: one_playing = 1
    if 2 in playing: two_playing = 1
    if 3 in playing: three_playing = 1
    if 4 in playing: opponent_one_playing = 1
    if 5 in playing: opponent_two_playing = 1
    if 6 in playing: opponent_three_playing = 1
    
    data_to_be_scaled = np.array([team_id,w_pct,fg_pct,fg3_pct,ft_pct,oreb,reb,ast,stl,blk,tov,pf,pts, opponent_id,
                                  opponent_w_pct, opponent_fg_pct, opponent_fg3_pct, opponent_ft_pct,
                                  opponent_oreb,opponent_reb,opponent_ast,opponent_stl,opponent_blk,
                                  opponent_tov,opponent_pf,opponent_pts,last_10_games, head_to_head,
                                  back_to_back, home_away, one_playing, two_playing, three_playing,
                                  opponent_one_playing, opponent_two_playing, opponent_three_playing]).reshape(1,-1)

    columns = ['Team_ID', 'W_PCT','FG_PCT','FG3_PCT','FT_PCT','OREB','REB','AST','STL','BLK','TOV','PF','PTS','Team_ID_Opponent','W_PCT_Opponent',
               'FG_PCT_Opponent','FG3_PCT_Opponent','FT_PCT_Opponent','OREB_Opponent','REB_Opponent','AST_Opponent','STL_Opponent',
               'BLK_Opponent','TOV_Opponent','PF_Opponent','PTS_Opponent', 'Last_10_Game_W_PCT','Head_To_Head_W_PCT','Back-To-Back',
               'Home-Away','No.1 Plus Minus Playing','No.2 Plus Minus Playing','No.3 Plus Minus Playing','No.1 Plus Minus Opponent Playing',
               'No.2 Plus Minus Opponent Playing','No.3 Plus Minus Opponent Playing']
    columns_scale = ['Team_ID','OREB','REB','AST','STL','BLK','TOV','PF','PTS','Team_ID_Opponent','OREB_Opponent','REB_Opponent','AST_Opponent','STL_Opponent','BLK_Opponent','TOV_Opponent','PF_Opponent','PTS_Opponent']                              
    final = pd.DataFrame(data_to_be_scaled, columns=columns)
    final[columns_scale] = __scaler.transform(final[columns_scale])
    
    input = torch.FloatTensor(final.values)
    
    output = __model(input)

    win_loss = torch.argmax(output, dim=1).item()
    if win_loss == 1:
        team_win = nba_team
    else:
        team_win = nba_opponent

    team_probability = np.max(torch.softmax(output,dim=1).detach().numpy())

    result.append({
        'prediction': team_win,
        'probability': round(team_probability*100,2)
    })
    
    return convert_to_native_types(result)
    
def load_saved_artifacts():
    global __model
    global __scaler
    __scaler = joblib.load('./artifacts/scaler.pkl')
    __model = Model()
    __model.load_state_dict(torch.load('./artifacts/nba_prediction_model.pt',weights_only=True))
    __model.eval()

if __name__ == '__main__':
    load_saved_artifacts()
    print(nba_predict("Washington Wizards", "Cleveland Cavaliers",[1,3,5]))