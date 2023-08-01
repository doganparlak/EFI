from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import math
import json
from datetime import datetime
import copy
from matplotlib.axes import Axes
from matplotlib.transforms import Affine2D
from matplotlib.patches import Arc
from sys import intern
import pickle
import networkx as nx
from itertools import groupby
import operator
from sympy import Point, Line, pi
import floodlight.io.tracab as tr
from IPython.display import display
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
import optuna

class Match:
    
    def __init__(self, path):
        """ Initialize the match class parameters.
        Parameters
        ----------
        path : string
            path to match data.
        """
        try:
            self.meta, self.lineups = self.__parse(path)    
        except:
            print("Path Not Found!")
    def __parse(self, path):
        """ Parse the given match data.
        Parameters
        ----------
        path : string
            path to match data
        
        Return
        ----------
        meta : pandas data frame
            metadata of the match.
        lineups: pandas data frame
            lineups of both teams.
        """
        with open(path, encoding="utf-8") as f: # Open the event json file to read it
            match_dic = json.load(f)    
            meta_dic = {}
            meta_dic["match_id"] = match_dic["GameID"]
            meta_dic["home_team_id"] =  match_dic["HomeTeam"]["TeamID"]
            meta_dic["home_team_name"] =  match_dic["HomeTeam"]["ShortName"]
            meta_dic["away_team_id"] = match_dic["AwayTeam"]["TeamID"]
            meta_dic["away_team_name"] =  match_dic["AwayTeam"]["ShortName"]

            meta_dic["Phase1StartFrame"] = match_dic["Phase1StartFrame"]
            meta_dic["Phase1EndFrame"] = match_dic["Phase1EndFrame"]
            meta_dic["Phase2StartFrame"] = match_dic["Phase2StartFrame"]
            meta_dic["Phase2EndFrame"] = match_dic["Phase2EndFrame"]
            meta_dic["Phase3StartFrame"] = match_dic["Phase3StartFrame"]
            meta_dic["Phase3EndFrame"] = match_dic["Phase3EndFrame"]
            meta_dic["Phase4StartFrame"] = match_dic["Phase4StartFrame"]
            meta_dic["Phase4EndFrame"] = match_dic["Phase4EndFrame"]
            meta_dic["Phase5StartFrame"] = match_dic["Phase5StartFrame"]
            meta_dic["Phase5EndFrame"] = match_dic["Phase5EndFrame"]
        # save the team-player information
        lineup_dics = []
        for team in ["HomeTeam","AwayTeam"]:
            team_dic = match_dic[team]
            for player in team_dic["Players"]:
                lineup_dic = {}
                lineup_dic["team_id"] = team_dic["TeamID"]
                lineup_dic["player_id"] = player["PlayerID"]
                lineup_dic["player_first_name"] = player["FirstName"]
                lineup_dic["player_last_name"] = player["LastName"]
                lineup_dic["player_jersey_number"] = player["JerseyNo"]
                lineup_dic["player_starting_frame"] = player["StartFrameCount"]
                lineup_dic["player_ending_frame"] = player["EndFrameCount"]
                lineup_dics.append(lineup_dic)

        return pd.DataFrame([meta_dic]), pd.DataFrame(lineup_dics) 