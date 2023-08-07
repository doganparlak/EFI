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

class Tracking:
    def __init__(self, path, match, home_keeper_jersey = "1", away_keeper_jersey = "1", duel_zone = 2):
        """ Initialize the tracking class parameters.
        Parameters
        ----------
        path : string
            path to tracking data.
        match : match class object
            object consisting general match information.
        home_keeper_jersey: string
            home keeper jersey number.
        away_keeper_jersey: string
            away keeper jersey number.
        duel_zone: float, optional
            radius to define the duel zone.
        """
        self.tracking = self.__parse(path, match)            
        self.path = path
        self.home_coords, self.away_coords, self.home_coords_modified, self.away_coords_modified, self.home_jersey, self.away_jersey, self.home_speed, self.away_speed,  \
        self.ball_coords, self.ball_height, self.ball_state, self.ball_speed, self.ball_coords_modified_home, self.ball_coords_modified_away,\
        self.possession_info_raw, self.dirs, self.match_len = self.__get_coords(match, home_keeper_jersey, away_keeper_jersey)
        
        self.home_coords = self.__scale_coords(self.home_coords)
        self.away_coords = self.__scale_coords(self.away_coords)
        self.home_coords_modified = self.__scale_coords(self.home_coords_modified)
        self.away_coords_modified = self.__scale_coords(self.away_coords_modified)
        self.ball_coords = self.__scale_coords(self.ball_coords) 
        self.ball_coords_modified_home = self.__scale_coords(self.ball_coords_modified_home) 
        self.ball_coords_modified_away = self.__scale_coords(self.ball_coords_modified_away) 

        self.possession_info, self.possession_info_detailed = self.__get_coords_events(dz = duel_zone)
    def __parse(self, path, match):
        """ Parse the given match data.
        Parameters
        ----------
        path : string
            path to event data.
        match : match class object
            object consisting general match information.
        Return
        ----------
        tracking : pandas data frame
            recorded set of frames in the tracking data
        """
        tracking_dics = []
        num_lines = sum(1 for line in open(path,'r'))
        # Define a dictionary mapping the home (1) and away (0) teams to their values in a match
        team_id_mapping = {"1": match.meta.home_team_id.iloc[0], "0": match.meta.away_team_id.iloc[0], "3": "3"}
        # The tracking file will contain both player and ball tracking data
        with open(path, "r") as f:
            for line in tqdm(f, total=num_lines, disable=True): # Read line by line and extract the information pieces as elaborated in the documentation.
                frame_number, objects_xy_coordinates, ball_data, _ = line.split(":")
                for object in objects_xy_coordinates.split(";"):
                    if object=="":
                        break
                    team, not_known, jersey_number, x, y, speed = object.split(",")            
                    if team not in ["1", "0", "3"]: # If the object (i.e., person) does not belong to the Home (1) or Away (0) team. Then, ignore it. You can also keep the referees (3) if you want to study them later! 
                        continue

                    tracking_dic = {}
                    tracking_dic["frame"] = int(frame_number)
                    tracking_dic["team_id"] = int(team_id_mapping[team])
                    tracking_dic["player_jersey_number"] = jersey_number
                    tracking_dic["x"] = float(x)
                    tracking_dic["y"] = float(y)
                    tracking_dic["z"] = None
                    tracking_dic["speed"] = float(speed)
                    tracking_dic["state"] = None
                    tracking_dics.append(tracking_dic)
                
                ball_x, ball_y, ball_z, ball_speed, ball_owning_team, ball_state = ball_data.split(",")[:6]

                tracking_dic = {}
                tracking_dic["frame"] = int(frame_number)
                tracking_dic["team_id"] = -1 # Let's say -1 is the ball id
                tracking_dic["player_jersey_number"] = None
                tracking_dic["x"] = float(ball_x)
                tracking_dic["y"] = float(ball_y)
                tracking_dic["z"] = float(ball_z)
                tracking_dic["poss_team"] = str(ball_owning_team)
                tracking_dic["speed"] = float(ball_speed)
                tracking_dic["state"] = ball_state.replace(";","")
                tracking_dics.append(tracking_dic)

        return pd.DataFrame(tracking_dics) # Transform the list of dictionaries into a panda dataframe
    
    def __get_coords(self, match, home_keeper_jersey, away_keeper_jersey):
        """ Get the coordinates and the relevant information from the tracking data.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        home_keeper_jersey: string
            home keeper jersey number.
        away_keeper_jersey: string
            away keeper jersey number.

        Return 
        ---------- home_coords, away_coords, home_coords_modified, away_coords_modified, home_jersey, away_jersey, home_speed, away_speed, \
        ball_coords, ball_height, ball_state, ball_speed, ball_coords_modified_home, ball_coords_modified_away, possession_info_raw, dirs, match_len
        home_coords: numpy array
            home team coordinates at each frame.
        away_coords: numpy array
            away team coordinates at each frame.
        home_coords_modified: numpy array
            modified home team coordinates at each frame.
        away_coords_modified: numpy array
            modified away team coordinates at each frame.
        home_jersey: list
            jersey number of home team players at each frame.
        away_jersey: list
            jersey number of away team players at each frame.
        home_speed: numpy array
            speed of home team players at each frame.
        away_speed: numpy array
            speed of away team players at each frame.
        ball_coords: numpy array
            ball coordinates at each frame.
        ball_height: numpy array
            height of the ball at each frame.
        ball_state: numpy array
            state of the ball at each frame. 
        ball_coords_modified_home: numpy array 
            modified ball coords according to home team attacking direction.
        ball_coords_modified_away: numpy array 
            modified ball coords according to away team attacking direction.
        possession_info_raw: numpy array
            possession information provided in the tracking data.
        dirs: numpy array
            direction array that is used to convert the team coords to modified versions.
        match_len: integer
            number of periods in the game.
        """
        #ARRAY INITALIZATION
        home_coords = []
        away_coords = []
        ball_coords = []
        home_coords_modified = []
        away_coords_modified = []

        #FRAMES WHICH MATCH TAKES PLACE
        match_len = 2
        frames = list(range(match.meta["Phase1StartFrame"][0], match.meta["Phase1EndFrame"][0])) + list(range(match.meta["Phase2StartFrame"][0], match.meta["Phase2EndFrame"][0]))
        if  match.meta["Phase3StartFrame"][0] != 0:
            frames += list(range(match.meta["Phase3StartFrame"][0], match.meta["Phase3EndFrame"][0]))
            frames += list(range(match.meta["Phase4StartFrame"][0], match.meta["Phase4EndFrame"][0]))
            match_len = 4
            if match.meta["Phase5StartFrame"][0] != 0:
                match_len = 5
                frames += list(range(match.meta["Phase5StartFrame"][0], match.meta["Phase5EndFrame"][0]))

        #SPLIT FRAMES BY TEAM AND BALL
        relevant_frames =self.tracking.loc[self.tracking['frame'].isin(frames)]
        home_frames = relevant_frames[relevant_frames["team_id"] ==  match.meta.home_team_id[0]]
        away_frames = relevant_frames[relevant_frames["team_id"] ==  match.meta.away_team_id[0]]
        ball_frames = relevant_frames[relevant_frames["team_id"] ==  -1]
        
        #GET BALL COORDS
        ball_coords = np.array(list(zip(ball_frames["x"], ball_frames["y"])))
        ball_height = np.array(ball_frames["z"]) / 100 
        ball_state = np.array(ball_frames["state"])
        ball_speed = np.array(ball_frames["speed"])
        possession_info_raw = np.array(ball_frames["poss_team"])

        #GET HOME COORDS
        home_group_x = home_frames.groupby('frame')["x"].apply(list)
        home_group_y = home_frames.groupby('frame')["y"].apply(list)
        home_group_xy = pd.concat([home_group_x,home_group_y], axis = 1)
        for index,row in home_group_xy.iterrows():
            temp = np.array(list(zip(row["x"], row["y"])))
            home_coords.append(temp)
        #GET HOME JERSEY
        home_jersey = list(home_frames.groupby('frame')["player_jersey_number"].apply(list).values)
        #GET HOME SPEED
        home_speed = list(home_frames.groupby('frame')["speed"].apply(list).values)

        #GET AWAY COORDS
        away_group_x = away_frames.groupby('frame')["x"].apply(list)
        away_group_y = away_frames.groupby('frame')["y"].apply(list)
        away_group_xy = pd.concat([away_group_x,away_group_y], axis = 1)
        for index,row in away_group_xy.iterrows():
            temp = np.array(list(zip(row["x"], row["y"])))
            away_coords.append(temp)
        #GET AWAY JERSEY
        away_jersey = list(away_frames.groupby('frame')["player_jersey_number"].apply(list).values)
        #GET AWAY SPEED
        away_speed = list(away_frames.groupby('frame')["speed"].apply(list).values)

        #FIND RELEVANT DIRECTION FOR ADJUSTING COORDS
        home_goal_keeper_frames = home_frames[home_frames["player_jersey_number"] == home_keeper_jersey]
        away_goal_keeper_frames = away_frames[away_frames["player_jersey_number"] == away_keeper_jersey]
        x_vals_home = np.array(home_goal_keeper_frames["x"])
        x_vals_away = np.array(away_goal_keeper_frames["x"])
        dirs = np.zeros((len(x_vals_home),))
        if len(x_vals_home) >= len(x_vals_away):
            dirs[np.where(x_vals_home > 0)] = 1
            dirs[np.where(x_vals_home < 0)] = -1
        else:
            dirs = np.zeros((len(x_vals_away),))
            dirs[np.where(x_vals_away > 0)] = -1
            dirs[np.where(x_vals_away < 0)] = 1

        #GET MODIFIED HOME AND AWAY COORDS

        dirs = np.array(dirs)
        home_coords = np.array(home_coords)
        away_coords = np.array(away_coords)
        broad_a = np.broadcast_to(dirs, home_coords.T.shape).T
        broad_b =  np.broadcast_to(dirs, ball_coords.T.shape).T
        home_coords_modified = broad_a * home_coords
        away_coords_modified = -broad_a * away_coords
        ball_coords_modified_home = broad_b * ball_coords
        ball_coords_modified_away =  -broad_b * ball_coords
        return home_coords, away_coords, home_coords_modified, away_coords_modified, home_jersey, away_jersey, home_speed, away_speed, \
        ball_coords, ball_height, ball_state, ball_speed, ball_coords_modified_home, ball_coords_modified_away, possession_info_raw, dirs, match_len
    
    def __scale_coords(self, coords):
        #Utility function to convert the given coordinates to meters that can be analyzed and visualized on football pitch.
        coords = coords / 100
        coords = coords.astype(int)
        coords = coords[...,::-1]
        coords = coords * np.array([1,-1])
        return coords
    
    def __get_coords_events(self, dz = 2, center_r = 6, center_y = 4, penalty_area_x = 13, 
                            penalty_area_y = 40, goal_line_x = 6, goal_line_y = 51, bound_box_x = 4, bound_y1 = 45, bound_y2 = 39,
                            goal_kick_x = 6, goal_kick_y = -48, corner_kick_x= 34, corner_kick_r = 2, throw_in_x = 33):
        #Utility function to apply automatic event detection algorithm.
        frame_len = len(self.home_coords)
        possession_info = []
        possession_info_detailed = []
        for f in range(frame_len):
            if self.ball_state[f] == 'Alive':
                home_dist_to_ball = np.sqrt(np.sum((self.home_coords[f] - self.ball_coords[f]) ** 2, axis = 1))
                away_dist_to_ball = np.sqrt(np.sum((self.away_coords[f] - self.ball_coords[f]) ** 2, axis = 1))
                home_in_dz = home_dist_to_ball[home_dist_to_ball < dz]
                away_in_dz = away_dist_to_ball[away_dist_to_ball < dz]
                if len(home_in_dz) > 0 and  len(away_in_dz) > 0:
                    possession_info_detailed.append('In_Contest')
                    possession_info.append('In_Contest')
                else:
                    if self.possession_info_raw[f] == "H":
                        possession_info.append('Home')
                        possession_info_detailed.append('Home')
                    elif self.possession_info_raw[f] == "A":
                        possession_info.append('Away')
                        possession_info_detailed.append('Away')
                    else:
                        possession_info.append('Dead')
                        possession_info_detailed.append('Free_Kick')
            else:
                possession_info.append("Dead")
                #kick-off
                home_dist_to_center = np.sqrt(np.sum((self.home_coords_modified[f] - np.array([0,0])) ** 2, axis = 1))
                away_dist_to_center = np.sqrt(np.sum((self.away_coords_modified[f] - np.array([0,0])) ** 2, axis = 1))
                home_in_center = home_dist_to_center[home_dist_to_center <= center_r]
                away_in_center = away_dist_to_center[away_dist_to_center <= center_r]
                home_in_back_half_case1 = self.home_coords[f].T[1][self.home_coords[f].T[1] >= center_y] 
                home_in_back_half_case2 = self.home_coords[f].T[1][self.home_coords[f].T[1] <= -center_y] 
                away_in_back_half_case2 = self.away_coords[f].T[1][self.away_coords[f].T[1] >= center_y] 
                away_in_back_half_case1 = self.away_coords[f].T[1][self.away_coords[f].T[1] <= -center_y] 
                if ((len(home_in_center) + len(away_in_center) >= 1) and 
                    ((len(home_in_back_half_case1) +len(away_in_back_half_case1) == 0) or (len(home_in_back_half_case2) +len(away_in_back_half_case2) == 0))):
                    possession_info_detailed.append("Kick_Off")
                else:
                    #penalty
                    penalty_area_home = self.home_coords[f][self.home_coords[f].T[0] >= -penalty_area_x] 
                    penalty_area_home = penalty_area_home[penalty_area_home.T[0] <= penalty_area_x] 
                    penalty_area_home = penalty_area_home[penalty_area_home.T[1] <= 53]
                    penalty_area_home = penalty_area_home[penalty_area_home.T[1] >= penalty_area_y]
                    
                    penalty_area_home_f = self.home_coords[f][self.home_coords[f].T[0] >= -penalty_area_x] 
                    penalty_area_home_f = penalty_area_home_f[penalty_area_home_f.T[0] <= penalty_area_x] 
                    penalty_area_home_f = penalty_area_home_f[penalty_area_home_f.T[1] >= -53]
                    penalty_area_home_f = penalty_area_home_f[penalty_area_home_f.T[1] <= -penalty_area_y]

                    penalty_area_away = self.away_coords[f][self.away_coords[f].T[0] >= -penalty_area_x] 
                    penalty_area_away = penalty_area_away[penalty_area_away.T[0] <= penalty_area_x] 
                    penalty_area_away = penalty_area_away[penalty_area_away.T[1] <= 53]
                    penalty_area_away = penalty_area_away[penalty_area_away.T[1] >= penalty_area_y]

                    penalty_area_away_f = self.away_coords[f][self.away_coords[f].T[0] >= -penalty_area_x] 
                    penalty_area_away_f = penalty_area_away_f[penalty_area_away_f.T[0] <= penalty_area_x] 
                    penalty_area_away_f = penalty_area_away_f[penalty_area_away_f.T[1] >= -53]
                    penalty_area_away_f = penalty_area_away_f[penalty_area_away_f.T[1] <= -penalty_area_y]
                    ball = (self.ball_coords[f][0] <= penalty_area_x) and (self.ball_coords[f][0] >= -penalty_area_x) and (self.ball_coords[f][1] >= penalty_area_y) and (self.ball_coords[f][1] <= 53)
                    ball_f = (self.ball_coords[f][0] <= penalty_area_x) and (self.ball_coords[f][0] >= -penalty_area_x) and (self.ball_coords[f][1] <= -penalty_area_y) and (self.ball_coords[f][1] >= -53)

                    if (len(penalty_area_home) == 1 and len(penalty_area_away) == 1)and ball:
                        keeper_home = (penalty_area_home[0][0] <= goal_line_x) and (penalty_area_home[0][0] >= -goal_line_x) and  (penalty_area_home[0][1] >= goal_line_y) and (penalty_area_home[0][1] <= 53)  
                        keeper_away = (penalty_area_away[0][0] <= goal_line_x) and (penalty_area_away[0][0] >= -goal_line_x) and  (penalty_area_away[0][1] >= goal_line_y) and (penalty_area_away[0][1] <= 53)           
                        penalty_home = (penalty_area_home[0][0] <= bound_box_x) and (penalty_area_home[0][0] >= -bound_box_x) and  (penalty_area_home[0][1] >= bound_y2) and (penalty_area_home[0][1] <= bound_y1)   
                        penalty_away = (penalty_area_away[0][0] <= bound_box_x) and (penalty_area_away[0][0] >= -bound_box_x) and  (penalty_area_away[0][1] >= bound_y2) and (penalty_area_away[0][1] <= bound_y1)
                        
                        if ((penalty_home and keeper_away) or (penalty_away and keeper_home)):
                            possession_info_detailed.append("Penalty")
                        else:
                            possession_info_detailed.append("Free_Kick")
                    elif(len(penalty_area_home_f) == 1 and len(penalty_area_away_f) == 1)and ball_f:
                        keeper_home_f = (penalty_area_home_f[0][0] <= goal_line_x) and (penalty_area_home_f[0][0] >= -goal_line_x) and  (penalty_area_home_f[0][1] <= -goal_line_y) and (penalty_area_home_f[0][1] >= -53)
                        keeper_away_f = (penalty_area_away_f[0][0] <= goal_line_x) and (penalty_area_away_f[0][0] >= -goal_line_x) and  (penalty_area_away_f[0][1] <= -goal_line_y) and (penalty_area_away_f[0][1] >= -53)
                        penalty_home_f = (penalty_area_home_f[0][0] <= bound_box_x) and (penalty_area_home_f[0][0] >= -bound_box_x) and  (penalty_area_home_f[0][1] <= -bound_y2) and (penalty_area_home_f[0][1] >= -bound_y1)
                        penalty_away_f = (penalty_area_away_f[0][0] <= bound_box_x) and (penalty_area_away_f[0][0] >= -bound_box_x) and  (penalty_area_away_f[0][1] <= -bound_y2) and (penalty_area_away_f[0][1] >= -bound_y1)
                        
                        if ((penalty_home_f and keeper_away_f) or (penalty_away_f and keeper_home_f)):
                            possession_info_detailed.append("Penalty")
                        else:
                            possession_info_detailed.append("Free_Kick")
                    else:
                       #goal-kick
                        goal_kick_home_f = self.home_coords[f][self.home_coords[f].T[0] >= -goal_kick_x] 
                        goal_kick_home_f = goal_kick_home_f[goal_kick_home_f.T[0] <= goal_kick_x]
                        goal_kick_home_f = goal_kick_home_f[goal_kick_home_f.T[1] >= -53]
                        goal_kick_home_f = goal_kick_home_f[goal_kick_home_f.T[1] <= goal_kick_y]
                        
                        goal_kick_home = self.home_coords[f][self.home_coords[f].T[0] >= -goal_kick_x] 
                        goal_kick_home = goal_kick_home[goal_kick_home.T[0] <= goal_kick_x]
                        goal_kick_home = goal_kick_home[goal_kick_home.T[1] <= 53]
                        goal_kick_home = goal_kick_home[goal_kick_home.T[1] >= -goal_kick_y]

                        goal_kick_away_f = self.away_coords[f][self.away_coords[f].T[0] >= -goal_kick_x] 
                        goal_kick_away_f = goal_kick_away_f[goal_kick_away_f.T[0] <= goal_kick_x]
                        goal_kick_away_f = goal_kick_away_f[goal_kick_away_f.T[1] >= -53]
                        goal_kick_away_f = goal_kick_away_f[goal_kick_away_f.T[1] <= goal_kick_y]

                        goal_kick_away = self.away_coords[f][self.away_coords[f].T[0] >= -goal_kick_x] 
                        goal_kick_away = goal_kick_away[goal_kick_away.T[0] <= goal_kick_x]
                        goal_kick_away = goal_kick_away[goal_kick_away.T[1] <= 53]
                        goal_kick_away = goal_kick_away[goal_kick_away.T[1] >= -goal_kick_y]

                        ball = (self.ball_coords[f][0] <= goal_kick_x) and (self.ball_coords[f][0] >= -goal_kick_x) and (self.ball_coords[f][1] >= -goal_kick_y) and (self.ball_coords[f][1] <= 53)
                        ball_f = (self.ball_coords[f][0] <= goal_kick_x) and (self.ball_coords[f][0] >= -goal_kick_x) and (self.ball_coords[f][1] <= goal_kick_y) and (self.ball_coords[f][1] >= -53)
                        if ((len(goal_kick_home_f) >= 1 and len(penalty_area_away_f) == 0 and ball_f) or (len(goal_kick_home) >= 1 and len(penalty_area_away) == 0 and ball) or
                            (len(goal_kick_away_f) >= 1 and len(penalty_area_home_f) == 0 and ball_f) or (len(goal_kick_away) >= 1 and len(penalty_area_home) == 0 and ball)) :
                            possession_info_detailed.append("Goal_Kick")
                        
                        else:
                            #corner-kick
                            corner_kick_dist_home_left_u =  np.sqrt(np.sum((self.home_coords_modified[f] - np.array([-corner_kick_x, 52.5])) ** 2, axis = 1))
                            corner_kick_dist_home_right_u =  np.sqrt(np.sum((self.home_coords_modified[f] - np.array([corner_kick_x, 52.5])) ** 2, axis = 1))
                            corner_kick_dist_away_left_u =  np.sqrt(np.sum((self.away_coords_modified[f] - np.array([-corner_kick_x, 52.5])) ** 2, axis = 1))
                            corner_kick_dist_away_right_u =  np.sqrt(np.sum((self.away_coords_modified[f] - np.array([corner_kick_x, 52.5])) ** 2, axis = 1))

                            corner_kick_dist_home_left_d =  np.sqrt(np.sum((self.home_coords_modified[f] - np.array([-corner_kick_x, -52.5])) ** 2, axis = 1))
                            corner_kick_dist_home_right_d =  np.sqrt(np.sum((self.home_coords_modified[f] - np.array([corner_kick_x, -52.5])) ** 2, axis = 1))
                            corner_kick_dist_away_left_d =  np.sqrt(np.sum((self.away_coords_modified[f] - np.array([-corner_kick_x, -52.5])) ** 2, axis = 1))
                            corner_kick_dist_away_right_d =  np.sqrt(np.sum((self.away_coords_modified[f] - np.array([corner_kick_x, -52.5])) ** 2, axis = 1))
                            
                            corner_kick_home_left_u =  corner_kick_dist_home_left_u[corner_kick_dist_home_left_u <= corner_kick_r]
                            corner_kick_home_right_u =  corner_kick_dist_home_right_u[corner_kick_dist_home_right_u <= corner_kick_r]
                            corner_kick_away_left_u =  corner_kick_dist_away_left_u[corner_kick_dist_away_left_u <= corner_kick_r]
                            corner_kick_away_right_u =  corner_kick_dist_away_right_u[corner_kick_dist_away_right_u <= corner_kick_r]

                            corner_kick_home_left_d =  corner_kick_dist_home_left_d[corner_kick_dist_home_left_d <= corner_kick_r]
                            corner_kick_home_right_d =  corner_kick_dist_home_right_d[corner_kick_dist_home_right_d <= corner_kick_r]
                            corner_kick_away_left_d =  corner_kick_dist_away_left_d[corner_kick_dist_away_left_d <= corner_kick_r]
                            corner_kick_away_right_d =  corner_kick_dist_away_right_d[corner_kick_dist_away_right_d <= corner_kick_r]

                            if (len(corner_kick_home_left_u) + len(corner_kick_home_right_u > 0) + len(corner_kick_away_left_u) + len(corner_kick_away_right_u)+
                               len(corner_kick_home_left_d) + len(corner_kick_home_right_d > 0) + len(corner_kick_away_left_d) + len(corner_kick_away_right_d)) >= 1:
                                possession_info_detailed.append("Corner_Kick")
                            
                            else:
                                #throw_in
                                throw_in_home_left = self.home_coords_modified[f][self.home_coords_modified[f].T[0] >= -38] 
                                throw_in_home_left = throw_in_home_left[throw_in_home_left.T[0] <= -throw_in_x]
                                throw_in_home_right = self.home_coords_modified[f][self.home_coords_modified[f].T[0] <= 38]
                                throw_in_home_right = throw_in_home_right[throw_in_home_right.T[0] >= throw_in_x]
                                
                                throw_in_away_left = self.away_coords_modified[f][self.away_coords_modified[f].T[0] >= -38] 
                                throw_in_away_left = throw_in_away_left[throw_in_away_left.T[0] <= -throw_in_x]
                                throw_in_away_right = self.away_coords_modified[f][self.away_coords_modified[f].T[0] <= 38]
                                throw_in_away_right = throw_in_away_right[throw_in_away_right.T[0] >= throw_in_x]
                                
                                ball_left = self.ball_coords[f][0]  <= -throw_in_x
                                ball_right = self.ball_coords[f][0]  >= throw_in_x
                                                                    
                                if (len(throw_in_home_left) + len(throw_in_home_right) + len(throw_in_away_left) + len(throw_in_away_right) >= 1) and (ball_left or ball_right):
                                        possession_info_detailed.append("Throw_In")
                
                                else:
                                        possession_info_detailed.append("Free_Kick")



        return np.array(possession_info), np.array(possession_info_detailed)
