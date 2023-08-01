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

class Visualizer():

    def draw_pitch(
        self,
        ax: Axes,
        pitch_center: tuple = (0, 0),
        pitch_length: float = 105,
        pitch_width: float = 68,
        linewidth: float = 1.2,
        linecolor="black",
        background_color=None,
        zorder: int = -10,
        orient_vertical: bool = False,
    ):
        """Draw a football pitch on a given axes.
        The pitch is fitted according to the provided center and width/length arguments.
        Scale is not guaranteed.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw the pitch on
        pitch_center : tuple
            Center of the pitch, by default (0, 34). The center is the point in the
            middle of the pitch, lengthwise and widthwise respectively. If orient_vertical
            is False (default), this translates to x and y axes.
        pitch_length : float
            Length of the pitch, by default 105
        pitch_width : float
            Width of the pitch, by default 68
        linewidth : float
            Width of the lines, passed to plot calls and patch initializations, by default 1.2
        linecolor : color
            Color of the lines, passed to plot calls and patch initializations, by default "black"
        background_color : color
            Color of the plot background as a matplotlib color, by default None
        zorder : int, optional
            Plotting order of the pitch on the axes, by default -10
        orient_vertical : bool, optional
            Change the pitch orientation to vertical, by default False
        """
        if orient_vertical:
            transform = Affine2D().rotate_deg(90).scale(-1, 1) + ax.transData
        else:
            transform = ax.transData
        x = lambda x: (x / 130) * pitch_length + pitch_center[0] - pitch_length / 2
        y = lambda y: (y / 90) * pitch_width + pitch_center[1] - pitch_width / 2
        rat_x = pitch_length / 130
        rat_y = pitch_width / 90
        plot_arguments = dict(
            color=linecolor, zorder=zorder, transform=transform, linewidth=linewidth
        )
        # Pitch Outline & Centre Line
        ax.plot([x(0), x(0)], [y(0), y(90)], **plot_arguments)
        ax.plot([x(0), x(130)], [y(90), y(90)], **plot_arguments)
        ax.plot([x(130), x(130)], [y(90), y(0)], **plot_arguments)
        ax.plot([x(130), x(0)], [y(0), y(0)], **plot_arguments)
        ax.plot([x(65), x(65)], [y(0), y(90)], **plot_arguments)
        # Left Penalty Area
        ax.plot([x(16.5), x(16.5)], [y(65.16), y(24.84)], **plot_arguments)
        ax.plot([x(0), x(16.5)], [y(65.16), y(65.16)], **plot_arguments)
        ax.plot([x(16.5), x(0)], [y(24.84), y(24.84)], **plot_arguments)
        # Right Penalty Area
        ax.plot([x(130), x(113.5)], [y(65.16), y(65.16)], **plot_arguments)
        ax.plot([x(113.5), x(113.5)], [y(65.16), y(24.84)], **plot_arguments)
        ax.plot([x(113.5), x(130)], [y(24.84), y(24.84)], **plot_arguments)
        # Left 6-yard Box
        ax.plot([x(0), x(5.5)], [y(54.16), y(54.16)], **plot_arguments)
        ax.plot([x(5.5), x(5.5)], [y(54.16), y(35.84)], **plot_arguments)
        ax.plot([x(5.5), x(0.5)], [y(35.84), y(35.84)], **plot_arguments)
        # Right 6-yard Box
        ax.plot([x(130), x(124.5)], [y(54.16), y(54.16)], **plot_arguments)
        ax.plot([x(124.5), x(124.5)], [y(54.16), y(35.84)], **plot_arguments)
        ax.plot([x(124.5), x(130)], [y(35.84), y(35.84)], **plot_arguments)

        # Prepare circles
        centre_circle = plt.Circle((x(65), y(45)), 9.15, fill=False, **plot_arguments)
        centre_spot = plt.Circle((x(65), y(45)), linewidth / 2, **plot_arguments)
        left_pen_spot = plt.Circle((x(11), y(45)), linewidth / 4, **plot_arguments)
        right_pen_spot = plt.Circle((x(119), y(45)), linewidth / 4, **plot_arguments)
        # Draw Circles
        ax.add_patch(centre_circle)
        ax.add_patch(centre_spot)
        ax.add_patch(left_pen_spot)
        ax.add_patch(right_pen_spot)
        # Prepare Arcs
        left_arc = Arc(
            (x(11), y(45)),
            height=18.3 * rat_y,
            width=18.3 * rat_x,
            angle=0,
            theta1=312,
            theta2=48,
            **plot_arguments,
        )
        right_arc = Arc(
            (x(119), y(45)),
            height=18.3 * rat_y,
            width=18.3 * rat_x,
            angle=0,
            theta1=128,
            theta2=232,
            **plot_arguments,
        )
        # Draw Arcs
        ax.add_patch(left_arc)
        ax.add_patch(right_arc)
        if background_color is not None:
            ax.set_facecolor(background_color)
    
    def basic_plotter_together(self, tracking, frame):
        """Visualize the given frame of the tracking data on a football pitch.
        Parameters
        ----------
        tracking : tracking class object
            object consisting tracking data features.
        frame : integer
            index value of the frame.
        """
       
        coords_home = tracking.home_coords[frame]
        coords_away = tracking.away_coords[frame]
        
        mus_home  = coords_home 
        mus_away  = coords_away
        fig = plt.figure(figsize=(6.8, 10.5))
        ax = plt.gca()
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        self.draw_pitch(ax, orient_vertical=True, pitch_center=(0, 0))  
        mus_home = np.asarray([[pair[0], pair[1]] for pair in mus_home])
        mus_away = np.asarray([[pair[0], pair[1]] for pair in mus_away])
        for p in range(len(coords_home)):
            ax.scatter([mus_home[p][0]], [mus_home[p][1]], s=100, color = 'red', edgecolors='k')
            ax.scatter([mus_away[p][0]], [mus_away[p][1]], s=100, color = 'blue', edgecolors='k')

        ax.text(-30, 54, s = tracking.possession_info_raw[frame], color='red')
        ax.scatter([tracking.ball_coords[frame][0]], [tracking.ball_coords[frame][1]], s=50, color = 'green', edgecolors='k')
            
    def possession_control_plotter(self, match, home_possession, in_contest, away_possession):
        """Visualize the possession control results on a horizontal bar plot.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        home_possession: float
            percent of home team in-possession.
        in_contest: float
            percent of in-contest.
        away_possession: float
            percent of away team in-possession.
        """
        category_names = [match.meta.home_team_name.values[0], 'In-Contest',
                  match.meta.away_team_name.values[0]]
        results = {"Possession \n Control ": [home_possession, in_contest, away_possession]}

        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = ["crimson", "silver", "deepskyblue"]
        fig, ax = plt.subplots(figsize=(12, 1.3))
        legend_size = 16
        data_size = 14
        title_size = 20
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.5,
                            label=colname, color=color)
            text_color = "black"
            ax.bar_label(rects, labels=[f'{w}%' for w in widths], label_type='center', color=text_color, fontsize = data_size, weight = "bold")

        poss_label = ax.get_yticklabels()[0]  # Get the first label (Possession Control)
        poss_label.set_fontsize(title_size)  # Set the font size
        poss_label.set_weight("bold")
        
        ax.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=len(category_names), fontsize= legend_size)
        
    def ball_recovery_plotter(self, match, home_recovery_time, away_recovery_time):
        """Visualize the ball recovery time results on a pie chart.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        home_recovery_time: float
            average home team recovery time duration.
        away_recovery_time: float
            average away team recovery time duration.
        """
        values = [home_recovery_time, away_recovery_time]
        color = ["deepskyblue", "crimson"]
        if values[0] > values[1]:
            color = ["crimson", "deepskyblue"]

        labels = [match.meta.home_team_name.values[0], match.meta.away_team_name.values[0]]
        label_color = 'black'
        value_color = 'black'
        title_color = 'black'

        def autopct_format(values):
            def my_format(pct):
                total = sum(values)
                val = pct*total/100.0
                return '{v:.2f} sec'.format(v=val)
            return my_format
        fig, ax = plt.subplots(figsize=(7,7))
        data_size = 16
        title_size = 20
        label_size = 16
        weight = "bold"
        fig.suptitle(f"Ball Recovery Time: \n {match.meta.home_team_name[0]} - {match.meta.away_team_name[0]}" , fontsize= title_size, color = title_color, weight= weight)
       
        pie = ax.pie(values, labels=labels,  autopct = autopct_format(values), startangle=90, colors = color, 
                                                                      textprops={'color': value_color, 'fontsize': data_size, 'weight' : weight})

        for label in pie[1]:
            label.set_color(label_color)  # Set the color for labels
            label.set_fontsize(label_size)
        ax.set_aspect('equal')  # Ensure pie is drawn as a circle
    
    def pressure_on_ball_plotter(self, match, tracking, pressure_home_coords, pressure_away_coords):
        """Visualize the pressure on the ball results on football pitch.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        tracking : tracking class object
            object consisting tracking data features.
        pressure_home_coords: List
            coordinates at the frames where home team applied pressure.
        pressure_away_coords: List
            coordinates at the frames where away team applied pressure.
        """
         
        fig, axs = plt.subplots(figsize = (7.4*2, 10.5), ncols=2, nrows=1)
        data_size = 14
        direction_size = 10
        title_size = 20
        weight = "bold"
        title_color = "black"
        fig.suptitle(f"Pressure on the Ball: {match.meta.home_team_name[0]} - {match.meta.away_team_name[0]}", fontsize = title_size, weight = weight, color = title_color)
        fig.subplots_adjust(wspace=0.05, top = 0.95)
        ax1 = axs[0]
        ax2 = axs[1]
        limx = 40
        # Specify the start and end locations of the arrow
        arrow_start = [38, 31]
        arrow_end = [38, 49]
        # Direction text start and end locations 
        text_x = 36
        text_y = 40
        text = "DIRECTION"
        
        ax1.set_xlim(-limx,limx)
        ax1.set_facecolor('white')
        ax1.set_xticks([])
        ax1.set_yticks([])
        self.draw_pitch(ax1, orient_vertical=True, pitch_center=(0, 0))  

        for h in pressure_home_coords:
            temp_ball_coords = tracking.ball_coords_modified_home[h]
            ax1.scatter([temp_ball_coords[0]], [temp_ball_coords[1]], s = 50, color = "crimson")

        ax1.text(-30, 54, s = match.meta.home_team_name.values[0] +" Defensive Pressure: "+str(len(pressure_home_coords) - 1), color='black', fontsize = data_size)

        # Add the arrow to the ax1
        ax1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax1
        ax1.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = direction_size, weight = "bold")

        ax2.set_xlim(-limx,limx)
        ax2.set_facecolor('white')
        ax2.set_xticks([])
        ax2.set_yticks([])
        self.draw_pitch(ax2, orient_vertical=True, pitch_center=(0, 0))  

        for h in pressure_away_coords:
            temp_ball_coords = tracking.ball_coords_modified_away[h]
            ax2.scatter([temp_ball_coords[0]], [temp_ball_coords[1]], s = 50, color = "deepskyblue")

        ax2.text(-30, 54, s = match.meta.away_team_name.values[0] +" Defensive Pressure: "+str(len(pressure_away_coords) - 1) , color='black', fontsize = data_size)

        # Add the arrow to the ax2
        ax2.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax2
        ax2.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = direction_size, weight = "bold")

    def forced_turnover_plotter(self, match, tracking, forced_turnover_home, forced_turnover_away):
        """Visualize the forced turnover results on football pitch.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        tracking : tracking class object
            object consisting tracking data features.
        forced_turnover_home: List
            coordinates at the frames where home team regain the possession.
        forced_turnover_away: List
            coordinates at the frames where away team regain the possession.
        """
        fig, axs = plt.subplots(figsize = (7.4*2, 10.5), ncols=2, nrows=1)
        data_size = 14
        direction_size = 10
        title_size = 20
        weight = "bold"
        title_color = "black"
        fig.suptitle(f"Forced Turnover: {match.meta.home_team_name[0]} - {match.meta.away_team_name[0]}" , fontsize= title_size, color = title_color, weight= weight)
        fig.subplots_adjust(wspace=0.05, top = 0.95)
        ax1 = axs[0]
        ax2 = axs[1]
        limx = 40
        # Specify the start and end locations of the arrow
        arrow_start = [38, 31]
        arrow_end = [38, 49]
        # Direction text start and end locations 
        text_x = 36
        text_y = 40
        text = "DIRECTION"

        ax1.set_xlim(-limx,limx)
        ax1.set_facecolor('white')
        ax1.set_xticks([])
        ax1.set_yticks([])
        self.draw_pitch(ax1, orient_vertical=True, pitch_center=(0, 0))  

        for f in forced_turnover_home:
            temp_ball_coords = tracking.ball_coords_modified_home[f]
            ax1.scatter([temp_ball_coords[0]], [temp_ball_coords[1]], s = 50, color = "crimson")
        
        ax1.text(-30, 54, s = match.meta.home_team_name.values[0] +" Possession Regains: "+str(len(forced_turnover_home)), color='black', fontsize = data_size)
       
        # Add the arrow to the ax1
        ax1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax1
        ax1.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = direction_size, weight = "bold")

        ax2.set_xlim(-limx,limx)
        ax2.set_facecolor('white')
        ax2.set_xticks([])
        ax2.set_yticks([])
        self.draw_pitch(ax2, orient_vertical=True, pitch_center=(0, 0))  

        for h in forced_turnover_away:
            temp_ball_coords = tracking.ball_coords_modified_away[h]
            ax2.scatter([temp_ball_coords[0]], [temp_ball_coords[1]], s = 50, color = "deepskyblue")

        ax2.text(-30, 54, s = match.meta.away_team_name.values[0] +" Possession Regains: "+str(len(forced_turnover_away)) , color='black', fontsize = data_size)
        
        # Add the arrow to the ax2
        ax2.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax2
        ax2.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = direction_size, weight = "bold")

    def team_shape_plotter(self, match, tracking, home_shape, away_shape, home_in_pos_shape, away_in_pos_shape, home_out_pos_shape, away_out_pos_shape):
        """Visualize the team shape results on football pitch.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        tracking : tracking class object
            object consisting tracking data features.
        home_shape: List
            overall home team shape.
        away_shape: List
            overall away team shape.
        home_in_pos_shape: List
            in-possession home team shape.
        away_in_pos_shape: List
            in-possession away team shape.
        home_out_pos_shape: List
            out-of-possession home team shape.
        away_out_pos_shape: List
            out-of-possession away team shape.
        """
        home_coords_dic = {}
        away_coords_dic = {}
        home_in_pos_coords_dic = {}
        away_in_pos_coords_dic = {}
        home_out_pos_coords_dic = {}
        away_out_pos_coords_dic = {}
        for j in tracking.home_jersey[0]:
            home_coords_dic[j] = []
            home_in_pos_coords_dic[j] = []
            home_out_pos_coords_dic[j] = []

        for j in tracking.away_jersey[0]:
            away_coords_dic[j] = []
            away_in_pos_coords_dic[j] = []
            away_out_pos_coords_dic[j] = []

        home_flag = True
        away_flag = True
        cnt = 0
        for i,p in enumerate(tracking.possession_info_raw):
            if home_flag == False or away_flag == False:
                break
            if tracking.ball_state[i] == "Alive":
                #Overall Formation
                if home_flag == True:
                    for k in home_coords_dic.keys():
                        try:
                            player_index = tracking.home_jersey[i].index(k)
                        except:
                            home_flag = False
                            break
                        player_coords = tracking.home_coords_modified[i][player_index]
                        home_coords_dic[k].append(player_coords)
                    cnt += 1
                if away_flag == True:
                    for k in away_coords_dic.keys():
                        try:
                            player_index = tracking.away_jersey[i].index(k)
                        except:
                            away_flag = False
                            break
                        player_coords = tracking.away_coords_modified[i][player_index]
                        away_coords_dic[k].append(player_coords)
                #Home Possession
                if p == "H":
                    if home_flag == True:
                        for k in home_in_pos_coords_dic.keys():
                            try:
                                player_index = tracking.home_jersey[i].index(k)
                            except:
                                home_flag = False
                                break
                            player_coords = tracking.home_coords_modified[i][player_index]
                            home_in_pos_coords_dic[k].append(player_coords)
                    if away_flag == True:
                        for k in away_out_pos_coords_dic.keys():
                            try:
                                player_index = tracking.away_jersey[i].index(k)
                            except:
                                away_flag = False
                                break
                            player_coords = tracking.away_coords_modified[i][player_index]
                            away_out_pos_coords_dic[k].append(player_coords)
                #Away Possession
                elif p == "A":
                     if away_flag == True:
                        for k in away_in_pos_coords_dic.keys():
                            try:
                                player_index = tracking.away_jersey[i].index(k)
                            except:
                                away_flag = False
                                break
                            player_coords = tracking.away_coords_modified[i][player_index]
                            away_in_pos_coords_dic[k].append(player_coords)
                     if home_flag == True:
                        for k in home_out_pos_coords_dic.keys():
                            try:
                                player_index = tracking.home_jersey[i].index(k)
                            except:
                                home_flag = False
                                break
                            player_coords = tracking.home_coords_modified[i][player_index]
                            home_out_pos_coords_dic[k].append(player_coords)
                    
        home_coords = []
        for k,v in home_coords_dic.items():
            home_coords.append(np.average(np.array(v), axis = 0))
        home_in_pos_coords = []
        for k,v in home_in_pos_coords_dic.items():
            home_in_pos_coords.append(np.average(np.array(v), axis = 0))
        home_out_pos_coords = []
        for k,v in home_out_pos_coords_dic.items():
            home_out_pos_coords.append(np.average(np.array(v), axis = 0))

        away_coords = []
        for k,v in away_coords_dic.items():
            away_coords.append(np.average(np.array(v), axis = 0))
        away_in_pos_coords = []
        for k,v in away_in_pos_coords_dic.items():
            away_in_pos_coords.append(np.average(np.array(v), axis = 0))
        away_out_pos_coords = []
        for k,v in away_out_pos_coords_dic.items():
            away_out_pos_coords.append(np.average(np.array(v), axis = 0))

        
        fig, axs = plt.subplots(figsize = (6.8 * 3, 10.5 * 2), ncols=3, nrows=2)
        title_size = 25
        weight = "bold"
        title_color = "black"
        fig.suptitle(f"Team Shape: {match.meta.home_team_name[0]} - {match.meta.away_team_name[0]}" , fontsize= title_size, color = title_color, weight = weight)
        fig.subplots_adjust(wspace=0.05,  hspace=0.1, top = 0.95)
        ax1 = axs[0][0]
        ax2 = axs[0][1]
        ax3 = axs[0][2]
        ax4 = axs[1][0]
        ax5 = axs[1][1]
        ax6 = axs[1][2]

        # Specify the start and end locations of the arrow
        arrow_start = [31, 31]
        arrow_end = [31, 49]
        # Direction text start and end locations 
        text_x = 29
        text_y = 40
        text = "DIRECTION"

        s = 140
        fontsize = 14.5
        text_start = -34
        print(text_start)
        # Home Overall Shape
        ax1.set_facecolor('white')
        ax1.set_xticks([])
        ax1.set_yticks([])
        self.draw_pitch(ax1, orient_vertical=True, pitch_center=(0, 0))  
        
        for h in home_coords:
            ax1.scatter([h[0]], [h[1]], s = s, color = "crimson")

        if len(home_shape) == 9:
            ax1.text(text_start, 54, s = match.meta.home_team_name.values[0] +" Overall Shape: ("+ str(home_shape[1]) + "-" + str(home_shape[4]) + "-" + str(home_shape[7]) + ")", color='black', fontsize = fontsize)
        elif len(home_shape) == 12:
            ax1.text(text_start, 54, s = match.meta.home_team_name.values[0] +" Overall Shape: ("+ str(home_shape[1]) + "-" + str(home_shape[4]) + "-" + str(home_shape[7]) + "-" + str(home_shape[10]) + ")", color='black', fontsize = fontsize)

       
        # Add the arrow to the ax1
        ax1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax1
        ax1.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = 10, weight = "bold")

        # Home In-Possession Shape
        ax2.set_facecolor('white')
        ax2.set_xticks([])
        ax2.set_yticks([])
        self.draw_pitch(ax2, orient_vertical=True, pitch_center=(0, 0))  
        for h in home_in_pos_coords:
            ax2.scatter([h[0]], [h[1]], s = s, color = "crimson")

        if len(home_in_pos_shape) == 9:
            ax2.text(text_start, 54, s = match.meta.home_team_name.values[0] +" In-Possession Shape: ("+ str(home_in_pos_shape[1]) + "-" + str(home_in_pos_shape[4]) + "-" + str(home_in_pos_shape[7]) + ")", color='black', fontsize = fontsize)
        elif len(home_in_pos_shape) == 12:
            ax2.text(text_start, 54, s = match.meta.home_team_name.values[0] +" In-Possession Shape: ("+ str(home_in_pos_shape[1]) + "-" + str(home_in_pos_shape[4]) + "-" + str(home_in_pos_shape[7]) + "-" + str(home_in_pos_shape[10]) + ")", color='black', fontsize = fontsize)

        # Add the arrow to the ax2
        ax2.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax2
        ax2.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = 10, weight = "bold")


        # Home Out-of-Possession Shape
        ax3.set_facecolor('white')
        ax3.set_xticks([])
        ax3.set_yticks([])
        self.draw_pitch(ax3, orient_vertical=True, pitch_center=(0, 0))  
        for h in home_out_pos_coords:
            ax3.scatter([h[0]], [h[1]], s = s, color = "crimson")

        if len(home_out_pos_shape) == 9:
            ax3.text(text_start, 54, s = match.meta.home_team_name.values[0] +" Out-of-Possession Shape: ("+ str(home_out_pos_shape[1]) + "-" + str(home_out_pos_shape[4]) + "-" + str(home_out_pos_shape[7]) + ")", color='black', fontsize = fontsize)
        elif len(home_out_pos_shape) == 12:
            ax3.text(text_start, 54, s = match.meta.home_team_name.values[0] +" Out-of-Possession Shape: ("+ str(home_out_pos_shape[1]) + "-" + str(home_out_pos_shape[4]) + "-" + str(home_out_pos_shape[7]) + "-" + str(home_out_pos_shape[10]) + ")", color='black', fontsize = fontsize)

        # Add the arrow to the ax3
        ax3.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax3
        ax3.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = 10, weight = "bold")
        
        # Away Overall Shape
        ax4.set_facecolor('white')
        ax4.set_xticks([])
        ax4.set_yticks([])
        self.draw_pitch(ax4, orient_vertical=True, pitch_center=(0, 0))  
        for h in away_coords:
            ax4.scatter([h[0]], [h[1]], s = s, color = "deepskyblue")

        if len(away_shape) == 9:
            ax4.text(text_start, 54, s = match.meta.away_team_name.values[0] +" Overall Shape: ("+ str(away_shape[1]) + "-" + str(away_shape[4]) + "-" + str(away_shape[7]) + ")", color='black', fontsize =fontsize)
        elif len(away_shape) == 12:
            ax4.text(text_start, 54, s = match.meta.away_team_name.values[0] +" Overall Shape: ("+ str(away_shape[1]) + "-" + str(away_shape[4]) + "-" + str(away_shape[7]) + "-" + str(away_shape[10]) + ")", color='black', fontsize = fontsize)

        # Add the arrow to the ax4
        ax4.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax4
        ax4.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = 10, weight = "bold")

        # Away In-Possession Shape
        ax5.set_facecolor('white')
        ax5.set_xticks([])
        ax5.set_yticks([])
        self.draw_pitch(ax5, orient_vertical=True, pitch_center=(0, 0))  
        for h in away_in_pos_coords:
            ax5.scatter([h[0]], [h[1]], s = s, color = "deepskyblue")

        if len(away_in_pos_shape) == 9:
            ax5.text(text_start, 54, s = match.meta.away_team_name.values[0] +" In-Possession Shape: ("+ str(away_in_pos_shape[1]) + "-" + str(away_in_pos_shape[4]) + "-" + str(away_in_pos_shape[7]) + ")", color='black', fontsize = fontsize)
        elif len(away_in_pos_shape) == 12:
            ax5.text(text_start, 54, s = match.meta.away_team_name.values[0] +" In-Possession Shape: ("+ str(away_in_pos_shape[1]) + "-" + str(away_in_pos_shape[4]) + "-" + str(away_in_pos_shape[7]) + "-" + str(away_in_pos_shape[10]) + ")", color='black', fontsize = fontsize)


        # Add the arrow to the ax5
        ax5.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax5
        ax5.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = 10, weight = "bold")

        # Away Out-of-Possession Shape
        ax6.set_facecolor('white')
        ax6.set_xticks([])
        ax6.set_yticks([])
        self.draw_pitch(ax6, orient_vertical=True, pitch_center=(0, 0))  
        for h in away_out_pos_coords:
            ax6.scatter([h[0]], [h[1]], s = s, color = "deepskyblue")

        if len(away_out_pos_shape) == 9:
            ax6.text(text_start, 54, s = match.meta.away_team_name.values[0] +" Out-of-Possession Shape: ("+ str(away_out_pos_shape[1]) + "-" + str(away_out_pos_shape[4]) + "-" + str(away_out_pos_shape[7]) + ")", color='black', fontsize = fontsize)
        elif len(away_out_pos_shape) == 12:
            ax6.text(text_start, 54, s = match.meta.away_team_name.values[0] +" Out-of-Possession Shape: ("+ str(away_out_pos_shape[1]) + "-" + str(away_out_pos_shape[4]) + "-" + str(away_out_pos_shape[7]) + "-" + str(away_out_pos_shape[10]) + ")", color='black', fontsize = fontsize)

        # Add the arrow to the ax6
        ax6.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                  color='gray', width=0.4, length_includes_head=True)
        
        # Add the direction text to the ax6
        ax6.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = "gray", fontsize = 10, weight = "bold")

    def xG_plotter(self, match, shots_df, probs, home_xG_tot, away_xG_tot, home_score, away_score):
        """Visualize the xG results on football pitch.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        shots_df: pandas data frame
            data frame consisting shots taken at the given match.
        probs: list
            probabilities assigned to each shot.
        home_xG_tot: float
            xG of the home team.
        away_xG_tot: float
            xG of the away team.
        home_score: integer
            actual score of the home team.
        away_score: integer
            actual score of the away team.
        """
        probs = np.array(probs)
        home_team_id = match.meta.home_team_id[0]
        away_team_id = match.meta.away_team_id[0]
        home_df = shots_df[shots_df.team_id == home_team_id]
        away_df = shots_df[shots_df.team_id == away_team_id]
        home_shots_x = home_df["x_location_start_mirrored"].values
        home_shots_y = home_df["y_location_start_mirrored"].values
        away_shots_x = away_df["x_location_start_mirrored"].values
        away_shots_y = away_df["y_location_start_mirrored"].values
        home_body_type = home_df["body_type"].values
        away_body_type = away_df["body_type"].values
        home_xG = probs[home_df.index]
        away_xG = probs[away_df.index]

        y_test_home = shots_df[shots_df.team_id == home_team_id]["goal_info"].values
        y_test_away = shots_df[shots_df.team_id == away_team_id]["goal_info"].values
        home_origin = shots_df[shots_df.team_id == home_team_id]["origin"].values
        away_origin = shots_df[shots_df.team_id == away_team_id]["origin"].values
        coef = 1.2
        # star denoting a goal, purple: header, blue: right foot, green: left foot
        fig, axs = plt.subplots(figsize = (7.2 * 2 * coef , 5.9 * 1 * coef), ncols=2, nrows=1)
        title_size = 20
        weight = "bold"
        title_color = "black"
        fig.suptitle(f"Shot Locations and Expected Goals: \n {match.meta.home_team_name[0]} - {match.meta.away_team_name[0]}" , fontsize= title_size, color = title_color, weight = weight)
        fig.subplots_adjust(wspace=0.05)
        ax1 = axs[0]
        ax2 = axs[1]

        ax1.set_ylim(-2,56)
        ax1.set_facecolor('white')
        ax1.set_xticks([])
        ax1.set_yticks([])
        self.draw_pitch(ax1, orient_vertical=True, pitch_center=(0, 0))  

        shift_y = 1
        shift_x = -1
        shot_loc_size = 500
        fontsize = 9
        size_offset = 15
        right_foot_home = "red" 
        left_foot_home = "pink"
        head_home = "maroon"

        right_foot_away =  "blue"
        left_foot_away = "lightskyblue"
        head_away = "midnightblue"

        title_fontsize = 16
        legend_fontsize = 14
        for i in range(len(home_shots_x)):
            home_xG[i] = round(home_xG[i], 2)
            if home_origin[i] == "penalty":
                home_shots_x[i] = 0
                home_shots_y[i] = 44
            if y_test_home[i] == 1:
                if home_body_type[i] == "right_foot":
                    ax1.scatter([home_shots_x[i]], [home_shots_y[i]], s = shot_loc_size * home_xG[i] + size_offset, marker='^', color = right_foot_home)
                    #ax1.text(home_shots_x[i] + shift_x, home_shots_y[i] + shift_y , s = str(home_xG[i]), color = "black", fontsize = fontsize)
                
                elif home_body_type[i] == "left_foot":
                    ax1.scatter([home_shots_x[i]], [home_shots_y[i]], s = shot_loc_size * home_xG[i]+ size_offset, marker='^', color = left_foot_home)
                    #ax1.text(home_shots_x[i] + shift_x, home_shots_y[i] + shift_y , s = str(home_xG[i]), color = "black", fontsize = fontsize)
                
                elif home_body_type[i] == "head":
                    ax1.scatter([home_shots_x[i]], [home_shots_y[i]], s = shot_loc_size * home_xG[i]+ size_offset, marker='^', color = head_home)
                    #ax1.text(home_shots_x[i] + shift_x, home_shots_y[i] + shift_y, s = str(home_xG[i]), color = "black", fontsize = fontsize)
            
            elif y_test_home[i] == 0:
                if home_body_type[i] == "right_foot":
                    ax1.scatter([home_shots_x[i]], [home_shots_y[i]], s = shot_loc_size * home_xG[i]+ size_offset, color = right_foot_home)
                    #ax1.text(home_shots_x[i] + shift_x, home_shots_y[i] + shift_y, s = str(home_xG[i]), color = "black", fontsize = fontsize)
                
                elif home_body_type[i] == "left_foot":
                    ax1.scatter([home_shots_x[i]], [home_shots_y[i]], s = shot_loc_size * home_xG[i]+ size_offset, color = left_foot_home)
                    #ax1.text(home_shots_x[i] + shift_x, home_shots_y[i] + shift_y, s = str(home_xG[i]), color = "black", fontsize = fontsize)
                
                elif home_body_type[i] == "head":
                    ax1.scatter([home_shots_x[i]], [home_shots_y[i]], s = shot_loc_size * home_xG[i]+ size_offset, color = head_home)
                    #ax1.text(home_shots_x[i] + shift_x, home_shots_y[i] + shift_y, s = str(home_xG[i]), color = "black", fontsize = fontsize)
        
        ax1.text(-30, 53 , s = match.meta.home_team_name.values[0] + " Shots - (xG: " + str(home_xG_tot) + " - Score: " + str(home_score) + ")", color = "black", fontsize = title_fontsize)
        
        custom_handles = [plt.Line2D([], [], marker='o', color='black', linestyle='None'),
                          plt.Line2D([], [], marker='^', color='black', linestyle='None'),
                          plt.Line2D([], [], marker='s', color= right_foot_home, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= left_foot_home , linestyle='None'),
                          plt.Line2D([], [], marker='s', color= head_home, linestyle='None')]

        custom_labels = ['No Goal', 'Goal', 'Right Foot', 'Left Foot', 'Head']
        ax1.legend(custom_handles, custom_labels, loc='lower left', fontsize = legend_fontsize)

        ax2.set_ylim(-2,56)
        ax2.set_facecolor('white')
        ax2.set_xticks([])
        ax2.set_yticks([])   
        self.draw_pitch(ax2, orient_vertical=True, pitch_center=(0, 0))  
        for i in range(len(away_shots_x)):
            away_xG[i] = round(away_xG[i], 2)
            if away_origin[i] == "penalty":
                away_shots_x[i] = 0
                away_shots_y[i] = 44
            if y_test_away[i] == 1:
                if away_body_type[i] == "right_foot":
                    ax2.scatter([away_shots_x[i]], [away_shots_y[i]], s = shot_loc_size * away_xG[i]+ size_offset, marker='^', color = right_foot_away)
                    #ax2.text(away_shots_x[i] + shift_x, away_shots_y[i] + shift_y, s = str(away_xG[i]), color = "black", fontsize = fontsize)
                
                elif away_body_type[i] == "left_foot":
                    ax2.scatter([away_shots_x[i]], [away_shots_y[i]], s = shot_loc_size * away_xG[i]+ size_offset, marker='^', color = left_foot_away)
                    #ax2.text(away_shots_x[i] + shift_x, away_shots_y[i] + shift_y, s = str(away_xG[i]), color = "black", fontsize = fontsize)
                
                elif away_body_type[i] == "head":
                    ax2.scatter([away_shots_x[i]], [away_shots_y[i]], s = shot_loc_size * away_xG[i]+ size_offset, marker='^', color = head_away)
                    #ax2.text(away_shots_x[i] + shift_x, away_shots_y[i] + shift_y, s = str(away_xG[i]), color = "black", fontsize = fontsize)
            
            elif y_test_away[i] == 0:
                if away_body_type[i] == "right_foot":
                    ax2.scatter([away_shots_x[i]], [away_shots_y[i]], s = shot_loc_size * away_xG[i]+ size_offset, color = right_foot_away)
                    #ax2.text(away_shots_x[i] + shift_x, away_shots_y[i] + shift_y, s = str(away_xG[i]), color = "black", fontsize = fontsize)
                
                elif away_body_type[i] == "left_foot":
                    ax2.scatter([away_shots_x[i]], [away_shots_y[i]], s = shot_loc_size * away_xG[i]+ size_offset, color = left_foot_away)
                    #ax2.text(away_shots_x[i] + shift_x, away_shots_y[i] + shift_y, s = str(away_xG[i]), color = "black", fontsize = fontsize)
                
                elif away_body_type[i] == "head":
                    ax2.scatter([away_shots_x[i]], [away_shots_y[i]], s = shot_loc_size * away_xG[i]+ size_offset, color = head_away)
                    #ax2.text(away_shots_x[i] + shift_x, away_shots_y[i] + shift_y, s = str(away_xG[i]), color = "black", fontsize = fontsize)

        ax2.text(-30, 53 , s = match.meta.away_team_name.values[0] + " Shots - (xG: " + str(away_xG_tot) + " - Score: " + str(away_score) + ")",  color = "black", fontsize = title_fontsize)

        custom_handles = [plt.Line2D([], [], marker='o', color='black', linestyle='None'),
                          plt.Line2D([], [], marker='^', color='black', linestyle='None'),
                          plt.Line2D([], [], marker='s', color= right_foot_away, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= left_foot_away, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= head_away, linestyle='None')]
        custom_labels = ['No Goal', 'Goal', 'Right_Foot', 'Left Foot', 'Head']
        ax2.legend(custom_handles, custom_labels, loc='lower left', fontsize = legend_fontsize)

    def phases_of_play_plotter(self, match, home_in_phases, home_out_phases, away_in_phases, away_out_phases):
        """Visualize the phases of play results on horizontal bar chart.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        home_in_phases: list
            percent of times spent by home team in in-possession phases.
        away_in_phases: list
            percent of times spent by away team in in-possession phases.
        home_out_phases: list
            percent of times spent by home team in out-of-possession phases.
        away_out_phases: list
            percent of times spent by away team in out-of-possession phases.
        """
        in_pos_labels = ["Build Up \n Unopposed", "Build Up \n Opposed", "Progression", "Final Third", 
                        "Long Ball", "Attacking \n Transition", "Counter \n Attack", "Set Piece"]
        out_pos_labels = ["High Press", "Mid Press", "Low Press", "High Block", "Mid Block",
                          "Low Block", "Recovery", "Defensive \n Transition", "Counter \n Press"]
        home_team_name = match.meta.home_team_name[0]
        away_team_name = match.meta.away_team_name[0]
        # Set the width of each bar
        bar_width = 0.35
    
        fig, axs = plt.subplots(figsize = (8 * 1.5, 11 * 1.4), ncols=1, nrows=2)
        #fig.subplots_adjust(wspace=0.35)
         # Add a title to the figure
        title_size = 28
        weight = "bold"
        fig.suptitle(f"Phases of Play: {home_team_name} - {away_team_name}" , fontsize= title_size, weight = weight)

        ax1 = axs[0]
        ax2 = axs[1]

        # In-Possession Phases
        x_label = "Phase Distributions (%)"
        y_label = "In-Possession Phases"
        # Set the positions of the bars on the x-axis
        away_r = np.arange(len(in_pos_labels))
        home_r = [x + bar_width for x in away_r]

        # Invert the order of labels and values
        in_pos_labels = in_pos_labels[::-1]
        home_in_phases = home_in_phases[::-1]
        away_in_phases = away_in_phases[::-1]

        #away team bar
        ax1.barh(away_r, away_in_phases, color='deepskyblue', height=bar_width, label = away_team_name)

        #home team bar
        ax1.barh(home_r, home_in_phases, color='crimson', height=bar_width, label = home_team_name)
        

        # Customize the plot
        legend_size = 16
        label_size = 25
        coord_size = 16
        ax1.set_yticks(np.arange(len(in_pos_labels)) + bar_width / 2)
        ax1.set_yticklabels(in_pos_labels)
        ax1.set_xlabel(x_label, fontsize = label_size)
        ax1.set_ylabel(y_label, fontsize = label_size)
        ax1.legend(fontsize = legend_size)
        # Increase font size of values on x-axis
        ax1.tick_params(axis='x', labelsize=coord_size)

        # Increase font size of values on y-axis
        ax1.tick_params(axis='y', labelsize=coord_size)

        # Out-of-Possession Phases
        x_label = "Phase Distributions (%)"
        y_label = "Out-of-Possession Phases"
        # Set the positions of the bars on the x-axis
        away_r = np.arange(len(out_pos_labels))
        home_r = [x + bar_width for x in away_r]

        # Invert the order of labels and values
        out_pos_labels = out_pos_labels[::-1]
        home_out_phases = home_out_phases[::-1]
        away_out_phases = away_out_phases[::-1]

        #away team bar
        ax2.barh(away_r, away_out_phases, color='deepskyblue', height=bar_width, label = away_team_name)

         #home team bar
        ax2.barh(home_r, home_out_phases, color='crimson', height=bar_width, label = home_team_name)
        
        # Customize the plot
        ax2.set_yticks(np.arange(len(out_pos_labels)) + bar_width / 2)
        ax2.set_yticklabels(out_pos_labels)
        ax2.set_xlabel(x_label, fontsize = label_size)
        ax2.set_ylabel(y_label, fontsize = label_size)
        ax2.yaxis.set_label_coords(-0.175, 0.5)
        ax2.legend(fontsize = legend_size)
        # Increase font size of values on x-axis
        ax2.tick_params(axis='x', labelsize=coord_size)

        # Increase font size of values on y-axis
        ax2.tick_params(axis='y', labelsize=coord_size)

    def normalize_values(self, values, min_val, max_val, desired_range=(0.3, 1.5)):
        """Helper function to normalize the given set of values.
        Parameters
        ----------
        values: list 
            values to be normalized.
        min_val: float
            minimum value in the initial set of values.
        max_val: float
            maximum value in the initial set of values.
        desired_range: tuple, optional
            minimum and maximum values to have in the normalizes set of values.
        
        Return
        ----------
        scaled values: list
            scaled values.
        """
        normalized_values = (values - min_val) / (max_val - min_val)
        scaled_values = (desired_range[1] - desired_range[0]) * normalized_values + desired_range[0]
        return scaled_values

    def final_third_entries_plotter(self, match, home_fte, away_fte):
        """Visualize final third entries results on a football pitch.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        home_fte: numpy array
            home team final third entry counts.
        away_fte: numpay array
            away team final third entry counts.
        """
        home_team_name = match.meta.home_team_name[0]
        away_team_name = match.meta.away_team_name[0]
        fig = plt.figure(figsize=(7.8, 9))
        ax = plt.gca()
        title_size = 25
        weight = "bold"
        title_color = "black"
        ax.set_title(f"Final Third Entries: \n {home_team_name} - {away_team_name}" , fontsize= title_size, weight = weight, color = title_color)
        self.draw_pitch(ax, orient_vertical=True, pitch_center=(0, 0))  
        ax.set_ylim(-20,56)
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        x_left_end = -33.75
        x_left_pa = -15.25
        x_left_ga = -7
        x_right_end = 34
        x_right_pa = 15.25
        x_right_ga = 7
        y_end = 17.45

        line_width = 2 
        edge_color = None
        facecolor_end = "#fff7bc"
        facecolor_in = "#dfc27d"
        facecolor_mid = "#a6611a"
        alpha = 0.6

        # Final Third Channel Drawing

        #Left Channel
        left_ch = patches.Rectangle((x_left_end, y_end), x_left_pa -  x_left_end, 2 * y_end, linewidth= line_width, edgecolor= edge_color, facecolor= facecolor_end, alpha= alpha)
        ax.add_patch(left_ch)
        #Left In Channel
        left_in_ch = patches.Rectangle((x_left_pa, y_end), x_left_ga -  x_left_pa, 2 * y_end, linewidth= line_width, edgecolor= edge_color, facecolor= facecolor_in, alpha= alpha)
        ax.add_patch(left_in_ch)
        #Mid Channel
        mid_ch = patches.Rectangle((x_left_ga, y_end), x_right_ga -  x_left_ga, 2 * y_end, linewidth= line_width, edgecolor= edge_color, facecolor= facecolor_mid, alpha= alpha)
        ax.add_patch(mid_ch)
        #Right In Channel
        right_in_ch = patches.Rectangle((x_right_ga, y_end), x_right_pa -  x_right_ga, 2 * y_end, linewidth= line_width, edgecolor= edge_color, facecolor= facecolor_in, alpha= alpha)
        ax.add_patch(right_in_ch)
        #Right Channel
        right_ch = patches.Rectangle((x_right_pa, y_end), x_right_end -  x_right_pa, 2 * y_end, linewidth= line_width, edgecolor= edge_color, facecolor= facecolor_end, alpha= alpha)
        ax.add_patch(right_ch)

        # Final Third Arrow Drawing
        home_color_arr = "crimson"
        away_color_arr = "deepskyblue"
        y_start = -2
        y_end = 17.5
        x_left_arr_home = -30
        x_left_arr_away = -25
        x_left_in_arr_home = -13
        x_left_in_arr_away = -9
        x_mid_arr_home = -2
        x_mid_arr_away = 2
        x_right_in_arr_home = 9.5
        x_right_in_arr_away = 13.5
        x_right_arr_home = 26
        x_right_arr_away = 31
        
        weight='bold'
        fontsize= 17
        x_offset = 2
        y_offset = 3.5

        width_arr_min = 0.3
        width_arr_max = 1.3

        home_away_fte = np.concatenate([home_fte, away_fte])
        min_fte = np.min(home_away_fte)
        max_fte = np.max(home_away_fte)
        home_fte_widths = self.normalize_values(home_fte, min_fte, max_fte, desired_range=(width_arr_min, width_arr_max))
        away_fte_widths = self.normalize_values(away_fte, min_fte, max_fte, desired_range=(width_arr_min, width_arr_max))

        point_size = 30

        #Left Channel
        #Home
        home_left_arrow_start = [x_left_arr_home, y_start]
        home_left_arrow_end = [x_left_arr_home, y_end]

        ax.arrow(home_left_arrow_start[0], home_left_arrow_start[1], home_left_arrow_end[0] - home_left_arrow_start[0], home_left_arrow_end[1] - home_left_arrow_start[1], 
                  color= home_color_arr, width= home_fte_widths[0], length_includes_head=True)
        
        text = str(home_fte[0])
        ax.annotate(text, xy=(home_left_arrow_start[0] - x_offset, home_left_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)
        
        #Away
        away_left_arrow_start = [x_left_arr_away, y_start]
        away_left_arrow_end = [x_left_arr_away, y_end]
        
        ax.arrow(away_left_arrow_start[0], away_left_arrow_start[1], away_left_arrow_end[0] - away_left_arrow_start[0], away_left_arrow_end[1] - away_left_arrow_start[1], 
                  color= away_color_arr, width= away_fte_widths[0], length_includes_head=True)
    
        text = str(away_fte[0])
        ax.annotate(text, xy=(away_left_arrow_start[0]- x_offset, away_left_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)

        #Left In Channel
        #Home
        home_left_in_arrow_start = [x_left_in_arr_home, y_start]
        home_left_in_arrow_end = [x_left_in_arr_home, y_end]

        ax.arrow(home_left_in_arrow_start[0], home_left_in_arrow_start[1], home_left_in_arrow_end[0] - home_left_in_arrow_start[0], home_left_in_arrow_end[1] - home_left_in_arrow_start[1], 
                  color= home_color_arr, width= home_fte_widths[1], length_includes_head=True)

        text = str(home_fte[1])
        ax.annotate(text, xy=(home_left_in_arrow_start[0]- x_offset, home_left_in_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)

        
        #Away
        away_left_in_arrow_start = [x_left_in_arr_away, y_start]
        away_left_in_arrow_end = [x_left_in_arr_away, y_end]
        
        ax.arrow(away_left_in_arrow_start[0], away_left_in_arrow_start[1], away_left_in_arrow_end[0] - away_left_in_arrow_start[0], away_left_in_arrow_end[1] - away_left_in_arrow_start[1], 
                  color= away_color_arr, width= away_fte_widths[1], length_includes_head=True)
        
        text = str(away_fte[1])
        ax.annotate(text, xy=(away_left_in_arrow_start[0]- x_offset, away_left_in_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)


        #Mid Channel
        #Home
        home_mid_arrow_start = [x_mid_arr_home, y_start]
        home_mid_arrow_end = [x_mid_arr_home, y_end]

        ax.arrow(home_mid_arrow_start[0], home_mid_arrow_start[1], home_mid_arrow_end[0] - home_mid_arrow_start[0], home_mid_arrow_end[1] - home_mid_arrow_start[1], 
                  color= home_color_arr, width= home_fte_widths[2], length_includes_head=True)
        
        text = str(home_fte[2])
        ax.annotate(text, xy=(home_mid_arrow_start[0]- x_offset, home_mid_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)
        

        #Away
        away_mid_arrow_start = [x_mid_arr_away, y_start]
        away_mid_arrow_end = [x_mid_arr_away, y_end]
        
        ax.arrow(away_mid_arrow_start[0], away_mid_arrow_start[1], away_mid_arrow_end[0] - away_mid_arrow_start[0], away_mid_arrow_end[1] - away_mid_arrow_start[1], 
                  color= away_color_arr, width= away_fte_widths[2], length_includes_head=True)
        
        text = str(away_fte[2])
        ax.annotate(text, xy=(away_mid_arrow_start[0]- x_offset, away_mid_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)
        
        
        #Right In Channel
        #Home
        home_right_in_arrow_start = [x_right_in_arr_home, y_start]
        home_right_in_arrow_end = [x_right_in_arr_home, y_end]

        ax.arrow(home_right_in_arrow_start[0], home_right_in_arrow_start[1], home_right_in_arrow_end[0] - home_right_in_arrow_start[0], home_right_in_arrow_end[1] - home_right_in_arrow_start[1], 
                  color= home_color_arr, width= home_fte_widths[3], length_includes_head=True)
        
        text = str(home_fte[3])
        ax.annotate(text, xy=(home_right_in_arrow_start[0]- x_offset, home_right_in_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)
        

        #Away
        away_right_in_arrow_start = [x_right_in_arr_away, y_start]
        away_right_in_arrow_end = [x_right_in_arr_away, y_end]
        
        ax.arrow(away_right_in_arrow_start[0], away_right_in_arrow_start[1], away_right_in_arrow_end[0] - away_right_in_arrow_start[0], away_right_in_arrow_end[1] - away_right_in_arrow_start[1], 
                  color= away_color_arr, width= away_fte_widths[3], length_includes_head=True)

        text = str(away_fte[3])
        ax.annotate(text, xy=(away_right_in_arrow_start[0]- x_offset, away_right_in_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)


        #Right Channel
        #Home
        home_right_arrow_start = [x_right_arr_home, y_start]
        home_right_arrow_end = [x_right_arr_home, y_end]

        ax.arrow(home_right_arrow_start[0], home_right_arrow_start[1], home_right_arrow_end[0] - home_right_arrow_start[0], home_right_arrow_end[1] - home_right_arrow_start[1], 
                  color= home_color_arr, width= home_fte_widths[4], length_includes_head=True)
        
        text = str(home_fte[4])
        ax.annotate(text, xy=(home_right_arrow_start[0]- x_offset, home_right_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)


        #Away
        away_right_arrow_start = [x_right_arr_away, y_start]
        away_right_arrow_end = [x_right_arr_away, y_end]
        
        ax.arrow(away_right_arrow_start[0], away_right_arrow_start[1], away_right_arrow_end[0] - away_right_arrow_start[0], away_right_arrow_end[1] - away_right_arrow_start[1], 
                  color= away_color_arr, width= away_fte_widths[4], length_includes_head=True)
        
        text = str(away_fte[4])
        ax.annotate(text, xy=(away_right_arrow_start[0]- x_offset, away_right_arrow_start[1] - y_offset), xytext=(10, 10),
             textcoords='offset points', ha='center', va='center', color='black', weight=weight, fontsize=fontsize)
                
        custom_handles = [plt.Line2D([], [], marker='s', color= home_color_arr, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= away_color_arr, linestyle='None')]
        custom_labels = [home_team_name, away_team_name]
        legend_size = 16
        ax.legend(custom_handles, custom_labels, loc='lower right', fontsize = legend_size)

    def line_breaks_plotter(self, match, lb_pass_events, lb_prog_events):
        """ Visualize line breaks results on football pitch.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        lb_pass_events: pandas data frame
            line break events via pass.
        lb_prog_events: pandas data frame
            line break events via ball progession.
        """
         # Team IDs
        home_team_id = match.meta.home_team_id[0]
        away_team_id = match.meta.away_team_id[0]
        home_team_name =  match.meta.home_team_name[0]
        away_team_name = match.meta.away_team_name[0]

        #Pass and Cross Line Breaks
        lb_pass_events = lb_pass_events[lb_pass_events.is_line_break == 1]
        warnings.filterwarnings('ignore')
        #adjust directions
        mask = lb_pass_events["y_location_end_tracking"] < lb_pass_events["y_location_start_tracking"]
        lb_pass_events.loc[mask, "x_location_start_tracking"] *= -1
        lb_pass_events.loc[mask, "x_location_end_tracking"] *= -1
        lb_pass_events.loc[mask, "y_location_start_tracking"] *= -1
        lb_pass_events.loc[mask, "y_location_end_tracking"] *= -1
        #split teams
        home_lb_pass_events = lb_pass_events[lb_pass_events.team_id == home_team_id]
        away_lb_pass_events = lb_pass_events[lb_pass_events.team_id == away_team_id]
        #split through, around and over
        home_lb_through_events = home_lb_pass_events[home_lb_pass_events.direction == 0]
        home_lb_around_events = home_lb_pass_events[home_lb_pass_events.direction == 1]
        home_lb_over_events = home_lb_pass_events[home_lb_pass_events.direction == 2]
        away_lb_through_events = away_lb_pass_events[away_lb_pass_events.direction == 0]
        away_lb_around_events = away_lb_pass_events[away_lb_pass_events.direction == 1]
        away_lb_over_events = away_lb_pass_events[away_lb_pass_events.direction == 2]

        "split pass and cross"
        home_lb_cross_events = home_lb_pass_events[home_lb_pass_events.event == "cross"]
        home_lb_pass_events = home_lb_pass_events[home_lb_pass_events.event == "pass"]
        away_lb_cross_events = away_lb_pass_events[away_lb_pass_events.event == "cross"]
        away_lb_pass_events = away_lb_pass_events[away_lb_pass_events.event == "pass"]

        #put into arrays
        # through
        home_through_start_x = home_lb_through_events["x_location_start_tracking"].values
        home_through_start_y = home_lb_through_events["y_location_start_tracking"].values
        home_through_end_x = home_lb_through_events["x_location_end_tracking"].values
        home_through_end_y = home_lb_through_events["y_location_end_tracking"].values

        away_through_start_x = away_lb_through_events["x_location_start_tracking"].values
        away_through_start_y = away_lb_through_events["y_location_start_tracking"].values
        away_through_end_x = away_lb_through_events["x_location_end_tracking"].values
        away_through_end_y = away_lb_through_events["y_location_end_tracking"].values

        # around
        home_around_start_x = home_lb_around_events["x_location_start_tracking"].values
        home_around_start_y = home_lb_around_events["y_location_start_tracking"].values
        home_around_end_x = home_lb_around_events["x_location_end_tracking"].values
        home_around_end_y = home_lb_around_events["y_location_end_tracking"].values

        away_around_start_x = away_lb_around_events["x_location_start_tracking"].values
        away_around_start_y = away_lb_around_events["y_location_start_tracking"].values
        away_around_end_x = away_lb_around_events["x_location_end_tracking"].values
        away_around_end_y = away_lb_around_events["y_location_end_tracking"].values

        #over
        home_over_start_x = home_lb_over_events["x_location_start_tracking"].values
        home_over_start_y = home_lb_over_events["y_location_start_tracking"].values
        home_over_end_x = home_lb_over_events["x_location_end_tracking"].values
        home_over_end_y = home_lb_over_events["y_location_end_tracking"].values

        away_over_start_x = away_lb_over_events["x_location_start_tracking"].values
        away_over_start_y = away_lb_over_events["y_location_start_tracking"].values
        away_over_end_x = away_lb_over_events["x_location_end_tracking"].values
        away_over_end_y = away_lb_over_events["y_location_end_tracking"].values

        #pass
        home_pass_start_x = home_lb_pass_events["x_location_start_tracking"].values
        home_pass_start_y = home_lb_pass_events["y_location_start_tracking"].values
        home_pass_end_x = home_lb_pass_events["x_location_end_tracking"].values
        home_pass_end_y = home_lb_pass_events["y_location_end_tracking"].values

        away_pass_start_x = away_lb_pass_events["x_location_start_tracking"].values
        away_pass_start_y = away_lb_pass_events["y_location_start_tracking"].values
        away_pass_end_x = away_lb_pass_events["x_location_end_tracking"].values
        away_pass_end_y = away_lb_pass_events["y_location_end_tracking"].values

        #cross
        home_cross_start_x = home_lb_cross_events["x_location_start_tracking"].values
        home_cross_start_y = home_lb_cross_events["y_location_start_tracking"].values
        home_cross_end_x = home_lb_cross_events["x_location_end_tracking"].values
        home_cross_end_y = home_lb_cross_events["y_location_end_tracking"].values
        
        away_cross_start_x = away_lb_cross_events["x_location_start_tracking"].values
        away_cross_start_y = away_lb_cross_events["y_location_start_tracking"].values
        away_cross_end_x = away_lb_cross_events["x_location_end_tracking"].values
        away_cross_end_y = away_lb_cross_events["y_location_end_tracking"].values

        

        #Progression Line Breaks
        lb_prog_events = lb_prog_events[lb_prog_events.is_line_break == 1]
        #adjust directions
        mask = lb_prog_events["y_location_end_tracking"] < lb_prog_events["y_location_start_tracking"]
        lb_prog_events.loc[mask, "x_location_start_tracking"] *= -1
        lb_prog_events.loc[mask, "x_location_end_tracking"] *= -1
        lb_prog_events.loc[mask, "y_location_start_tracking"] *= -1
        lb_prog_events.loc[mask, "y_location_end_tracking"] *= -1
        #split teams
        home_lb_prog_events = lb_prog_events[lb_prog_events.team_id == home_team_id]
        away_lb_prog_events = lb_prog_events[lb_prog_events.team_id == away_team_id]
        #split through and around
        home_prog_lb_through_events = home_lb_prog_events[home_lb_prog_events.direction == 0]
        home_prog_lb_around_events = home_lb_prog_events[home_lb_prog_events.direction == 1]
        away_prog_lb_through_events = away_lb_prog_events[away_lb_prog_events.direction == 0]
        away_prog_lb_around_events = away_lb_prog_events[away_lb_prog_events.direction == 1]
        #put into arrays

        # through
        home_prog_through_start_x = home_prog_lb_through_events["x_location_start_tracking"].values
        home_prog_through_start_y = home_prog_lb_through_events["y_location_start_tracking"].values
        home_prog_through_end_x = home_prog_lb_through_events["x_location_end_tracking"].values
        home_prog_through_end_y = home_prog_lb_through_events["y_location_end_tracking"].values

        away_prog_through_start_x = away_prog_lb_through_events["x_location_start_tracking"].values
        away_prog_through_start_y = away_prog_lb_through_events["y_location_start_tracking"].values
        away_prog_through_end_x = away_prog_lb_through_events["x_location_end_tracking"].values
        away_prog_through_end_y = away_prog_lb_through_events["y_location_end_tracking"].values

        # around
        home_prog_around_start_x = home_prog_lb_around_events["x_location_start_tracking"].values
        home_prog_around_start_y = home_prog_lb_around_events["y_location_start_tracking"].values
        home_prog_around_end_x = home_prog_lb_around_events["x_location_end_tracking"].values
        home_prog_around_end_y = home_prog_lb_around_events["y_location_end_tracking"].values

        away_prog_around_start_x = away_prog_lb_around_events["x_location_start_tracking"].values
        away_prog_around_start_y = away_prog_lb_around_events["y_location_start_tracking"].values
        away_prog_around_end_x = away_prog_lb_around_events["x_location_end_tracking"].values
        away_prog_around_end_y = away_prog_lb_around_events["y_location_end_tracking"].values

        # progression
        home_prog_start_x = home_lb_prog_events["x_location_start_tracking"].values
        home_prog_start_y = home_lb_prog_events["y_location_start_tracking"].values
        home_prog_end_x = home_lb_prog_events["x_location_end_tracking"].values
        home_prog_end_y = home_lb_prog_events["y_location_end_tracking"].values

        away_prog_start_x = away_lb_prog_events["x_location_start_tracking"].values
        away_prog_start_y = away_lb_prog_events["y_location_start_tracking"].values
        away_prog_end_x = away_lb_prog_events["x_location_end_tracking"].values
        away_prog_end_y = away_lb_prog_events["y_location_end_tracking"].values

        
        coef = 1.1
        fig, axs = plt.subplots(figsize = (6.8 * 2, 10.5 * 2 * coef), ncols=2, nrows=2)
        title_size = 25
        weight = "bold"
        title_color = "black"
        fig.suptitle(f"Line Breaks: \n {home_team_name} ({str(len(home_pass_end_x) + len(home_cross_end_x) + len(home_prog_end_x))}) - {away_team_name} ({str(len(away_pass_end_x) + len(away_cross_end_x) + len(away_prog_end_x))})" ,
                      fontsize= title_size, color = title_color, weight = weight)
        fig.subplots_adjust(wspace=0.06, hspace= 0.08, top=0.94)
        
        ax1 = axs[0][0]
        ax1_1 = axs[0][1]
        ax2 = axs[1][0]
        ax2_1 = axs[1][1]
        
        width_arr = 0.30
        
        pass_color_home =  "red" 
        cross_color_home = "peachpuff"
        prog_color_home = "maroon"

        pass_color_away = "blue"
        cross_color_away = "powderblue"
        prog_color_away = "midnightblue"

        through_color_home = "maroon"
        around_color_home = "peachpuff"
        over_color_home = "red"

        through_color_away = "midnightblue"
        around_color_away = "powderblue"
        over_color_away= "blue"



        title_fontsize = 15
        alpha = 0.8

        custom_handles_home = [plt.Line2D([], [], marker='s', color= prog_color_home, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= pass_color_home, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= cross_color_home, linestyle='None')]
    
        custom_handles_away = [plt.Line2D([], [], marker='s', color= prog_color_away, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= pass_color_away, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= cross_color_away, linestyle='None')]
        
        custom_handles_dir_home = [plt.Line2D([], [], marker='s', color= through_color_home, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= over_color_home, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= around_color_home, linestyle='None')]

        custom_handles_dir_away = [plt.Line2D([], [], marker='s', color= through_color_away, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= over_color_away, linestyle='None'),
                          plt.Line2D([], [], marker='s', color= around_color_away, linestyle='None')]

        
        
        custom_labels = ['Ball Progression', 'Pass', 'Cross']
        custom_labels_dir = ['Through', 'Over', 'Around']
        
        #Home Plot
        self.draw_pitch(ax1, orient_vertical=True, pitch_center=(0, 0))  
        ax1.set_facecolor('white')
        ax1.set_xticks([])
        ax1.set_yticks([])

        #Pass
        for i in range(len(home_pass_start_x)):
            arrow_start = [home_pass_start_x[i], home_pass_start_y[i]]
            arrow_end = [home_pass_end_x[i], home_pass_end_y[i]]
        
            ax1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= pass_color_home, width= width_arr, length_includes_head=True, alpha = alpha)
            
        #Cross
        for i in range(len(home_cross_start_x)):
            arrow_start = [home_cross_start_x[i], home_cross_start_y[i]]
            arrow_end = [home_cross_end_x[i], home_cross_end_y[i]]
        
            ax1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= cross_color_home, width= width_arr, length_includes_head=True, alpha = alpha)
        #Progression
        for i in range(len(home_prog_start_x)):
            arrow_start = [home_prog_start_x[i], home_prog_start_y[i]]
            arrow_end = [home_prog_end_x[i], home_prog_end_y[i]]
        
            ax1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= prog_color_home, width= width_arr, length_includes_head=True, alpha = alpha)
        
        ax1.text(-30, 54 , s = match.meta.home_team_name.values[0] + " Line Breaks - Distribution Type", color = "black", fontsize = title_fontsize)
        ax1.legend(custom_handles_home, custom_labels, loc='lower right', fontsize = title_fontsize)
        

        self.draw_pitch(ax1_1, orient_vertical=True, pitch_center=(0, 0))  
        ax1_1.set_facecolor('white')
        ax1_1.set_xticks([])
        ax1_1.set_yticks([])
        #Through
        for i in range(len(home_through_start_x)):
            arrow_start = [home_through_start_x[i], home_through_start_y[i]]
            arrow_end = [home_through_end_x[i], home_through_end_y[i]]
            ax1_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= through_color_home, width= width_arr, length_includes_head=True, alpha = alpha)
            
        for i in range(len(home_prog_through_start_x)):
            arrow_start = [home_prog_through_start_x[i], home_prog_through_start_y[i]]
            arrow_end = [home_prog_through_end_x[i], home_prog_through_end_y[i]]
            ax1_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= through_color_home, width= width_arr, length_includes_head=True, alpha = alpha)
        #Around
        for i in range(len(home_around_start_x)):
            arrow_start = [home_around_start_x[i], home_around_start_y[i]]
            arrow_end = [home_around_end_x[i], home_around_end_y[i]]
            ax1_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= around_color_home, width= width_arr, length_includes_head=True, alpha = alpha)
            
        for i in range(len(home_prog_around_start_x)):
            arrow_start = [home_prog_around_start_x[i], home_prog_around_start_y[i]]
            arrow_end = [home_prog_around_end_x[i], home_prog_around_end_y[i]]
            ax1_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= around_color_home, width= width_arr, length_includes_head=True, alpha = alpha)
        #Over
        for i in range(len(home_over_start_x)):
            arrow_start = [home_over_start_x[i], home_over_start_y[i]]
            arrow_end = [home_over_end_x[i], home_over_end_y[i]]
            ax1_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= over_color_home, width= width_arr, length_includes_head=True, alpha = alpha)

        ax1_1.text(-30, 54 , s = match.meta.home_team_name.values[0] + " Line Breaks - Direction Type", color = "black", fontsize = title_fontsize)
        ax1_1.legend(custom_handles_dir_home, custom_labels_dir, loc='lower right', fontsize = title_fontsize)
        #Away Plot
        self.draw_pitch(ax2, orient_vertical=True, pitch_center=(0, 0))  
        ax2.set_facecolor('white')
        ax2.set_xticks([])
        ax2.set_yticks([])

        #Pass
        for i in range(len(away_pass_start_x)):
            arrow_start = [away_pass_start_x[i], away_pass_start_y[i]]
            arrow_end = [away_pass_end_x[i], away_pass_end_y[i]]
        
            ax2.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= pass_color_away, width= width_arr, length_includes_head=True, alpha = alpha)
        #Cross
        for i in range(len(away_cross_start_x)):
            arrow_start = [away_cross_start_x[i], away_cross_start_y[i]]
            arrow_end = [away_cross_end_x[i], away_cross_end_y[i]]
        
            ax2.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= cross_color_away, width= width_arr, length_includes_head=True, alpha = alpha)
        #Progression
        for i in range(len(away_prog_start_x)):
            arrow_start = [away_prog_start_x[i], away_prog_start_y[i]]
            arrow_end = [away_prog_end_x[i], away_prog_end_y[i]]
        
            ax2.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= prog_color_away, width= width_arr, length_includes_head=True, alpha = alpha)
        
        ax2.text(-30, 54 , s = match.meta.away_team_name.values[0] + " Line Breaks - Distribution Type", color = "black", fontsize = title_fontsize)
        ax2.legend(custom_handles_away, custom_labels, loc='lower right', fontsize = title_fontsize)


        self.draw_pitch(ax2_1, orient_vertical=True, pitch_center=(0, 0))  
        ax2_1.set_facecolor('white')
        ax2_1.set_xticks([])
        ax2_1.set_yticks([])
        #Through
        for i in range(len(away_through_start_x)):
            arrow_start = [away_through_start_x[i], away_through_start_y[i]]
            arrow_end = [away_through_end_x[i], away_through_end_y[i]]
            ax2_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= through_color_away, width= width_arr, length_includes_head=True, alpha = alpha)
            
        for i in range(len(away_prog_through_start_x)):
            arrow_start = [away_prog_through_start_x[i], away_prog_through_start_y[i]]
            arrow_end = [away_prog_through_end_x[i], away_prog_through_end_y[i]]
            ax2_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= through_color_away, width= width_arr, length_includes_head=True, alpha = alpha)
        #Around
        for i in range(len(away_around_start_x)):
            arrow_start = [away_around_start_x[i], away_around_start_y[i]]
            arrow_end = [away_around_end_x[i], away_around_end_y[i]]
            ax2_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= around_color_away, width= width_arr, length_includes_head=True, alpha = alpha)
            
        for i in range(len(away_prog_around_start_x)):
            arrow_start = [away_prog_around_start_x[i], away_prog_around_start_y[i]]
            arrow_end = [away_prog_around_end_x[i], away_prog_around_end_y[i]]
            ax2_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= around_color_away, width= width_arr, length_includes_head=True, alpha = alpha)
        #Over
        for i in range(len(away_over_start_x)):
            arrow_start = [away_over_start_x[i], away_over_start_y[i]]
            arrow_end = [away_over_end_x[i], away_over_end_y[i]]
            ax2_1.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                    color= over_color_away, width= width_arr, length_includes_head=True, alpha = alpha)
            
        ax2_1.text(-30, 54 , s = match.meta.away_team_name.values[0] + " Line Breaks - Direction Type", color = "black", fontsize = title_fontsize)
        ax2_1.legend(custom_handles_dir_away, custom_labels_dir, loc='lower right', fontsize = title_fontsize)

    def reception_plotter(self, match, recep_events):
        """ Visualize reception behind midfield and defensive line results on football pitch.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        recep_events: pandas data frame
            reception behind midfield and defensive line events.
        """
        home_team_id = match.meta.home_team_id[0]
        away_team_id = match.meta.away_team_id[0]
        home_team_name = match.meta.home_team_name[0]
        away_team_name = match.meta.away_team_name[0]

        recep_events = recep_events[recep_events.is_reception == 1]
        mask = recep_events["need_mirror"] == 1
        recep_events.loc[mask, "x_tracking"] *= -1
        recep_events.loc[mask, "y_tracking"] *= -1

        #split by team
        home_recep_events = recep_events[recep_events.team_id == home_team_id]
        away_recep_events = recep_events[recep_events.team_id == away_team_id]

        #split by reception between midfield and defensive unit - behind defensive unit
        home_between_mid_def_recep_events = home_recep_events[home_recep_events.reception_unit == 0]
        home_behind_def_recep_events = home_recep_events[home_recep_events.reception_unit == 1]
        away_between_mid_def_recep_events = away_recep_events[away_recep_events.reception_unit == 0]
        away_behind_def_recep_events = away_recep_events[away_recep_events.reception_unit == 1]

        #Get locations
        home_between_mid_def_recep_events_x = home_between_mid_def_recep_events["x_tracking"].values
        home_between_mid_def_recep_events_y = home_between_mid_def_recep_events["y_tracking"].values
        away_between_mid_def_recep_events_x = away_between_mid_def_recep_events["x_tracking"].values
        away_between_mid_def_recep_events_y = away_between_mid_def_recep_events["y_tracking"].values

        home_behind_def_recep_events_x = home_behind_def_recep_events["x_tracking"].values
        home_behind_def_recep_events_y = home_behind_def_recep_events["y_tracking"].values
        away_behind_def_recep_events_x = away_behind_def_recep_events["x_tracking"].values
        away_behind_def_recep_events_y = away_behind_def_recep_events["y_tracking"].values

        warnings.filterwarnings('ignore')
        coef = 1
        fig, axs = plt.subplots(figsize = (6.8 * 2, 10.5 * 1 * coef), ncols=2, nrows=1)
        title_size = 20
        weight = "bold"
        title_color = "black"
        fig.suptitle(f"Receptions Behind Midfield and Defensive Lines: \n {home_team_name} - {away_team_name}" , fontsize= title_size, weight = weight, color = title_color)
        fig.subplots_adjust(wspace=0.06, hspace= 0.08, top=0.9)
        ax1 = axs[0]
        ax2 = axs[1]


        between_mid_def_color_home = "pink"
        behind_def_color_home = "red" 
        between_mid_def_color_away = "lightskyblue"
        behind_def_color_away = "blue" 
        s = 80
        marker = "o"
        title_fontsize = 14
        legend_fontsize = 11

        custom_handles_home = [plt.Line2D([], [], marker= marker, color= between_mid_def_color_home, linestyle='None'),
                          plt.Line2D([], [], marker= marker, color= behind_def_color_home, linestyle='None')]
        custom_handles_away = [plt.Line2D([], [], marker= marker, color= between_mid_def_color_away, linestyle='None'),
                          plt.Line2D([], [], marker= marker, color= behind_def_color_away, linestyle='None')]
        custom_labels= ["Between Midfield and Defensive Lines", "Behind Defensive Line"]

        self.draw_pitch(ax1, orient_vertical=True, pitch_center=(0, 0))  
        ax1.set_facecolor('white')
        ax1.set_xticks([])
        ax1.set_yticks([])

        #Home 
        #Between midfield and defensive unit
        ax1.scatter([home_between_mid_def_recep_events_x], [home_between_mid_def_recep_events_y], color =  between_mid_def_color_home, s= s, marker = marker)
        #Behind defensive unit
        ax1.scatter([home_behind_def_recep_events_x], [home_behind_def_recep_events_y], color =  behind_def_color_home, s= s, marker = marker)

        ax1.text(-30, 54 , s = match.meta.home_team_name.values[0] + " Receptions - (" + str(len(home_between_mid_def_recep_events_x)) + ", " + str(len(home_behind_def_recep_events_x)) + ")", color = "black", fontsize = title_fontsize)
        ax1.legend(custom_handles_home, custom_labels, loc='lower right', fontsize = legend_fontsize)


        self.draw_pitch(ax2, orient_vertical=True, pitch_center=(0, 0))  
        ax2.set_facecolor('white')
        ax2.set_xticks([])
        ax2.set_yticks([])

        #Away
        #Between midfield and defensive unit
        ax2.scatter([away_between_mid_def_recep_events_x], [away_between_mid_def_recep_events_y], color =  between_mid_def_color_away, s= s, marker = marker)
        #Behind defensive unit
        ax2.scatter([away_behind_def_recep_events_x], [away_behind_def_recep_events_y], color =  behind_def_color_away, s= s, marker = marker) 

        ax2.text(-30, 54 , s = match.meta.away_team_name.values[0] + " Receptions - (" + str(len(away_between_mid_def_recep_events_x)) + ", " + str(len(away_behind_def_recep_events_x)) + ")", color = "black", fontsize = title_fontsize)
        ax2.legend(custom_handles_away, custom_labels, loc='lower right', fontsize = legend_fontsize)

    def line_height_team_length_plotter(self,match, avg_home_defensive_line_heights, avg_home_offensive_line_heights, avg_away_defensive_line_heights, avg_away_offensive_line_heights, \
                                                    avg_home_defensive_team_lengths, avg_home_offensive_team_lengths, avg_away_defensive_team_lengths, avg_away_offensive_team_lengths, \
                                                    avg_home_defensive_team_widths, avg_home_offensive_team_widths, avg_away_defensive_team_widths, avg_away_offensive_team_widths ):

        """ Visualize line height team length results on football pitch.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        avg_home_defensive_line_heights: list
            average home team defensive line heights.
        avg_home_offensive_line_heights: list
            average home team offensive line heights.
        avg_away_defensive_line_heights: list
            average away team defensive line heights.
        avg_away_offensive_line_heights: list
            average away team offensive line heights.

        avg_home_defensive_team_lengths: list
            average home team defensive team lengths.
        avg_home_offensive_team_lengths: list
            average home team offensive team lengths.
        avg_away_defensive_team_lengths: list
            average away team defensive team lengths.
        avg_away_offensive_team_lengths: list
            average away team offensive team lengths.

        avg_home_defensive_team_widths: list
            average home team defensive team widths.
        avg_home_offensive_team_widths: list
            average home team offensive team widths.
        avg_away_defensive_team_widths: list
            average away team defensive team widths.
        avg_away_offensive_team_widths: list
            average away team offensive team widths.
        """
        home_team_name = match.meta.home_team_name[0]
        away_team_name = match.meta.away_team_name[0]
        coef = 2
        title_fontsize = 80
        fig, axs = plt.subplots(figsize = (7.5 * 6 * coef, 11.2 * 2 * coef), ncols=6, nrows=2)
        weight = "bold"
        title_color = "black"
        fig.suptitle(f"Line Height & Team Lenght: \n {home_team_name} - {away_team_name}" , fontsize= title_fontsize, color = title_color, weight = weight)
        fig.subplots_adjust(wspace=0.1, hspace= 0.15, top=0.9)
        pitch_width = 68
        x_left_end = -34
        x_right_end = 34
        y_down_end = -52.5
        rectangle_color_home = "crimson"
        rectangle_color_away = "deepskyblue"
        edge_color = "none"
        line_width = 2.5
        line_width_bar = 2.5
        x_start_offset = 1.75
        y_start_offset = 1.75
        x_text_offset = 7
        y_text_offset = 1
        bar_height = 1
        alpha = 0.3
        titles = [[home_team_name + " - In Possession - First Third", home_team_name + " - In Possession - Second Third", home_team_name + " - In Possession - Final Third", 
                   home_team_name + " - Out of Possession - First Third", home_team_name + " - Out of Possession - Second Third", home_team_name + " - Out of Possession - Final Third"],
                  [away_team_name + " - In Possession - First Third", away_team_name + " - In Possession - Second Third", away_team_name + " - In Possession - Final Third",
                   away_team_name + " - Out of Possession - First Third", away_team_name + " - Out of Possession - Second Third", away_team_name + " - Out of Possession - Final Third"]]

        weight = "bold"
        fontsize = 19
        measurement_fontsize = 32
        arrow_start = [32, 31]
        arrow_end = [32, 49]
        text_x = 30
        text_y = 40
        text = "DIRECTION"
        arrow_color = "gray"
        sub_title_fontsize = 30
        for i in range(2):
            for j in range(6):    
                ax = axs[i][j]
                self.draw_pitch(ax, orient_vertical=True, pitch_center=(0, 0))  
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                #Title
                ax.text(-33, 54 , s = titles[i][j], color = "black", fontsize = sub_title_fontsize)

                if i == 0 and (j == 0 or j == 1 or j == 2): #Home In-Possession
                    # Rectangle
                    x_start = x_left_end + (pitch_width - avg_home_offensive_team_widths[j]) / 2
                    y_start = y_down_end + avg_home_offensive_line_heights[j]
                    rect = patches.Rectangle((x_start, y_start), width = avg_home_offensive_team_widths[j], height = avg_home_offensive_team_lengths[j], linewidth= line_width, edgecolor= edge_color, facecolor= rectangle_color_home, alpha= alpha)
                    ax.add_patch(rect) 
                    
                    # Bar Arrow Team Length
                    x_start = x_start - x_start_offset
                    x_text = x_start - x_text_offset
                    y_end_bar = y_start + avg_home_offensive_team_lengths[j]
                    y_text = (y_start + y_end_bar) / 2
                    ax.vlines(x_start, y_start, y_end_bar, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_end_bar, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_start, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_home_offensive_team_lengths[j]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    # Bar Arrow Team Width
                    x_start = x_start + x_start_offset
                    x_end = x_start + avg_home_offensive_team_widths[j]
                    y_start = avg_home_offensive_line_heights[j] + avg_home_offensive_team_lengths[j] + y_down_end + y_start_offset
                    x_text = (x_start + x_end - x_start_offset - 1.5) / 2
                    y_text = y_start + y_text_offset
                    ax.hlines(y_start, x_start, x_end, linewidth= line_width_bar, color = arrow_color)
                    ax.vlines(x_start, y_start - bar_height, y_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.vlines(x_end, y_start - bar_height, y_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_home_offensive_team_widths[j]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    # Bar Arrow Line Height
                    x_start = x_right_end - (pitch_width - avg_home_offensive_team_widths[j]) / 2 + x_start_offset
                    y_start = y_down_end + avg_home_offensive_line_heights[j]
                    y_end_bar = y_down_end
                    x_text = x_start + x_start_offset
                    y_text = (y_start + y_end_bar) / 2
                    ax.vlines(x_start, y_start, y_end_bar, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_end_bar, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_start, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_home_offensive_line_heights[j]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    
                
                elif i == 0 and (j == 3 or j == 4 or j == 5): # Home Out-0f-Possession
                    # Rectangle
                    x_start = x_left_end + (pitch_width - avg_home_defensive_team_widths[j-3]) / 2
                    y_start = y_down_end + avg_home_defensive_line_heights[j-3]
                    rect = patches.Rectangle((x_start, y_start), width = avg_home_defensive_team_widths[j-3], height = avg_home_defensive_team_lengths[j-3], linewidth= line_width, edgecolor= edge_color, facecolor= rectangle_color_home, alpha= alpha)
                    ax.add_patch(rect)

                    # Bar Arrow Team Length
                    x_start = x_start - x_start_offset
                    x_text = x_start - x_text_offset
                    y_end_bar = y_start + avg_home_defensive_team_lengths[j-3]
                    y_text = (y_start + y_end_bar) / 2
                    ax.vlines(x_start, y_start, y_end_bar, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_end_bar, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_start, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_home_defensive_team_lengths[j-3]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    # Bar Arrow Team Width
                    x_start = x_start + x_start_offset
                    x_end = x_start + avg_home_defensive_team_widths[j-3]
                    y_start = avg_home_defensive_line_heights[j-3] + avg_home_defensive_team_lengths[j-3] + y_down_end + y_start_offset
                    x_text = (x_start + x_end - x_start_offset - 1.5) / 2
                    y_text = y_start + y_text_offset
                    ax.hlines(y_start, x_start, x_end, linewidth= line_width_bar, color = arrow_color)
                    ax.vlines(x_end, y_start - bar_height, y_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.vlines(x_start, y_start - bar_height, y_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_home_defensive_team_widths[j-3]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    # Bar Arrow Line Height
                    x_start = x_right_end - (pitch_width - avg_home_defensive_team_widths[j-3]) / 2 + x_start_offset
                    y_start = y_down_end + avg_home_defensive_line_heights[j-3]
                    y_end_bar = y_down_end
                    x_text = x_start + x_start_offset
                    y_text = (y_start + y_end_bar) / 2
                    ax.vlines(x_start, y_start, y_end_bar, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_end_bar, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_start, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_home_defensive_line_heights[j-3]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)
                
                elif i == 1 and (j == 0 or j == 1 or j == 2): # Away In-Possession
                    # Rectangle
                    x_start = x_left_end + (pitch_width - avg_away_offensive_team_widths[j]) / 2
                    y_start = y_down_end + avg_away_offensive_line_heights[j]
                    rect = patches.Rectangle((x_start, y_start), width = avg_away_offensive_team_widths[j], height = avg_away_offensive_team_lengths[j], linewidth= line_width, edgecolor= edge_color, facecolor= rectangle_color_away, alpha= alpha)
                    ax.add_patch(rect)

                    # Bar Arrow Team Length
                    x_start = x_start - x_start_offset
                    x_text = x_start - x_text_offset
                    y_end_bar = y_start + avg_away_offensive_team_lengths[j]
                    y_text = (y_start + y_end_bar) / 2
                    ax.vlines(x_start, y_start, y_end_bar, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_end_bar, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_start, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_away_offensive_team_lengths[j]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    # Bar Arrow Team Width
                    x_start = x_start + x_start_offset
                    x_end = x_start + avg_away_offensive_team_widths[j]
                    y_start = avg_away_offensive_line_heights[j] + avg_away_offensive_team_lengths[j] + y_down_end + y_start_offset
                    x_text = (x_start + x_end - x_start_offset - 1.5) / 2
                    y_text = y_start + y_text_offset
                    ax.hlines(y_start, x_start, x_end, linewidth= line_width_bar, color = arrow_color)
                    ax.vlines(x_start, y_start - bar_height, y_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.vlines(x_end, y_start - bar_height, y_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_away_offensive_team_widths[j]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    # Bar Arrow Line Height
                    x_start = x_right_end - (pitch_width - avg_away_offensive_team_widths[j]) / 2 + x_start_offset
                    y_start = y_down_end + avg_away_offensive_line_heights[j]
                    y_end_bar = y_down_end
                    x_text = x_start + x_start_offset
                    y_text = (y_start + y_end_bar) / 2
                    ax.vlines(x_start, y_start, y_end_bar, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_end_bar, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_start, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_away_offensive_line_heights[j]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                elif i == 1 and (j == 3 or j == 4 or j == 5): # Away Out-0f-Possession
                    x_start = x_left_end + (pitch_width - avg_away_defensive_team_widths[j-3]) / 2
                    y_start = y_down_end + avg_away_defensive_line_heights[j-3]
                    rect = patches.Rectangle((x_start, y_start), width = avg_away_defensive_team_widths[j-3], height = avg_away_defensive_team_lengths[j-3], linewidth= line_width, edgecolor= edge_color, facecolor= rectangle_color_away, alpha= alpha)
                    ax.add_patch(rect)

                    # Bar Arrow Team Length
                    x_start = x_start - x_start_offset
                    x_text = x_start - x_text_offset
                    y_end_bar = y_start + avg_away_defensive_team_lengths[j-3]
                    y_text = (y_start + y_end_bar) / 2
                    ax.vlines(x_start, y_start, y_end_bar, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_end_bar, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_start, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_away_defensive_team_lengths[j-3]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    # Bar Arrow Team Width
                    x_start = x_start + x_start_offset
                    x_end = x_start + avg_away_defensive_team_widths[j-3]
                    y_start = avg_away_defensive_line_heights[j-3] + avg_away_defensive_team_lengths[j-3] + y_down_end + y_start_offset
                    x_text = (x_start + x_end - x_start_offset - 1.5) / 2
                    y_text = y_start + y_text_offset
                    ax.hlines(y_start, x_start, x_end, linewidth= line_width_bar, color = arrow_color)
                    ax.vlines(x_start, y_start - bar_height, y_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.vlines(x_end, y_start - bar_height, y_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_away_defensive_team_widths[j-3]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    # Bar Arrow Line Height
                    x_start = x_right_end - (pitch_width - avg_away_defensive_team_widths[j-3]) / 2 + x_start_offset
                    y_start = y_down_end + avg_away_defensive_line_heights[j-3]
                    y_end_bar = y_down_end
                    x_text = x_start + x_start_offset
                    y_text = (y_start + y_end_bar) / 2
                    ax.vlines(x_start, y_start, y_end_bar, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_end_bar, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.hlines(y_start, x_start - bar_height, x_start + bar_height, linewidth= line_width_bar, color = arrow_color)
                    ax.text(x_text, y_text , s = str(avg_away_defensive_line_heights[j-3]) + "m", color = "black", fontsize = measurement_fontsize, weight = weight)

                    

                
                 # Add the arrow to the ax1
                ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], 
                        color= arrow_color, width=0.4, length_includes_head=True)
                
                # Add the direction text to the ax1
                ax.text(text_x, text_y, text, rotation=90, va='center', ha='center', color = arrow_color, fontsize = fontsize, weight = weight)
