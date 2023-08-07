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

class Event:
    def __init__(self, path):
        """ Initialize the event class parameters.
        Parameters
        ----------
        path : string
            path to event data.
        """
        try: 
            self.events =  self.__parse(path)
        except:
            print("Path Not Found!")
    def __parse(self, path):
        """ Parse the given match data.
        Parameters
        ----------
        path : string
            path to event data.
        
        Return
        ----------
        events : pandas data frame
            recorded events in the match.
        """

        event_dics = [] # A list of event dictionaries that we will fill in

        with open(path) as f: # Open the event json file to read it
            events = f.readlines() # Read all the lines at once
            #print(f"Number of recorded events in the match are {len(events)}.\n")
            for event in tqdm(events, disable=True): # Iterate over the events one by one and convert them into a dictionary (i.e., a data structure with keys and values)
                event_dic = json.loads(event)
                event_dics.append(event_dic)

        return pd.DataFrame(event_dics) # Transform the list of event dictionaries into a panda dataframe