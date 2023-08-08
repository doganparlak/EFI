import pandas as pd
import json
from tqdm import tqdm
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