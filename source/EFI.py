import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class EFI():
    def __init__(self, match, tracking, events):
        """ Initialize the EFI class parameters.
        Parameters
        ----------
        match : match class object
            object consisting general match information.
        tracking : tracking class object
            object consisting tracking data features.
        events: event class object
            object consisting recorded events.
        """
        self.match = match
        self.tracking = tracking
        self.events = events

    def possession_control(self):
        """ Possession control calculation.
        Return
        ----------
        home_frames_pct: float
            percent of home team in-possession.
        in_contest_pct: float
            percent of in-contest.
        away_frames_pct: float
            percent of away team in-possession.
        """
        home_frames = len(self.tracking.possession_info_detailed[self.tracking.possession_info_detailed == "Home"])
        away_frames = len(self.tracking.possession_info_detailed[self.tracking.possession_info_detailed == "Away"])
        in_contest_frames = len(self.tracking.possession_info_detailed[self.tracking.possession_info_detailed == "In_Contest"])
        total = home_frames + away_frames + in_contest_frames
        home_frames_pct = round((home_frames/total) * 100 , 1)
        away_frames_pct = round((away_frames/total) * 100 , 1)
        in_contest_pct = round((in_contest_frames/total) * 100 , 1)

        return home_frames_pct, in_contest_pct, away_frames_pct
    
    def ball_recovery_time(self, recovery_threshold = 24):
        """ Ball recovery time calculation.
        Parameters
        ----------
        recovery_threshold: integer, optional
            number of frames which ensures the possession is really lost.
        Return
        ----------
        home_avg_recovery_time: float
            average home team recovery time duration.
        away_avg_recovery_time: float
            average away team recovery time duration.
        """
        alive_frames = self.tracking.possession_info_raw[np.where(self.tracking.ball_state == "Alive")]
        self.tracking.ball_state
        prev_team = ""
        frame_cnt = 0
        home_recovery_time = []
        away_recovery_time = []
        for i,p in enumerate(alive_frames):
            if p == "H":
                if  prev_team == "A":
                    home_recovery_time.append(frame_cnt) # ball recovery home
                    frame_cnt = 1
                else:
                    frame_cnt += 1 # count possession for home
                prev_team = "H"
            elif p == "A":
                if prev_team == "H":
                    away_recovery_time.append(frame_cnt) #ball recovery away
                    frame_cnt = 1
                else:
                    frame_cnt += 1 # count possession for away
                prev_team = "A"

        home_recovery_time = np.array(home_recovery_time)
        away_recovery_time = np.array(away_recovery_time)
        home_recovery_time_filtered = home_recovery_time[home_recovery_time>recovery_threshold] # number of frames which ensures the possession is really lost
        away_recovery_time_filtered = away_recovery_time[away_recovery_time>recovery_threshold] # number of frames which ensures the possession is really lost
        home_avg_recovery_time = (sum(home_recovery_time_filtered) / len(home_recovery_time_filtered)) / 25 #average in seconds
        away_avg_recovery_time = (sum(away_recovery_time_filtered) / len(away_recovery_time_filtered)) / 25 #average in seconds

        return round(home_avg_recovery_time,2), round(away_avg_recovery_time,2)
    
    def pressure_on_ball(self, pressure_r0  = 3, pressure_r1 = 2.5, pressure_r2 = 2, pressure_r3 = 1, 
                         angle_h0 = 30, angle_h1 = 60 , angle_h2 = 90, pressure_threshold = 27, cont_pressure_threshold = 5):
        
        """ Pressure on the ball calculation.
        Parameters
        ----------
        pressure_r0: float, optional
            required radius for pressure to be detected within 0 to 30 degrees.
        pressure_r1: float, optional
            required radius for pressure to be detected within 30 to 60 degrees.
        pressure_r2: float, optional
            required radius for pressure to be detected within 60 to 90 degrees.
        pressure_r3: float, optional
            required radius for pressure to be detected with > 90 degrees.

        angle_h0: integer, optional
            angle defining the smallest section for pressure presence.
        angle_h1: integer, optional
            angle defining the middle section for pressure presence.
        angle_h2: integer, optional
            angle defining the largest section for pressure presence.

        pressure_threshold: integer, optional
            required number of frame gap between two pressure events.
        cont_pressure_threshold: integer, optional
            required pressure duration in frame level.

        Return
        ---------- 
        pressure_home_cnt: int
            home team pressure count.
        pressure_away_cnt: int
            away team pressure count.
        pressure_home: list
            frames at which home pressure observed.
        pressure_away: list
            frames at which away pressure observed.
        pressure_index_home: list
            filtered pressure home frames.
        pressure_index_away: list
            filtered pressure away frames.
        """
        pressure_home = []
        pressure_away = []
        for i,p in enumerate(self.tracking.possession_info_raw):
            if self.tracking.ball_state[i] == "Alive": #ball should be alive
                if p == "H":
                    #Defender distance to ball
                    away_dist_to_ball = np.sqrt(np.sum((self.tracking.away_coords[i] - self.tracking.ball_coords[i]) ** 2, axis = 1))
                    away_in_r0 = away_dist_to_ball[away_dist_to_ball <= pressure_r0] 
                    away_in_r1 = away_dist_to_ball[away_dist_to_ball <= pressure_r1]  
                    away_in_r2 = away_dist_to_ball[away_dist_to_ball <= pressure_r2]  
                    away_in_r3 = away_dist_to_ball[away_dist_to_ball <= pressure_r3]

                    #Proximity and Angle 
                    defender_coords = self.tracking.away_coords[i][np.argmin(away_dist_to_ball)] #defender coords
                    attacker_dist_to_ball = np.sqrt(np.sum((self.tracking.home_coords[i] - self.tracking.ball_coords[i]) ** 2, axis = 1)) #home distance to ball
                    attacker_coords = self.tracking.home_coords[i][np.argmin(attacker_dist_to_ball)] #player in possession
                    ball_attacker_vec = attacker_coords - self.tracking.ball_coords[i] #vector between ball and attacker
                    attacker_defender_vec = attacker_coords - defender_coords #vector between attacker and defender
                    if (np.linalg.norm(ball_attacker_vec) * np.linalg.norm(attacker_defender_vec)) == 0: #avoid division by 0
                            if len(away_in_r3) > 0:
                                pressure_away.append(i)
                    else:
                        cosine_angle = np.dot(ball_attacker_vec, attacker_defender_vec) / (np.linalg.norm(ball_attacker_vec) * np.linalg.norm(attacker_defender_vec))
                        if cosine_angle <=1 and cosine_angle >= -1:
                            angle = np.arccos(cosine_angle)
                            angle_degrees = np.degrees(angle)
                            if len(away_in_r0) > 0 and  angle_degrees <= angle_h0: #section 0 
                                pressure_away.append(i)
                            elif len(away_in_r1) > 0 and  angle_degrees > angle_h0 and angle_degrees <= angle_h1: #section 1
                                pressure_away.append(i)
                            elif len(away_in_r2) > 0 and  angle_degrees > angle_h1 and angle_degrees < angle_h2: #section 2
                                pressure_away.append(i)
                            elif len(away_in_r3) > 0 and  angle_degrees >= angle_h2: #section 3
                                pressure_away.append(i)
                elif p == "A":
                    #Defender distance to ball
                    home_dist_to_ball = np.sqrt(np.sum((self.tracking.home_coords[i] - self.tracking.ball_coords[i]) ** 2, axis = 1))
                    home_in_r0 = home_dist_to_ball[home_dist_to_ball <= pressure_r0] 
                    home_in_r1 = home_dist_to_ball[home_dist_to_ball <= pressure_r1] 
                    home_in_r2 = home_dist_to_ball[home_dist_to_ball <= pressure_r2] 
                    home_in_r3 = home_dist_to_ball[home_dist_to_ball <= pressure_r3] 
                    #Proximity and Angle 
                    defender_coords = self.tracking.home_coords[i][np.argmin(home_dist_to_ball)] #defender coords
                    attacker_dist_to_ball = np.sqrt(np.sum((self.tracking.away_coords[i] - self.tracking.ball_coords[i]) ** 2, axis = 1))#home distance to ball
                    attacker_coords = self.tracking.away_coords[i][np.argmin(attacker_dist_to_ball)]#player in possession
                    ball_attacker_vec = attacker_coords - self.tracking.ball_coords[i]#vector between ball and attacker
                    attacker_defender_vec = attacker_coords - defender_coords#vector between attacker and defender
                    if (np.linalg.norm(ball_attacker_vec) * np.linalg.norm(attacker_defender_vec)) == 0: #avoid division by 0
                            if len(home_in_r3) > 0:
                                pressure_home.append(i)
                    else:
                        cosine_angle = np.dot(ball_attacker_vec, attacker_defender_vec) / (np.linalg.norm(ball_attacker_vec) * np.linalg.norm(attacker_defender_vec))
                        if cosine_angle <=1 and cosine_angle >= -1:
                            angle = np.arccos(cosine_angle)
                            angle_degrees = np.degrees(angle)      
                            if len(home_in_r0) > 0 and  angle_degrees <= angle_h0: #section 0
                                pressure_home.append(i)
                            elif len(home_in_r1) > 0 and  angle_degrees > angle_h0 and angle_degrees <= angle_h1: #section 1
                                pressure_home.append(i)
                            elif len(home_in_r2) > 0 and  angle_degrees > angle_h1 and angle_degrees < angle_h2: #section 2
                                pressure_home.append(i)
                            elif len(home_in_r3) > 0 and  angle_degrees >= angle_h2: #section 3
                                pressure_home.append(i)
                                
        pressure_home_cnt = 0
        pressure_away_cnt = 0
        pressure_index_home = [pressure_home[0]]
        pressure_index_away = [pressure_away[0]]
        temp_cnt = 1
        for i in range(len(pressure_home)-1):
            if pressure_home[i + 1] - pressure_home[i] > pressure_threshold: # it was a really a pressure and not a continuation of an existing one
                if temp_cnt > cont_pressure_threshold: # it was really a pressure and took place a certain amount of frame
                    pressure_index_home.append(pressure_home[i + 1])
                    pressure_home_cnt += 1
                temp_cnt = 1
            else:
                temp_cnt += 1

        temp_cnt = 1
        for i in range(len(pressure_away)-1):
            if pressure_away[i + 1] - pressure_away[i] > pressure_threshold: # it was a really a pressure and not a continuation of an existing one
                if temp_cnt > cont_pressure_threshold: # it was really a pressure and took place a certain amount of frame
                    pressure_index_away.append(pressure_away[i + 1])
                    pressure_away_cnt += 1
                temp_cnt = 1
            else:
                temp_cnt += 1
        return pressure_home_cnt, pressure_away_cnt, pressure_home, pressure_away, pressure_index_home, pressure_index_away
    
    def forced_turnover(self, possession_threshold = 97):
        """ Forced turnover calculation.
        Parameters
        ----------
        pressure_threshold: integer, optional
            required number of frames which ensures the team actually had the possession previously.
            
        Return
        ---------- 
        home_forced_turnover_cnt: int
            home team forced turnovers count.
        away_forced_turonver_cnt: int
            away team forced turnovers count.
        home_forced_turnovers: list
            frames at which home forced turnovers observed.
        away_forced_turnovers: list
            frames at which away forced turnovers observed.
        """

        _, _, pressure_home, pressure_away, _, _ = self.pressure_on_ball()
        prev_team = ""
        home_forced_turnovers = []
        away_forced_turnovers = []
        poss_cnt = 0
        for i,p in enumerate(self.tracking.possession_info_raw):
            if self.tracking.ball_state[i] == "Alive":
                if p == "H":
                    if prev_team  == "A": #team in possession changes
                        if i-1 in pressure_home: #possession gained due to pressure 
                            if poss_cnt > possession_threshold : # away team actually had the possession previously
                                home_forced_turnovers.append(i) # home team force away team to lost the ball 
                        poss_cnt = 1
                    else: #ball is still in possession of home team
                        poss_cnt += 1
                    prev_team = "H"
                elif p == "A":
                    if prev_team == "H":#team in possession changes
                        if i-1 in pressure_away: #possession gained due to pressure 
                            if poss_cnt > possession_threshold : # home team actually had the possession previously
                                away_forced_turnovers.append(i) # away to force home team to lost the ball 
                        poss_cnt = 1
                    else: #ball is still in possession of away team
                        poss_cnt += 1
                    prev_team = "A"
        
        home_forced_turnover_cnt = len(home_forced_turnovers)
        away_forced_turonver_cnt = len(away_forced_turnovers)
       
        return home_forced_turnover_cnt, away_forced_turonver_cnt, home_forced_turnovers, away_forced_turnovers
    
    def final_third_entries(self, out_of_final_third_threshold = 128, possession_threshold = 107, zone_r = 1.84, pass_reception_duration = 14,
                            ch_x = 34, in_ch_x = 20.16, cent_ch_x = 9.16, y_final_up = 52.5, y_final_down = 17.5, x_final = 34,
                            event_x_final_down = 0.67, event_x_final_up = 1, y_event_ch_left = 1, y_event_ch_right = 0, 
                            y_event_in_ch_left = 0.796, y_event_in_ch_right = 0.204, y_event_cent_ch_left = 0.634, y_event_cent_ch_right = 0.366):

        """ Final third entries calculation.
        Parameters
        ----------
        out_of_final_third_threshold: integer, optional
            required number of frame gap between two final third entries events.
        pressure_threshold: integer, optional
            required number of frames which ensures the team actually had the possession previously.
        zone_r: float, optional
            radius ensuring the ball is progressed to the final third area.
        pass_reception_duration: integer, optional
            number of frames used to improve the syncronization of event data and tracking data in frame level.
        Return
        ---------- 
        home_entries: int
            home team final third entries counts for each channel.
        away_entries: int
            away team final third entries counts for each channel.
        home_entries_idx: list
            home team final third entries frames.
        away_entries_idx: list
            away team final third entries frames.
        """

        ft_pass_events = self.events.events[self.events.events["event"].isin(["pass", "cross"])][['x_location_start_mirrored',"x_location_end_mirrored", "y_location_end_mirrored", "match_run_time_in_ms", "from_player_id", "to_player_id"]]
        ft_pass_events = ft_pass_events.dropna(subset = ["from_player_id"])
        ft_pass_events = ft_pass_events.dropna(subset = ["to_player_id"])
        ft_pass_events = ft_pass_events.astype({'from_player_id': 'int64'})
        ft_pass_events = ft_pass_events.astype({'to_player_id': 'int64'})
        ft_pass_events["match_run_time_in_ms"] = ft_pass_events["match_run_time_in_ms"] // 40 # 1 frame = 40 ms
        ft_pass_events = ft_pass_events.rename(columns={"match_run_time_in_ms":"frame"})
        ft_pass_events = ft_pass_events[ft_pass_events["x_location_start_mirrored"] <= event_x_final_down] # start before final third
        ft_pass_events = ft_pass_events[ft_pass_events["x_location_end_mirrored"] > event_x_final_down] # end in final third
        ft_pass_events = ft_pass_events[ft_pass_events["x_location_end_mirrored"] <= event_x_final_up] # inside pitch
        ft_pass_events = ft_pass_events[ft_pass_events["y_location_end_mirrored"] <= y_event_ch_left] # inside pitch
        ft_pass_events = ft_pass_events[ft_pass_events["y_location_end_mirrored"] >= y_event_ch_right] #inside pitch

        match_start_frame = list(self.tracking.tracking["frame"])[0]
        pass_frames = np.array(ft_pass_events["frame"]) + match_start_frame
        adjusted_pass_frames = []
        #adjust frames for passes
        for p in pass_frames:
            old_p = p 
            if old_p > self.match.meta["Phase2StartFrame"][0]:
                p -= self.match.meta["Phase2StartFrame"][0] - self.match.meta["Phase1EndFrame"][0]

            if self.tracking.match_len > 2: # Over Time
                if old_p > self.match.meta["Phase3StartFrame"][0]: # end of match - start of first OT
                    p-= self.match.meta["Phase3StartFrame"][0] - self.match.meta["Phase2EndFrame"][0]
                if old_p > self.match.meta["Phase4StartFrame"][0]: # start of first OT - start of second OT
                    p-= self.match.meta["Phase4StartFrame"][0] - self.match.meta["Phase3EndFrame"][0]
                
                if self.tracking.match_len > 4: # Penalties
                    if old_p > self.match.meta["Phase5StartFrame"][0]: # end of match - start of first OT
                        p-= self.match.meta["Phase5StartFrame"][0] - self.match.meta["Phase4EndFrame"][0]
            
            adjusted_pass_frames.append(p - match_start_frame)

        y_end_locations = list(ft_pass_events["y_location_end_mirrored"])
        player_ids_event_passer = list(ft_pass_events["from_player_id"])
        player_ids_event_receiver = list(ft_pass_events["to_player_id"])
        team_ids = list(self.match.lineups["team_id"])
        player_ids_match = list(self.match.lineups["player_id"])
        home_id = self.match.meta["home_team_id"][0]
        away_id = self.match.meta["away_team_id"][0]
        #home entries
        home_left_ch = []
        home_left_in_ch = []
        home_cent_ch = []
        home_right_in_ch = []
        home_right_ch = []
        #away entries
        away_left_ch = []
        away_left_in_ch = []
        away_cent_ch = []
        away_right_in_ch = []
        away_right_ch = []

        for i,f in enumerate(adjusted_pass_frames): #count final third entries via pass 
            f += pass_reception_duration # make frame close to the end of the pass
            passer = player_ids_event_passer[i]
            passer_idx = player_ids_match.index(passer)
            passer_team = team_ids[passer_idx]
            receiver = player_ids_event_receiver[i] 
            receiver_idx = player_ids_match.index(receiver)
            receiver_team = team_ids[receiver_idx]

            if receiver_team == home_id and passer_team == home_id: # successful pass home
                if y_end_locations[i] <= y_event_ch_left and  y_end_locations[i] > y_event_in_ch_left: # left channel
                    home_left_ch.append(f)
                elif y_end_locations[i] <= y_event_in_ch_left and y_end_locations[i] >= y_event_cent_ch_left: # left inside channel
                    home_left_in_ch.append(f)
                elif y_end_locations[i] < y_event_cent_ch_left and y_end_locations[i] > y_event_cent_ch_right: # central channel
                    home_cent_ch.append(f)
                elif y_end_locations[i] <= y_event_cent_ch_right and  y_end_locations[i] >= y_event_in_ch_right: # right inside channel
                    home_right_in_ch.append(f)
                elif y_end_locations[i] < y_event_in_ch_right and  y_end_locations[i] >= y_event_ch_right: # left channel
                    home_right_ch.append(f)
                                            
            elif receiver_team == away_id and passer_team == away_id: # successful pass away
                if y_end_locations[i] <= y_event_ch_left and  y_end_locations[i] > y_event_in_ch_left: # left channel
                    away_left_ch.append(f)
                elif y_end_locations[i] <= y_event_in_ch_left and y_end_locations[i] >= y_event_cent_ch_left: # left inside channel
                    away_left_in_ch.append(f)
                elif y_end_locations[i] < y_event_cent_ch_left and y_end_locations[i] > y_event_cent_ch_right: # central channel
                    away_cent_ch.append(f)
                elif y_end_locations[i] <= y_event_cent_ch_right and  y_end_locations[i] >= y_event_in_ch_right: # right inside channel
                    away_right_in_ch.append(f)
                elif y_end_locations[i] < y_event_in_ch_right and  y_end_locations[i] >= y_event_ch_right: # left channel
                    away_right_ch.append(f)

        for i,p in enumerate(self.tracking.possession_info_raw): #count final third entries via ball carrying
            if len(self.tracking.possession_info_raw) - possession_threshold > i and i > 0:  
                if p == "H" and self.tracking.ball_state[i] == "Alive": #home ball carrying
                    if ((self.tracking.ball_coords_modified_home[i][1] > y_final_down and self.tracking.ball_coords_modified_home[i][1] <= y_final_up and
                        self.tracking.ball_coords_modified_home[i][0] >= -x_final and self.tracking.ball_coords_modified_home[i][0] <= x_final) and 
                        (self.tracking.ball_coords_modified_home[i-1][1] <= y_final_down and self.tracking.ball_coords_modified_home[i-1][1] >= -y_final_up and
                        self.tracking.ball_coords_modified_home[i-1][0] >= -x_final and self.tracking.ball_coords_modified_home[i-1][0] <= x_final)): # ball in final third 
                                                                                                                                                      # previously out of final third

                        if np.all(self.tracking.possession_info_raw[i: i+possession_threshold] == "H"): #possession home ensured
                            
                            home_dist_to_ball = np.sqrt(np.sum((self.tracking.home_coords_modified[i] - self.tracking.ball_coords_modified_home[i]) ** 2, axis = 1))
                            home_in_dz = home_dist_to_ball[home_dist_to_ball < zone_r]
                            if len(home_in_dz) > 0:# are you really carrying the ball
                                if self.tracking.ball_coords_modified_home[i][0] >= -ch_x and self.tracking.ball_coords_modified_home[i][0] < -in_ch_x: # left channel
                                    home_left_ch.append(i)
                                elif self.tracking.ball_coords_modified_home[i][0] >= -in_ch_x and self.tracking.ball_coords_modified_home[i][0] <= -cent_ch_x: # left inside channel
                                    home_left_in_ch.append(i)
                                elif self.tracking.ball_coords_modified_home[i][0] > -cent_ch_x and  self.tracking.ball_coords_modified_home[i][0] < cent_ch_x: # central channel
                                    home_cent_ch.append(i)
                                elif self.tracking.ball_coords_modified_home[i][0] >= cent_ch_x and self.tracking.ball_coords_modified_home[i][0] <= in_ch_x: # right inside channel
                                    home_right_in_ch.append(i)
                                elif self.tracking.ball_coords_modified_home[i][0] > in_ch_x and self.tracking.ball_coords_modified_home[i][0] < ch_x: # right channel
                                    home_right_ch.append(i)
                            
                elif p == "A" and self.tracking.ball_state[i] == "Alive": #away ball carrying
                    if ((self.tracking.ball_coords_modified_away[i][1] > y_final_down and self.tracking.ball_coords_modified_away[i][1] <= y_final_up and 
                        self.tracking.ball_coords_modified_away[i][0] >= -x_final and self.tracking.ball_coords_modified_away[i][0] <= x_final) and 
                        (self.tracking.ball_coords_modified_away[i-1][1] <= y_final_down and self.tracking.ball_coords_modified_away[i-1][1] >= -y_final_up and
                        self.tracking.ball_coords_modified_away[i-1][0] >= -x_final and self.tracking.ball_coords_modified_away[i-1][0] <= x_final)): # ball in final third                                                                                                 # previously out of final third
                                                                                                                                                      # previously out of final third
                        if np.all(self.tracking.possession_info_raw[i: i+possession_threshold] == "A"): #possession away ensured

                            away_dist_to_ball = np.sqrt(np.sum((self.tracking.away_coords_modified[i] - self.tracking.ball_coords_modified_away[i]) ** 2, axis = 1))
                            away_in_dz = away_dist_to_ball[away_dist_to_ball < zone_r]
                            if len(away_in_dz) > 0: # are you really carrying the ball
                                if self.tracking.ball_coords_modified_away[i][0] >= -ch_x and self.tracking.ball_coords_modified_away[i][0] < -in_ch_x: # left channel
                                    away_left_ch.append(i)
                                elif self.tracking.ball_coords_modified_away[i][0] >= -in_ch_x and self.tracking.ball_coords_modified_away[i][0] <= -cent_ch_x: # left inside channel
                                    away_left_in_ch.append(i)
                                elif self.tracking.ball_coords_modified_away[i][0] > -cent_ch_x and  self.tracking.ball_coords_modified_away[i][0] < cent_ch_x: # central channel
                                    away_cent_ch.append(i)
                                elif self.tracking.ball_coords_modified_away[i][0] >= cent_ch_x and self.tracking.ball_coords_modified_away[i][0] <= in_ch_x: # right inside channel
                                    away_right_in_ch.append(i)
                                elif self.tracking.ball_coords_modified_away[i][0] > in_ch_x and self.tracking.ball_coords_modified_away[i][0] < ch_x: # right channel
                                    away_right_ch.append(i)
        
        home_entries_merged = sorted(home_left_ch + home_left_in_ch + home_cent_ch + home_right_in_ch + home_right_ch)
        away_entries_merged = sorted(away_left_ch + away_left_in_ch + away_cent_ch + away_right_in_ch + away_right_ch)
        home_left_ch_new = []
        home_left_in_ch_new = []
        home_cent_ch_new = []
        home_right_in_ch_new = []
        home_right_ch_new = []

        away_left_ch_new = []
        away_left_in_ch_new = []
        away_cent_ch_new = []
        away_right_in_ch_new = []
        away_right_ch_new = []
        #count number of entries
        for i in range(len(home_entries_merged)-1, -1, -1):
            if i == 0:
                if home_entries_merged[i] in home_left_ch:
                    home_left_ch_new.append(home_entries_merged[i])

                elif home_entries_merged[i] in home_left_in_ch:
                    home_left_in_ch_new.append(home_entries_merged[i])

                elif home_entries_merged[i] in home_cent_ch:
                    home_cent_ch_new.append(home_entries_merged[i])

                elif home_entries_merged[i] in home_right_in_ch:
                    home_right_in_ch_new.append(home_entries_merged[i])
        
                elif home_entries_merged[i] in home_right_ch:
                    home_right_ch_new.append(home_entries_merged[i])

            elif home_entries_merged[i] -  home_entries_merged[i-1] > out_of_final_third_threshold: # avoid duplicate counting home
                if home_entries_merged[i] in home_left_ch:
                    home_left_ch_new.append(home_entries_merged[i])

                elif home_entries_merged[i] in home_left_in_ch:
                    home_left_in_ch_new.append(home_entries_merged[i])

                elif home_entries_merged[i] in home_cent_ch:
                    home_cent_ch_new.append(home_entries_merged[i])

                elif home_entries_merged[i] in home_right_in_ch:
                    home_right_in_ch_new.append(home_entries_merged[i])
        
                elif home_entries_merged[i] in home_right_ch:
                    home_right_ch_new.append(home_entries_merged[i])
        
        
        for i in range(len(away_entries_merged)-1, -1, -1):
            if i == 0:
                if away_entries_merged[i] in away_left_ch:
                    away_left_ch_new.append(away_entries_merged[i])

                elif away_entries_merged[i] in away_left_in_ch:
                    away_left_in_ch_new.append(away_entries_merged[i])

                elif away_entries_merged[i] in away_cent_ch:
                    away_cent_ch_new.append(away_entries_merged[i])

                elif away_entries_merged[i] in away_right_in_ch:
                    away_right_in_ch_new.append(away_entries_merged[i])
        
                elif away_entries_merged[i] in away_right_ch:
                    away_right_ch_new.append(away_entries_merged[i])

            if away_entries_merged[i] -  away_entries_merged[i-1] > out_of_final_third_threshold: # avoid duplicate counting away
                if away_entries_merged[i] in away_left_ch:
                    away_left_ch_new.append(away_entries_merged[i])

                elif away_entries_merged[i] in away_left_in_ch:
                    away_left_in_ch_new.append(away_entries_merged[i])

                elif away_entries_merged[i] in away_cent_ch:
                    away_cent_ch_new.append(away_entries_merged[i])

                elif away_entries_merged[i] in away_right_in_ch:
                    away_right_in_ch_new.append(away_entries_merged[i])
        
                elif away_entries_merged[i] in away_right_ch:
                    away_right_ch_new.append(away_entries_merged[i])

        home_entries_idx = [home_left_ch_new, home_left_in_ch_new, home_cent_ch_new, home_right_in_ch_new, home_right_ch_new]
        away_entries_idx = [away_left_ch_new, away_left_in_ch_new, away_cent_ch_new, away_right_in_ch_new, away_right_ch_new]
        home_entries = [len(home_left_ch_new), len(home_left_in_ch_new), len(home_cent_ch_new), len(home_right_in_ch_new), len(home_right_ch_new)]
        away_entries = [len(away_left_ch_new), len(away_left_in_ch_new), len(away_cent_ch_new), len(away_right_in_ch_new), len(away_right_ch_new)]

        return np.array(home_entries), np.array(away_entries), home_entries_idx, away_entries_idx
    
    def team_shape(self):
        """ Team shape calculation.
        Return
        ---------- 
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
        home_team_shapes = []
        away_team_shapes = []
        home_team_shapes_in = []
        away_team_shapes_in = []
        home_team_shapes_out = []
        away_team_shapes_out = []

        alive_frame_idx = np.where(self.tracking.ball_state == "Alive")[0] #get alive frames for possession
        for i in alive_frame_idx:
            temp_home_coords = self.tracking.home_coords_modified[i][:,1] #current home coords
            temp_away_coords = self.tracking.away_coords_modified[i][:,1] #current away coords
            home_keeper_idx = np.argmin(temp_home_coords) # home keeper 
            away_keeper_idx = np.argmin(temp_away_coords) # away keeper
            temp_home_coords = np.delete(temp_home_coords, home_keeper_idx) #exclude home keeper
            temp_away_coords = np.delete(temp_away_coords, away_keeper_idx) #exclude away keeper

            home_defensive_down = np.min(temp_home_coords) # home deepest 
            away_defensive_down = np.min(temp_away_coords) # away deepest
            home_offensive_up = np.max(temp_home_coords) # home farthest
            away_offensive_up = np.max(temp_away_coords) # away farthest

            home_interval = (home_offensive_up - home_defensive_down) / 4 # y - interval home
            away_interval = (away_offensive_up - away_defensive_down) / 4 # y - interval away

            home_offensive_down = home_offensive_up - home_interval # home offensive down
            away_offensive_down = away_offensive_up - away_interval # away offensive down
            home_defensive_up = home_defensive_down + home_interval # home defensive up
            away_defensive_up = away_defensive_down + away_interval # away defensive up

            home_offensive_down = home_offensive_up - home_interval # home offensive down
            away_offensive_down = away_offensive_up - away_interval # away offensive down
            home_defensive_up = home_defensive_down + home_interval # home defensive up
            away_defensive_up = away_defensive_down + away_interval # away defensive up

            home_mid = home_defensive_up + home_interval # home mid
            away_mid = away_defensive_up + away_interval # away mid

            home_offensive_players = len(temp_home_coords[(temp_home_coords <= home_offensive_up) & (temp_home_coords > home_offensive_down)]) #home offensive players
            home_midfield_up_players = len(temp_home_coords[(temp_home_coords <= home_offensive_down) & (temp_home_coords >= home_mid)]) #home mid up players
            home_midfield_down_players = len(temp_home_coords[(temp_home_coords < home_mid) & (temp_home_coords >= home_defensive_up)]) #home mid down players
            home_defensive_players = len(temp_home_coords[(temp_home_coords < home_defensive_up) & (temp_home_coords >= home_defensive_down)]) #home defensive players

            away_offensive_players = len(temp_away_coords[(temp_away_coords <= away_offensive_up) & (temp_away_coords > away_offensive_down)]) #away offensive players
            away_midfield_up_players = len(temp_home_coords[(temp_away_coords <= away_offensive_down) & (temp_away_coords >= away_mid)]) #away mid up players
            away_midfield_down_players = len(temp_home_coords[(temp_away_coords < away_mid) & (temp_away_coords >= away_defensive_up)]) #away mid  down players
            away_defensive_players = len(temp_away_coords[(temp_away_coords < away_defensive_up) & (temp_away_coords >= away_defensive_down)]) #away offensive players

            # Try to capture 4-2-3-1 , 4-1-4-1 and 3-4-1-2 formations
            if home_midfield_down_players + home_midfield_up_players >= 5: #4 - code home 
                home_team_shapes.append([home_defensive_players, home_midfield_up_players, home_midfield_down_players, home_offensive_players])
            else: # 3 - code home
                home_interval = (home_offensive_up - home_defensive_down) / 3 # y - interval home
                home_offensive_down = home_offensive_up - home_interval # home offensive down
                home_defensive_up = home_defensive_down + home_interval # home defensive up
                home_offensive_players = len(temp_home_coords[(temp_home_coords <= home_offensive_up) & (temp_home_coords > home_offensive_down)]) #home offensive players
                home_midfield_players = len(temp_home_coords[(temp_home_coords <= home_offensive_down) & (temp_home_coords >= home_defensive_up)]) #home mid players
                home_defensive_players = len(temp_home_coords[(temp_home_coords < home_defensive_up) & (temp_home_coords >= home_defensive_down)]) #home defensive players
                home_team_shapes.append([home_defensive_players, home_midfield_players, home_offensive_players]) #home team shape

            if away_midfield_down_players + away_midfield_up_players >= 5: #4 - code away
                away_team_shapes.append([away_defensive_players, away_midfield_down_players, away_midfield_up_players, away_offensive_players])
            else: # 3 - code away
                away_interval = (away_offensive_up - away_defensive_down) / 3 # y - interval away
                away_offensive_down = away_offensive_up - away_interval # away offensive down
                away_defensive_up = away_defensive_down + away_interval # away defensive up
                away_offensive_players = len(temp_away_coords[(temp_away_coords <= away_offensive_up) & (temp_away_coords > away_offensive_down)]) #away offensive players
                away_midfield_players = len(temp_away_coords[(temp_away_coords <= away_offensive_down) & (temp_away_coords >= away_defensive_up)]) #away offensive players
                away_defensive_players = len(temp_away_coords[(temp_away_coords < away_defensive_up) & (temp_away_coords >= away_defensive_down)]) #away offensive players
                away_team_shapes.append([away_defensive_players, away_midfield_players, away_offensive_players])#away team shape
            

            if self.tracking.possession_info_raw[i] == "H": # home in possession - away out of possession
                home_team_shapes_in.append(home_team_shapes[-1])
                away_team_shapes_out.append(away_team_shapes[-1])
            elif self.tracking.possession_info_raw[i] == "A": # home out of possession -  away in possession
                home_team_shapes_out.append(home_team_shapes[-1])
                away_team_shapes_in.append(away_team_shapes[-1])
           
        # Overall team shape count
        home_freq = {}
        away_freq = {}
        for t in home_team_shapes:
            if str(t) not in list(home_freq.keys()):
                home_freq[str(t)] = 0
            home_freq[str(t)] += 1
        for t in away_team_shapes:
            if str(t) not in list(away_freq.keys()):
                away_freq[str(t)] = 0
            away_freq[str(t)] += 1

        home_team_shape = max(home_freq, key=home_freq.get)
        away_team_shape = max(away_freq, key=away_freq.get)

        #Home in possession - Away out of possession team shape count
        home_freq_in = {}
        away_freq_out = {}
        for t in home_team_shapes_in:
            if str(t) not in list(home_freq_in.keys()):
                home_freq_in[str(t)] = 0
            home_freq_in[str(t)] += 1
        for t in away_team_shapes_out:
            if str(t) not in list(away_freq_out.keys()):
                away_freq_out[str(t)] = 0
            away_freq_out[str(t)] += 1

        home_team_shape_in = max(home_freq_in, key=home_freq_in.get)
        away_team_shape_out = max(away_freq_out, key=away_freq_out.get)

        #Home out of possession - Away in possession team shape count
        home_freq_out = {}
        away_freq_in = {}
        for t in home_team_shapes_out:
            if str(t) not in list(home_freq_out.keys()):
                home_freq_out[str(t)] = 0
            home_freq_out[str(t)] += 1
        for t in away_team_shapes_in:
            if str(t) not in list(away_freq_in.keys()):
                away_freq_in[str(t)] = 0
            away_freq_in[str(t)] += 1

        home_team_shape_out = max(home_freq_out, key=home_freq_out.get)
        away_team_shape_in = max(away_freq_in, key=away_freq_in.get)

        return home_team_shape, away_team_shape, home_team_shape_in, away_team_shape_in, home_team_shape_out, away_team_shape_out
    
    def line_height_team_length(self, y_final_up = 52.5, y_final_down = 17.5, y_middle_up = 17.5, 
                                y_middle_down = -17.5, y_defensive_up = -17.5, y_defensive_down = -52.5):
        """ Line height team length calculation.
        Return
        ---------- 
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
        #alive possession info
        alive_frames = self.tracking.possession_info_raw[np.where(self.tracking.ball_state == "Alive")] #get alive frames for possession
        #home
        alive_home_coords_modified = self.tracking.home_coords_modified[np.where(self.tracking.ball_state == "Alive")] #get alive frames for home coords
        alive_ball_coords_modified_home = self.tracking.ball_coords_modified_home[np.where(self.tracking.ball_state == "Alive")] #get alive frames for home ball coords modified
        #away
        alive_away_coords_modified = self.tracking.away_coords_modified[np.where(self.tracking.ball_state == "Alive")] #get alive frames for away coords
        alive_ball_coords_modified_away = self.tracking.ball_coords_modified_away[np.where(self.tracking.ball_state == "Alive")] #get alive frames for away ball coords modified

        #line height variables
        home_defensive_line_height_defensive_third = []
        home_defensive_line_height_middle_third = []
        home_defensive_line_height_final_third = []
        home_offensive_line_height_defensive_third = []
        home_offensive_line_height_middle_third = []
        home_offensive_line_height_final_third = []

        away_defensive_line_height_defensive_third = []
        away_defensive_line_height_middle_third = []
        away_defensive_line_height_final_third = []
        away_offensive_line_height_defensive_third = []
        away_offensive_line_height_middle_third = []
        away_offensive_line_height_final_third = []

        #team length variables
        home_defensive_team_length_defensive_third = []
        home_defensive_team_length_middle_third = []
        home_defensive_team_length_final_third = []
        home_offensive_team_length_defensive_third = []
        home_offensive_team_length_middle_third = []
        home_offensive_team_length_final_third = []

        away_defensive_team_length_defensive_third = []
        away_defensive_team_length_middle_third = []
        away_defensive_team_length_final_third = []
        away_offensive_team_length_defensive_third = []
        away_offensive_team_length_middle_third = []
        away_offensive_team_length_final_third = []

        #team width variables
        home_defensive_team_width_defensive_third = []
        home_defensive_team_width_middle_third = []
        home_defensive_team_width_final_third = []
        home_offensive_team_width_defensive_third = []
        home_offensive_team_width_middle_third = []
        home_offensive_team_width_final_third = []

        away_defensive_team_width_defensive_third = []
        away_defensive_team_width_middle_third = []
        away_defensive_team_width_final_third = []
        away_offensive_team_width_defensive_third = []
        away_offensive_team_width_middle_third = []
        away_offensive_team_width_final_third = []


        for i,p in enumerate(alive_frames):
            #line height calculation
            deepest_y_home = np.sort(alive_home_coords_modified[i][:,1])[1] #exclude goal keeper
            line_height_home = deepest_y_home - y_defensive_down
            deepest_y_away = np.sort(alive_away_coords_modified[i][:,1])[1] #exclude goal keeper
            line_height_away = deepest_y_away - y_defensive_down

            #team length calculation
            highest_y_home = np.sort(alive_home_coords_modified[i][:,1])[-1]
            team_length_home = highest_y_home - deepest_y_home
            highest_y_away = np.sort(alive_away_coords_modified[i][:,1])[-1]
            team_length_away = highest_y_away - deepest_y_away

            #team width calculation
            left_most_x_home = np.sort(alive_home_coords_modified[i][:,0])[0] 
            right_most_x_home = np.sort(alive_home_coords_modified[i][:,0])[-1] 
            team_width_home = right_most_x_home - left_most_x_home
            left_most_x_away = np.sort(alive_away_coords_modified[i][:,0])[0] 
            right_most_x_away = np.sort(alive_away_coords_modified[i][:,0])[-1] 
            team_width_away = right_most_x_away - left_most_x_away

            if p == "H": #offensive home defensive away
                # in possession for Home - out of possession for Away
                y_ball = alive_ball_coords_modified_home[i][1]
                if y_ball <= y_final_up and y_ball > y_final_down: # final third Home - defensive third Away
                    home_offensive_line_height_final_third.append(line_height_home)
                    home_offensive_team_length_final_third.append(team_length_home)
                    home_offensive_team_width_final_third.append(team_width_home)

                    away_defensive_line_height_defensive_third.append(line_height_away)
                    away_defensive_team_length_defensive_third.append(team_length_away)
                    away_defensive_team_width_defensive_third.append(team_width_away)

                elif y_ball <= y_middle_up and y_ball >= y_middle_down: # middle third Home - middle third Away
                    home_offensive_line_height_middle_third.append(line_height_home)
                    home_offensive_team_length_middle_third.append(team_length_home)
                    home_offensive_team_width_middle_third.append(team_width_home)

                    away_defensive_line_height_middle_third.append(line_height_away)
                    away_defensive_team_length_middle_third.append(team_length_away)
                    away_defensive_team_width_middle_third.append(team_width_away)

                elif y_ball < y_defensive_up and y_ball >= y_defensive_down: # defensive third Home - final third Away
                    home_offensive_line_height_defensive_third.append(line_height_home)
                    home_offensive_team_length_defensive_third.append(team_length_home)
                    home_offensive_team_width_defensive_third.append(team_width_home)

                    away_defensive_line_height_final_third.append(line_height_away)
                    away_defensive_team_length_final_third.append(team_length_away)
                    away_defensive_team_width_final_third.append(team_width_away)
                    
            elif p == "A":  #offensive away defensive home
                # in possession for Away - out of possession for Home
                y_ball = alive_ball_coords_modified_away[i][1]
                if y_ball <= y_final_up and y_ball > y_final_down: # final third Away - defensive third Home
                    away_offensive_line_height_final_third.append(line_height_away)
                    away_offensive_team_length_final_third.append(team_length_away)
                    away_offensive_team_width_final_third.append(team_width_away)

                    home_defensive_line_height_defensive_third.append(line_height_home)
                    home_defensive_team_length_defensive_third.append(team_length_home)
                    home_defensive_team_width_defensive_third.append(team_width_home)

                elif y_ball <= y_middle_up and y_ball >= y_middle_down: # middle third Away - middle third Home
                    away_offensive_line_height_middle_third.append(line_height_away)
                    away_offensive_team_length_middle_third.append(team_length_away)
                    away_offensive_team_width_middle_third.append(team_width_away)

                    home_defensive_line_height_middle_third.append(line_height_home)
                    home_defensive_team_length_middle_third.append(team_length_home)
                    home_defensive_team_width_middle_third.append(team_width_home)

                elif y_ball < y_defensive_up and y_ball >= y_defensive_down: # defensive third Away - final third Home
                    away_offensive_line_height_defensive_third.append(line_height_away)
                    away_offensive_team_length_defensive_third.append(team_length_away)
                    away_offensive_team_width_defensive_third.append(team_width_away)

                    home_defensive_line_height_final_third.append(line_height_home)
                    home_defensive_team_length_final_third.append(team_length_home)
                    home_defensive_team_width_final_third.append(team_width_home)
        
        #aggregate results 
        #line height variables
        avg_home_defensive_line_height_defensive_third = round(sum(home_defensive_line_height_defensive_third) / len(home_defensive_line_height_defensive_third))
        avg_home_defensive_line_height_middle_third = round(sum(home_defensive_line_height_middle_third) / len(home_defensive_line_height_middle_third))
        avg_home_defensive_line_height_final_third = round(sum(home_defensive_line_height_final_third) / len(home_defensive_line_height_final_third))
        avg_home_defensive_line_heights = [avg_home_defensive_line_height_defensive_third, avg_home_defensive_line_height_middle_third, avg_home_defensive_line_height_final_third]

        avg_home_offensive_line_height_defensive_third = round(sum(home_offensive_line_height_defensive_third) / len(home_offensive_line_height_defensive_third))
        avg_home_offensive_line_height_middle_third = round(sum(home_offensive_line_height_middle_third) / len(home_offensive_line_height_middle_third))
        avg_home_offensive_line_height_final_third = round(sum(home_offensive_line_height_final_third) / len(home_offensive_line_height_final_third))
        avg_home_offensive_line_heights = [avg_home_offensive_line_height_defensive_third, avg_home_offensive_line_height_middle_third, avg_home_offensive_line_height_final_third]

        avg_away_defensive_line_height_defensive_third = round(sum(away_defensive_line_height_defensive_third) / len(away_defensive_line_height_defensive_third))
        avg_away_defensive_line_height_middle_third = round(sum(away_defensive_line_height_middle_third) / len(away_defensive_line_height_middle_third))
        avg_away_defensive_line_height_final_third = round(sum(away_defensive_line_height_final_third ) / len(away_defensive_line_height_final_third))
        avg_away_defensive_line_heights = [avg_away_defensive_line_height_defensive_third, avg_away_defensive_line_height_middle_third, avg_away_defensive_line_height_final_third]

        avg_away_offensive_line_height_defensive_third = round(sum(away_offensive_line_height_defensive_third) / len(away_offensive_line_height_defensive_third))
        avg_away_offensive_line_height_middle_third = round(sum(away_offensive_line_height_middle_third) / len(away_offensive_line_height_middle_third))
        avg_away_offensive_line_height_final_third = round(sum(away_offensive_line_height_final_third) / len(away_offensive_line_height_final_third))
        avg_away_offensive_line_heights = [avg_away_offensive_line_height_defensive_third, avg_away_offensive_line_height_middle_third, avg_away_offensive_line_height_final_third]

        #team length variables
        avg_home_defensive_team_length_defensive_third = round(sum(home_defensive_team_length_defensive_third) / len(home_defensive_team_length_defensive_third))
        avg_home_defensive_team_length_middle_third = round(sum(home_defensive_team_length_middle_third) / len(home_defensive_team_length_middle_third))
        avg_home_defensive_team_length_final_third =  round(sum(home_defensive_team_length_final_third) / len(home_defensive_team_length_final_third))
        avg_home_defensive_team_lengths = [avg_home_defensive_team_length_defensive_third, avg_home_defensive_team_length_middle_third, avg_home_defensive_team_length_final_third]
        
        avg_home_offensive_team_length_defensive_third = round(sum(home_offensive_team_length_defensive_third) / len(home_offensive_team_length_defensive_third))
        avg_home_offensive_team_length_middle_third = round(sum(home_offensive_team_length_middle_third) / len(home_offensive_team_length_middle_third))
        avg_home_offensive_team_length_final_third = round(sum(home_offensive_team_length_final_third) / len(home_offensive_team_length_final_third))
        avg_home_offensive_team_lengths = [avg_home_offensive_team_length_defensive_third, avg_home_offensive_team_length_middle_third, avg_home_offensive_team_length_final_third]

        avg_away_defensive_team_length_defensive_third =  round(sum(away_defensive_team_length_defensive_third) / len(away_defensive_team_length_defensive_third))
        avg_away_defensive_team_length_middle_third = round(sum(away_defensive_team_length_middle_third) / len(away_defensive_team_length_middle_third))
        avg_away_defensive_team_length_final_third = round(sum(away_defensive_team_length_final_third) / len(away_defensive_team_length_final_third))
        avg_away_defensive_team_lengths = [avg_away_defensive_team_length_defensive_third, avg_away_defensive_team_length_middle_third, avg_away_defensive_team_length_final_third]

        avg_away_offensive_team_length_defensive_third =  round(sum(away_offensive_team_length_defensive_third) / len(away_offensive_team_length_defensive_third))
        avg_away_offensive_team_length_middle_third = round(sum(away_offensive_team_length_middle_third) / len(away_offensive_team_length_middle_third))
        avg_away_offensive_team_length_final_third =  round(sum(away_offensive_team_length_final_third)/ len(away_offensive_team_length_final_third))
        avg_away_offensive_team_lengths = [avg_away_offensive_team_length_defensive_third, avg_away_offensive_team_length_middle_third, avg_away_offensive_team_length_final_third]

        #team width variables
        avg_home_defensive_team_width_defensive_third = round(sum(home_defensive_team_width_defensive_third) / len(home_defensive_team_width_defensive_third))
        avg_home_defensive_team_width_middle_third = round(sum(home_defensive_team_width_middle_third) / len(home_defensive_team_width_middle_third))
        avg_home_defensive_team_width_final_third =  round(sum(home_defensive_team_width_final_third) / len(home_defensive_team_width_final_third))
        avg_home_defensive_team_widths = [avg_home_defensive_team_width_defensive_third, avg_home_defensive_team_width_middle_third, avg_home_defensive_team_width_final_third]

        avg_home_offensive_team_width_defensive_third = round(sum(home_offensive_team_width_defensive_third) / len(home_offensive_team_width_defensive_third))
        avg_home_offensive_team_width_middle_third = round(sum(home_offensive_team_width_middle_third) / len(home_offensive_team_width_middle_third))
        avg_home_offensive_team_width_final_third = round(sum(home_offensive_team_width_final_third) / len(home_offensive_team_width_final_third))
        avg_home_offensive_team_widths = [avg_home_offensive_team_width_defensive_third, avg_home_offensive_team_width_middle_third, avg_home_offensive_team_width_final_third]

        avg_away_defensive_team_width_defensive_third = round(sum(away_defensive_team_width_defensive_third) / len(away_defensive_team_width_defensive_third))
        avg_away_defensive_team_width_middle_third = round(sum(away_defensive_team_width_middle_third) / len(away_defensive_team_width_middle_third))
        avg_away_defensive_team_width_final_third = round(sum(away_defensive_team_width_final_third) / len(away_defensive_team_width_final_third))
        avg_away_defensive_team_widths = [avg_away_defensive_team_width_defensive_third, avg_away_defensive_team_width_middle_third, avg_away_defensive_team_width_final_third]

        avg_away_offensive_team_width_defensive_third = round(sum(away_offensive_team_width_defensive_third) / len(away_offensive_team_width_defensive_third))
        avg_away_offensive_team_width_middle_third = round(sum(away_offensive_team_width_middle_third) / len(away_offensive_team_width_middle_third))
        avg_away_offensive_team_width_final_third = round(sum(away_offensive_team_width_final_third) / len(away_offensive_team_width_final_third))
        avg_away_offensive_team_widths = [avg_away_offensive_team_width_defensive_third, avg_away_offensive_team_width_middle_third, avg_away_offensive_team_width_final_third]

        return avg_home_defensive_line_heights, avg_home_offensive_line_heights, avg_away_defensive_line_heights, avg_away_offensive_line_heights, \
               avg_home_defensive_team_lengths, avg_home_offensive_team_lengths, avg_away_defensive_team_lengths, avg_away_offensive_team_lengths, \
               avg_home_defensive_team_widths, avg_home_offensive_team_widths, avg_away_defensive_team_widths, avg_away_offensive_team_widths
    
    def phases_of_play(self, event_x_final_up = 1, y_event_ch_left = 1, y_event_ch_right = 0, #pitch frame coordinates
                       set_piece_duration = 147, # set piece heuristics
                       long_ball_threshold = 197, long_ball_height_threshold = 3.24, long_ball_distance = 0.45, prev_pass_multiplier = 0.11, # long ball heuristics
                       forward_threshold = 40, counter_attack_threshold = 94, forward_distance_threshold = 2.11, slope_threshold = 0.511, # counter attack heuristics
                       counter_press_threshold = 21, counter_press_duration = 49, # counter press heuristics
                       recovery_threshold = 139,  # recovery heuristics
                       r_opp = 5.36, r_high = 5.07, r_mid = 2.28, r_low = 1.23, # distances for various pressure cases
                       low_block_threshold = 31.85, mid_block_threshold = 35.31, high_block_threshold = 37.70, # block thresholds
                       y_final_up = 52.5, y_middle_up = 17.5, y_middle_down = 0, y_middle_down_press = -17.5,  y_defensive_down = -52.5): # pitch area y-coordinate borders
        
        """ Phases of play calculation.
        Parameters
        ----------
        set_piece_duration: integer, optional
            assigned duration after a set-piece event took place.
        long_ball_threshold: integer, optional
            duration within long ball conditions checked.
        long_ball_height_threshold: float, optional
            required minimum height of the ball for long ball event.
        prev_pass_multiplier: float, optional
            ratio of previous frames to allocate before the ling ball event took place.
        forward_threshold: integer, optional
            number of frames to check after a possession change in order to ensure that the ball moved forward.
        counter_attack_threshold: integer, optional
            number of frames allocated for a counter attack event.
        forward_distance_threshold: float, optional
            required minimum distance needed to travel to ensure the ball moved forward.
        slope_threshold: float, optiona
            required minimum slope to ensure directness.
        counter_press_threshold: integer, optional
            number of frames to check for counter press after possession lost.
        counter_press_duration: integer, optional
            number of frames allocated for counter press event.
        recovery_threshold: integer, optional
            number of frames that limits the counter press duration.
        r_opp: float, optional
            distance required between ball and the opposition to ensure opposed buid up.
        r_high: float, optional
            distance required to detect high pressure
        r_mid: float, optional
            distance required to detect mid pressure
        r_low: float, optional
            distance required to detect low pressure
        high_block_threshold: float, optional
            team length reqired to form high block
        mid_block_threshold: float, optional
            team length reqired to form mid block
        low_block_threshold: float, optional
            team length reqired to form low block

        Return
        ---------- 
         home_in_phases: list
            percent of times spent by home team in in-possession phases.
        away_in_phases: list
            percent of times spent by away team in in-possession phases.
        home_out_phases: list
            percent of times spent by home team in out-of-possession phases.
        away_out_phases: list
            percent of times spent by away team in out-of-possession phases.
        """
        alive_home_poss_len = len(np.array(list(set(np.where(self.tracking.ball_state == "Alive")[0]).intersection(set(np.where(self.tracking.possession_info_raw == "H")[0])))))
        alive_away_poss_len = len(np.array(list(set(np.where(self.tracking.ball_state == "Alive")[0]).intersection(set(np.where(self.tracking.possession_info_raw == "A")[0])))))

        used_frames_home_poss = []
        used_frames_home_out_poss = []
        used_frames_away_poss = []
        used_frames_away_out_poss = []
        
        # ---------------- SET - PIECE ------------------
        set_piece_home = []
        set_piece_away = []
        set_piece_events = ["Goal_Kick", "Corner_Kick", "Free_Kick", "Kick_Off", "Penalty", "Throw_In"]
        set_piece_flag = False
        set_piece_idx = 0
        for i,p in enumerate(self.tracking.possession_info_detailed):
            if p in set_piece_events:
                if set_piece_flag == False:
                    set_piece_idx = i
                    set_piece_flag = True
            else:
                if set_piece_flag == True:
                    if self.tracking.possession_info_raw[set_piece_idx] == "H":
                        s = self.tracking.possession_info_detailed[set_piece_idx]
                        set_piece_home.extend(list(range(i + 1, i + 1 + set_piece_duration))) #set piece home
                        
                    elif  self.tracking.possession_info_raw[set_piece_idx] == "A":
                        s = self.tracking.possession_info_detailed[set_piece_idx]
                        set_piece_away.extend(list(range(i + 1, i + 1 + set_piece_duration))) # set piece away

                set_piece_flag = False
        #Set Piece - ensure ball is alive, frames are valid and not used
        set_piece_home = sorted(list(set(set_piece_home)))
        set_piece_away = sorted(list(set(set_piece_away)))
        set_piece_home_filtered = []
        set_piece_away_filtered = []
        for s in set_piece_home:
            if s < len(self.tracking.ball_state) and self.tracking.ball_state[s] == "Alive" and self.tracking.possession_info_raw[s] == "H" and s not in used_frames_home_poss:
                set_piece_home_filtered.append(s)
        for s in set_piece_away:
            if s < len(self.tracking.ball_state) and self.tracking.ball_state[s] == "Alive" and self.tracking.possession_info_raw[s] == "A" and s not in used_frames_away_poss:
                set_piece_away_filtered.append(s)

        home_set_piece_pct = round((len(set_piece_home_filtered) / alive_home_poss_len) * 100)
        away_set_piece_pct = round((len(set_piece_away_filtered) / alive_away_poss_len) * 100)
        used_frames_home_poss.extend(set_piece_home_filtered)
        used_frames_away_poss.extend(set_piece_away_filtered) 
        

        # --------------- LONG BALL ------------------
        home_long_ball = []
        away_long_ball = []

        # find passes and do pre-proccessing
        pp_pass_events = self.events.events[self.events.events["event"].isin(["pass", "cross"])][['x_location_start_mirrored',"x_location_end_mirrored", "y_location_end_mirrored", "match_run_time_in_ms", "from_player_id", "to_player_id"]]
        pp_pass_events = pp_pass_events.dropna(subset = ["from_player_id"])
        pp_pass_events = pp_pass_events.astype({'from_player_id': 'int64'})
        pp_pass_events["match_run_time_in_ms"] = pp_pass_events["match_run_time_in_ms"] // 40 # 1 frame = 40 ms
        pp_pass_events = pp_pass_events.rename(columns={"match_run_time_in_ms":"frame"})
        pp_pass_events["vertical_distance_covered"] = pp_pass_events["x_location_end_mirrored"] -  pp_pass_events["x_location_start_mirrored"] # vertical distance 
        pp_pass_events = pp_pass_events[pp_pass_events["vertical_distance_covered"] > long_ball_distance] # vertical distance 
        pp_pass_events = pp_pass_events[pp_pass_events["x_location_end_mirrored"] <= event_x_final_up] # inside pitch
        pp_pass_events = pp_pass_events[pp_pass_events["y_location_end_mirrored"] <= y_event_ch_left] # inside pitch
        pp_pass_events = pp_pass_events[pp_pass_events["y_location_end_mirrored"] >= y_event_ch_right] #inside pitch
        match_start_frame = list(self.tracking.tracking["frame"])[0]
        pass_frames = np.array(pp_pass_events["frame"]) + match_start_frame
        adjusted_pass_frames = []
        #adjust frames for passes
        for p in pass_frames:
            old_p = p
            if old_p > self.match.meta["Phase2StartFrame"][0]:
                p -= self.match.meta["Phase2StartFrame"][0] - self.match.meta["Phase1EndFrame"][0]

            if self.tracking.match_len > 2: # Over Time
                if old_p > self.match.meta["Phase3StartFrame"][0]: # end of match - start of first OT
                    p-= self.match.meta["Phase3StartFrame"][0] - self.match.meta["Phase2EndFrame"][0]
                if old_p > self.match.meta["Phase4StartFrame"][0]: # start of first OT - start of second OT
                    p-= self.match.meta["Phase4StartFrame"][0] - self.match.meta["Phase3EndFrame"][0]
                
                if self.tracking.match_len > 4: # Penalties
                    if old_p > self.match.meta["Phase5StartFrame"][0]: # end of match - start of first OT
                        p-= self.match.meta["Phase5StartFrame"][0] - self.match.meta["Phase4EndFrame"][0]
            
            adjusted_pass_frames.append(p - match_start_frame)
            
        player_ids_event_passer = list(pp_pass_events["from_player_id"])
        team_ids = list(self.match.lineups["team_id"])
        player_ids_match = list(self.match.lineups["player_id"])
        home_id = self.match.meta["home_team_id"][0]
        away_id = self.match.meta["away_team_id"][0]
        # store long ball frames
        for i,f in enumerate(adjusted_pass_frames): 
            passer = player_ids_event_passer[i]
            passer_idx = player_ids_match.index(passer)
            passer_team = team_ids[passer_idx]
            if f+long_ball_threshold < len(self.tracking.ball_state): 
                if len(np.where(self.tracking.ball_height[f: f+long_ball_threshold] > long_ball_height_threshold)[0]) > 0: # check the height of the ball whether it is a long ball
                    if passer_team == home_id and f not in used_frames_home_poss: # successful long ball home
                        home_long_ball.extend(list(range(f - int(long_ball_threshold * prev_pass_multiplier) , f + long_ball_threshold )))
                    elif passer_team == away_id and f not in used_frames_away_poss: # successful long ball away
                        away_long_ball.extend(list(range(f - int(long_ball_threshold * prev_pass_multiplier) , f + long_ball_threshold )))
            else: 
                if len(np.where(self.tracking.ball_height[f: len(self.tracking.ball_state)] > long_ball_height_threshold)[0]) > 0: # check the height of the ball whether it is a long ball
                    long_ball_threshold_end = len(self.tracking.ball_state) - f - 1
                    if passer_team == home_id and f not in used_frames_home_poss: # successful long ball home
                        home_long_ball.extend(list(range(f - int(long_ball_threshold_end * prev_pass_multiplier) , f + long_ball_threshold_end )))
                    elif passer_team == away_id and f not in used_frames_away_poss: # successful long ball away
                        away_long_ball.extend(list(range(f - int(long_ball_threshold_end * prev_pass_multiplier) , f + long_ball_threshold_end )))

        #Long Ball - ensure ball is alive, frames are valid and not used
        long_ball_home_filtered = []
        long_ball_away_filtered = []
        home_long_ball = sorted(list(set(home_long_ball)))
        away_long_ball = sorted(list(set(away_long_ball)))
        
        for l in home_long_ball:
            if l < len(self.tracking.ball_state) and self.tracking.ball_state[l] == "Alive" and self.tracking.possession_info_raw[l] == "H" and l not in used_frames_home_poss: 
                long_ball_home_filtered.append(l)
        for l in away_long_ball:
            if l < len(self.tracking.ball_state) and self.tracking.ball_state[l] == "Alive" and self.tracking.possession_info_raw[l] == "A" and l not in used_frames_away_poss:
                long_ball_away_filtered.append(l)
        
        home_long_ball_pct = round((len(long_ball_home_filtered) * 100) / alive_home_poss_len)
        away_long_ball_pct = round((len(long_ball_away_filtered) * 100) / alive_away_poss_len) 
        used_frames_home_poss.extend(long_ball_home_filtered)
        used_frames_away_poss.extend(long_ball_away_filtered)
        
        
        # ---------------- ATTACKING TRANSITION, DEFENSIVE TRANSITION, RECOVERY, COUNTER ATTACK, COUNTER PRESS ------------------
        prev_team = ""

        home_counter_attack = []
        away_counter_attack = []

        home_counter_press = []
        away_counter_press = []

        home_recovery = []
        away_recovery = []
        home_alive_out_of_poss = np.array(list(set(np.where(self.tracking.ball_state == "Alive")[0]).intersection(set(np.where(self.tracking.possession_info_raw == "A")[0]))))
        away_alive_out_of_poss = np.array(list(set(np.where(self.tracking.ball_state == "Alive")[0]).intersection(set(np.where(self.tracking.possession_info_raw == "H")[0]))))
        home_alive_speeds = np.array(self.tracking.home_speed)[home_alive_out_of_poss]
        away_alive_speeds = np.array(self.tracking.away_speed)[away_alive_out_of_poss]
        home_avg_speed = np.sum(home_alive_speeds) / (len(home_alive_speeds) * len(self.tracking.home_speed[0])) # home average speed
        away_avg_speed = np.sum(away_alive_speeds) / (len(away_alive_speeds) * len(self.tracking.away_speed[0])) # away average speed
        _, _, pressure_home, pressure_away, _, _ = self.pressure_on_ball() # only alive frames returned
        for i,p in enumerate(self.tracking.possession_info_raw):
            if len(self.tracking.possession_info_raw) > i + counter_attack_threshold: #avoid index error
                if p == "H": # Possession Home - Out of Possession Away
                    if prev_team == "A": #possession info changed to home 
                        # Counter Attack Home
                        slope = 1000
                        y_diff = -1
                        x_diff = -1
                        if i+forward_threshold < len(self.tracking.ball_coords_modified_home):
                            y_diff = self.tracking.ball_coords_modified_home[i+forward_threshold][1] - self.tracking.ball_coords_modified_home[i][1]
                            x_diff = self.tracking.ball_coords_modified_home[i+forward_threshold][0] - self.tracking.ball_coords_modified_home[i][0]
                        else:
                            y_diff = self.tracking.ball_coords_modified_home[len(self.tracking.ball_coords_modified_home) -1][1] - self.tracking.ball_coords_modified_home[i][1]
                            x_diff = self.tracking.ball_coords_modified_home[len(self.tracking.ball_coords_modified_home) -1][0] - self.tracking.ball_coords_modified_home[i][0]

                        if x_diff != 0:
                            slope = y_diff/x_diff

                        if i + counter_attack_threshold < len(self.tracking.ball_state):
                            if (y_diff >= forward_distance_threshold and slope > slope_threshold and  # ball moved direct
                            len(np.where(self.tracking.ball_state[i:i + counter_attack_threshold] == "Alive")[0]) == counter_attack_threshold and # ball alive
                            len(np.where(self.tracking.possession_info_raw[i:i + counter_attack_threshold] == "H")[0]) == counter_attack_threshold):  # possession home
                                
                                home_counter_attack.extend(range(i, i + counter_attack_threshold))
                        else:
                            if (y_diff >= forward_distance_threshold and slope > slope_threshold and  # ball moved direct
                            len(np.where(self.tracking.ball_state[i: len(self.tracking.ball_state)] == "Alive")[0]) == counter_attack_threshold and # ball alive
                            len(np.where(self.tracking.possession_info_raw[i: len(self.tracking.ball_state)] == "H")[0]) == counter_attack_threshold):  # possession home
                                
                                home_counter_attack.extend(range(i, len(self.tracking.ball_state)))

                        # Counter Press Away
                        counter_press_idx = -1 
                        for j in range(i,i+counter_press_threshold):
                            if j in pressure_away:
                                counter_press_idx = pressure_away.index(j)
                                break
                        if counter_press_idx != -1: # counter press found
                            away_counter_press.extend(list(range(i, pressure_away[counter_press_idx] + counter_press_duration)))
                        else:
                            # Recovery Away 
                            recovery_flag = False
                            j = i
                            poss_cnt = 1
                            while(True):
                                if  j == len(self.tracking.possession_info_raw) - 1:
                                    poss_cnt = len(self.tracking.possession_info_raw) - i - 1
                                    break
                                elif  self.tracking.possession_info_raw[j] == 'H' and self.tracking.ball_state[j] == "Alive": # recovery duration count
                                    poss_cnt += 1
                                else:
                                    break
                                j+=1
                            if poss_cnt < recovery_threshold:
                                away_speed_interval = self.tracking.away_speed[i: i + poss_cnt] 
                                away_avg_speed_interval = np.sum(away_speed_interval) / (poss_cnt * len(self.tracking.away_speed[i])) #  average speed during interval
                                away_y_start = np.average(self.tracking.away_coords_modified[i][:,1]) # y start average
                                away_y_end = np.average(self.tracking.away_coords_modified[i + poss_cnt][:,1]) # y end average
                                if away_avg_speed_interval > away_avg_speed and away_y_start > away_y_end : # quickly going back to defense
                                    recovery_flag = True
                                    away_recovery.extend(range(i, i + poss_cnt))
                            elif poss_cnt >= recovery_threshold:
                                away_speed_interval = self.tracking.away_speed[i: i + recovery_threshold] 
                                away_avg_speed_interval = np.sum(away_speed_interval) / (recovery_threshold * len(self.tracking.away_speed[i])) #  average speed during interval
                                away_y_start = np.average(self.tracking.away_coords_modified[i][:,1]) # y start average
                                away_y_end = np.average(self.tracking.away_coords_modified[i + recovery_threshold][:,1]) # y end average
                                if away_avg_speed_interval > away_avg_speed and away_y_start > away_y_end : # quickly going back to defense
                                    recovery_flag = True
                                    away_recovery.extend(range(i, i + recovery_threshold))

                    prev_team = "H"    

                elif p == "A":  # Possession Away - Out of Possession Home
                    if prev_team == "H": #possession info changed to away   
                        # Counter Attack Away
                        slope = 1000
                        y_diff = -1
                        x_diff = -1
                        if i+forward_threshold < len(self.tracking.ball_coords_modified_away):
                            y_diff = self.tracking.ball_coords_modified_away[i+forward_threshold][1] - self.tracking.ball_coords_modified_away[i][1]
                            x_diff = self.tracking.ball_coords_modified_away[i+forward_threshold][0] - self.tracking.ball_coords_modified_away[i][0]
                        else:
                            y_diff = self.tracking.ball_coords_modified_away[len(self.tracking.ball_coords_modified_away) -1][1] - self.tracking.ball_coords_modified_away[i][1]
                            x_diff = self.tracking.ball_coords_modified_away[len(self.tracking.ball_coords_modified_away) -1][0] - self.tracking.ball_coords_modified_away[i][0]

                        if x_diff != 0:
                            slope = y_diff/x_diff      

                        if i + counter_attack_threshold < len(self.tracking.ball_state):               
                            if (y_diff >= forward_distance_threshold and slope > slope_threshold and # ball moved direct
                            len(np.where(self.tracking.ball_state[i:i + counter_attack_threshold] == "Alive")[0]) ==  counter_attack_threshold and # ball alive
                            len(np.where(self.tracking.possession_info_raw[i:i + counter_attack_threshold] == "A")[0]) == counter_attack_threshold):  # possession away
                                
                                away_counter_attack.extend(range(i, i + counter_attack_threshold))
                        else:
                            if (y_diff >= forward_distance_threshold and slope > slope_threshold and # ball moved direct
                            len(np.where(self.tracking.ball_state[i:len(self.tracking.ball_state)] == "Alive")[0]) ==  counter_attack_threshold and # ball alive
                            len(np.where(self.tracking.possession_info_raw[i:len(self.tracking.ball_state)] == "A")[0]) == counter_attack_threshold):  # possession away
                                
                                away_counter_attack.extend(range(i, len(self.tracking.ball_state)))
                        # Counter Press Home
                        counter_press_idx = -1
                        for j in range(i,i+counter_press_threshold):
                            if j in pressure_home:
                                counter_press_idx = pressure_home.index(j)
                                break
                        if counter_press_idx != -1: # counter press found
                            home_counter_press.extend(list(range(i, pressure_home[counter_press_idx]+counter_press_duration)))
                        else:
                            # Recovery Home   
                            recovery_flag = False
                            j = i
                            poss_cnt = 1
                            while(True):
                                if j ==  len(self.tracking.possession_info_raw) -1:
                                    poss_cnt = len(self.tracking.possession_info_raw) - i - 1
                                    break
                                elif self.tracking.possession_info_raw[j] == 'A' and self.tracking.ball_state[j] == "Alive": # recovery duration count
                                    poss_cnt += 1
                                else:
                                    break
                                j+=1
                                
                            if poss_cnt < recovery_threshold:
                                home_speed_interval = self.tracking.home_speed[i: i + poss_cnt] 
                                home_avg_speed_interval = np.sum(home_speed_interval) / (poss_cnt * len(self.tracking.home_speed[i])) #  average speed during interval
                                home_y_start = np.average(self.tracking.home_coords_modified[i][:,1]) # y start average
                                home_y_end = np.average(self.tracking.home_coords_modified[i + poss_cnt][:,1]) # y end average
                                if home_avg_speed_interval > home_avg_speed and home_y_start > home_y_end : # quickly going back to defense
                                    recovery_flag = True
                                    home_recovery.extend(range(i, i + poss_cnt))
                            elif poss_cnt >= recovery_threshold:
                                home_speed_interval = self.tracking.home_speed[i: i + recovery_threshold] 
                                home_avg_speed_interval = np.sum(home_speed_interval) / (recovery_threshold * len(self.tracking.home_speed[i])) #  average speed during interval
                                home_y_start = np.average(self.tracking.home_coords_modified[i][:,1]) # y start average
                                home_y_end = np.average(self.tracking.home_coords_modified[i + recovery_threshold][:,1]) # y end average
                                if home_avg_speed_interval > home_avg_speed and home_y_start > home_y_end : # quickly going back to defense
                                    recovery_flag = True
                                    home_recovery.extend(range(i, i + recovery_threshold))

                    prev_team = "A"    

        #Counter Attack - ensure ball is alive, frames are valid and not used
        home_counter_attack = sorted(list(set(home_counter_attack)))
        away_counter_attack = sorted(list(set(away_counter_attack)))
        home_counter_attack_filtered = []
        away_counter_attack_filtered = [] 
        for c in home_counter_attack:
            if c < len(self.tracking.ball_state) and self.tracking.ball_state[c] == "Alive" and self.tracking.possession_info_raw[c] == "H" and c not in used_frames_home_poss:
                home_counter_attack_filtered.append(c)
        for c in away_counter_attack:
            if c < len(self.tracking.ball_state) and self.tracking.ball_state[c] == "Alive" and self.tracking.possession_info_raw[c] == "A" and c not in used_frames_away_poss:
                away_counter_attack_filtered.append(c)
        
        home_counter_attack_pct = round((len(home_counter_attack_filtered) * 100) / alive_home_poss_len)
        away_counter_attack_pct = round((len(away_counter_attack_filtered) * 100) / alive_away_poss_len)
        used_frames_home_poss.extend(home_counter_attack_filtered)
        used_frames_away_poss.extend(away_counter_attack_filtered)

        #Counter Press - ensure ball is alive, frames are valid and not used
        home_counter_press = sorted(list(set(home_counter_press)))
        away_counter_press = sorted(list(set(away_counter_press)))
        home_counter_press_filtered = []
        away_counter_press_filtered = []
        for c in home_counter_press:
            if c < len(self.tracking.ball_state) and self.tracking.ball_state[c] == "Alive" and  self.tracking.possession_info_raw[c] == "A" and c not in used_frames_home_out_poss:
                home_counter_press_filtered.append(c)
        for c in away_counter_press:
            if c < len(self.tracking.ball_state) and self.tracking.ball_state[c] == "Alive" and  self.tracking.possession_info_raw[c] == "H" and c not in used_frames_away_out_poss:
                away_counter_press_filtered.append(c)

        home_counter_press_pct = round((len(home_counter_press_filtered) * 100) / alive_away_poss_len)
        away_counter_press_pct = round((len(away_counter_press_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_counter_attack_filtered)
        used_frames_away_out_poss.extend(away_counter_attack_filtered)
        
        #Recovery - ensure ball is alive, frames are valid and not used
        home_recovery = sorted(list(set(home_recovery)))
        away_recovery = sorted(list(set(away_recovery)))
        home_recovery_filtered = []
        away_recovery_filtered = []
        for r in home_recovery:
            if r < len(self.tracking.ball_state) and self.tracking.ball_state[r] == "Alive" and  self.tracking.possession_info_raw[r] == "A" and r not in used_frames_home_out_poss:
                home_recovery_filtered.append(r)
        for r in away_recovery:
            if r < len(self.tracking.ball_state) and self.tracking.ball_state[r] == "Alive" and self.tracking.possession_info_raw[r] == "H" and r not in used_frames_away_out_poss:
                away_recovery_filtered.append(r)

        home_recovery_pct = round((len(home_recovery_filtered) * 100) / alive_away_poss_len)
        away_recovery_pct = round((len(away_recovery_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_recovery_filtered)
        used_frames_away_out_poss.extend(away_recovery_filtered)

        #Defensive Transition, Attacking Transition - ensure ball is alive, frames are valid and not used

        #Defensive Transition = counter pressing + recovery
        home_defensive_transition_filtered = sorted(list(set(home_recovery_filtered + home_counter_press_filtered)))
        away_defensive_transition_filtered = sorted(list(set(away_recovery_filtered + away_counter_press_filtered)))
        home_defensive_transition_pct = round((len(home_defensive_transition_filtered) * 100) / alive_away_poss_len)
        away_defensive_transition_pct = round((len(away_defensive_transition_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_defensive_transition_filtered)
        used_frames_away_out_poss.extend(away_defensive_transition_filtered)

        #Attacking Transition Home = Defensive Transition Away, Attacking Transition Away = Defensive Transition Home
        home_attacking_transition_filtered = away_defensive_transition_filtered
        away_attacking_transition_filtered = home_defensive_transition_filtered
        home_attacking_transition_pct = round((len(home_attacking_transition_filtered) * 100) / alive_home_poss_len)
        away_attacking_transition_pct = round((len(away_attacking_transition_filtered) * 100) / alive_away_poss_len)
        used_frames_home_poss.extend(home_attacking_transition_filtered)
        used_frames_away_poss.extend(away_attacking_transition_filtered)
        used_frames_home_poss = sorted(set(list(used_frames_home_poss)))
        used_frames_away_poss = sorted(set(list(used_frames_away_poss)))


         # ---------------- BUILD UP OPP/UNOPP - PROGRESSION - FINAL THIRD - HIGH/MID/LOW PRESS - HIGH/MID/LOW BLOCK ------------------
        garbage_collector_home = []
        garbage_collector_away = []
        home_build_up_unopp = []
        away_build_up_unopp = []
        home_build_up_opp = []
        away_build_up_opp = []

        home_progression = []
        away_progression = []

        home_final_third = []
        away_final_third = []

        home_high_press = []
        away_high_press = []
        home_mid_press = []
        away_mid_press = []
        home_low_press = []
        away_low_press = []

        home_high_block = []
        away_high_block = []
        home_mid_block = []
        away_mid_block = []
        home_low_block = []
        away_low_block = []

        for i,p in enumerate(self.tracking.possession_info_raw):
            if self.tracking.ball_state[i] == "Alive":
                if p == "H": # Home in Possession - Away out of Possession
                    is_classified = False
                    home_ball_y = self.tracking.ball_coords_modified_home[i][1] #y-coordinate ball
                    away_players_y = np.array(sorted(self.tracking.away_coords_modified[i][:,1])[1:]) # exclude goal keeper
                    away_team_length = np.max(away_players_y) - np.min(away_players_y)
                    away_unit_length = away_team_length / 3
                    away_y_offensive_unit = np.average(away_players_y[away_players_y > np.max(away_players_y) - away_unit_length]) # offensive unit y avg
                    away_y_offensive_mid_unit = np.average(away_players_y[away_players_y > np.min(away_players_y) + away_unit_length]) # offensive - mid- unit y avg
                    away_y_avg = np.average(away_players_y)
                    away_dist_to_ball = np.sqrt(np.sum((self.tracking.away_coords[i] - self.tracking.ball_coords[i]) ** 2, axis = 1))

                    if home_ball_y >= y_defensive_down and home_ball_y < y_middle_down: # Ball in Defensive Third Home - Final Third Away   
                        if away_y_offensive_unit > y_middle_down and len(away_dist_to_ball[away_dist_to_ball <= r_opp]) > 0: # Build-up Opposed Home
                            home_build_up_opp.append(i)
                        else:
                            home_build_up_unopp.append(i)
                        
                    elif home_ball_y >= y_middle_down and home_ball_y <= y_middle_up: # Ball in  Middle Third Home - Middle Third Away
                        home_progression.append(i)

                    elif  home_ball_y > y_middle_up and home_ball_y <= y_final_up: # Ball in Final Third Home - Defensive Third Away
                        home_final_third.append(i)
                    
                    
                    if home_ball_y >= y_defensive_down and home_ball_y < y_middle_down_press:
                        if len(away_dist_to_ball[away_dist_to_ball <= r_high]) > 0: 
                            away_high_press.append(i) # High Press Away
                            is_classified = True
                        elif away_y_offensive_unit > y_middle_up and high_block_threshold >= away_team_length: 
                            away_high_block.append(i) # High Block Away
                            is_classified = True
                    
                    if is_classified == False and home_ball_y >= y_middle_down_press and home_ball_y <= y_middle_up: 
                        if len(away_dist_to_ball[away_dist_to_ball <= r_mid]) > 0: 
                            away_mid_press.append(i) # Mid Press Away
                            is_classified = True

                    if is_classified == False and home_ball_y > y_middle_up and home_ball_y <= y_final_up: 
                        if len(away_dist_to_ball[away_dist_to_ball <= r_low]) > 0: 
                            away_low_press.append(i) # Low Press Away
                            is_classified = True

                    if is_classified == False and away_y_avg < y_middle_down_press:
                        if  low_block_threshold >= away_team_length: 
                            away_low_block.append(i) # Low Block Away
                            is_classified = True

                    if is_classified == False and home_ball_y >= y_middle_down_press and away_y_offensive_mid_unit >= y_middle_down_press and away_y_offensive_mid_unit <= y_middle_up: 
                        if mid_block_threshold >= away_team_length: 
                            away_mid_block.append(i) # Mid Block Away
                            is_classified = True

                    if is_classified == False:
                        garbage_collector_away.append(i)

                elif p == "A": # Away in Possession - Home out of Possession
                    is_classified = False
                    away_ball_y = self.tracking.ball_coords_modified_away[i][1] #y-coordinate ball
                    home_players_y =  np.array(sorted(self.tracking.home_coords_modified[i][:,1])[1:]) # exclude goal keeper
                    home_team_length = np.max(home_players_y) - np.min(home_players_y)
                    home_unit_length = home_team_length / 3
                    home_y_offensive_unit = np.average(home_players_y[home_players_y > np.max(home_players_y) - home_unit_length]) # offensive unit y avg
                    home_y_offensive_mid_unit = np.average(home_players_y[home_players_y > np.max(home_players_y) - home_unit_length]) # offensive - mid unit y avg
                    home_y_avg = np.average(home_players_y)
                    home_dist_to_ball = np.sqrt(np.sum((self.tracking.home_coords[i] - self.tracking.ball_coords[i]) ** 2, axis = 1))

                    if away_ball_y >= y_defensive_down and away_ball_y < y_middle_down: # Ball in Defensive Third Away - Final Third Home
                        if home_y_offensive_unit > y_middle_down and len(home_dist_to_ball[home_dist_to_ball <= r_opp]) > 0: # Build-up Opposed Away
                            away_build_up_opp.append(i)
                        else:
                            away_build_up_unopp.append(i)
                        
                    elif away_ball_y >= y_middle_down and away_ball_y <= y_middle_up: # Ball in  Middle Third Away - Middle Third Home
                        away_progression.append(i)

                    elif  away_ball_y > y_middle_up and away_ball_y <= y_final_up: # Ball in Final Third Away - Defensive Third Home
                        away_final_third.append(i)


                    if away_ball_y >= y_defensive_down and away_ball_y < y_middle_down_press:
                        if len(home_dist_to_ball[home_dist_to_ball <= r_high]) > 0: 
                            home_high_press.append(i) # High Press Away
                            is_classified = True
                        elif home_y_offensive_unit > y_middle_up and high_block_threshold >= home_team_length:
                            home_high_block.append(i) # High Block Away
                            is_classified = True
                    
                    if is_classified == False and away_ball_y >= y_middle_down_press and away_ball_y <= y_middle_up: 
                        if len(home_dist_to_ball[home_dist_to_ball <= r_mid]) > 0: 
                            home_mid_press.append(i) # Mid Press Away
                            is_classified = True

                    if  is_classified == False and away_ball_y > y_middle_up and away_ball_y <= y_final_up: 
                        if len(home_dist_to_ball[home_dist_to_ball <= r_low]) > 0: 
                            home_low_press.append(i) # Low Press Away
                            is_classified = True
                     
                    if is_classified == False and home_y_avg < y_middle_down_press: 
                        if low_block_threshold >= home_team_length: 
                            home_low_block.append(i) # Low Block Away
                            is_classified = True
                                                
                    if is_classified == False and away_ball_y >= y_middle_down_press and home_y_offensive_unit >= y_middle_down_press and home_y_offensive_mid_unit <= y_middle_up: 
                        if mid_block_threshold >= home_team_length: 
                            home_mid_block.append(i) # Mid Block Away
                            is_classified = True

                    if is_classified == False:
                        garbage_collector_home.append(i)
        

       
        #Build up Unopposed - ensure ball is alive, frames are valid and not used
        home_build_up_unopp = sorted(list(set(home_build_up_unopp)))
        away_build_up_unopp = sorted(list(set(away_build_up_unopp)))
        home_build_up_unopp_filtered = []
        away_build_up_unopp_filtered = []
        for r in home_build_up_unopp:
            if r not in used_frames_home_poss:
                home_build_up_unopp_filtered.append(r)
        for r in away_build_up_unopp:
            if r not in used_frames_away_poss:
                away_build_up_unopp_filtered.append(r)
        
        home_build_up_unopp_pct = round((len(home_build_up_unopp_filtered) * 100) / alive_home_poss_len)
        away_build_up_unopp_pct = round((len(away_build_up_unopp_filtered) * 100) / alive_away_poss_len)
        used_frames_home_poss.extend(home_build_up_unopp_filtered)
        used_frames_away_poss.extend(away_build_up_unopp_filtered)

        #Build up Opposed - ensure ball is alive, frames are valid and not used
        home_build_up_opp = sorted(list(set(home_build_up_opp)))
        away_build_up_opp = sorted(list(set(away_build_up_opp)))
        home_build_up_opp_filtered = []
        away_build_up_opp_filtered = []
        for r in home_build_up_opp:
            if r not in used_frames_home_poss:
                home_build_up_opp_filtered.append(r)
        for r in away_build_up_opp:
            if r not in used_frames_away_poss:
                away_build_up_opp_filtered.append(r)

        home_build_up_opp_pct = round((len(home_build_up_opp_filtered) * 100) / alive_home_poss_len)
        away_build_up_opp_pct = round((len(away_build_up_opp_filtered) * 100) / alive_away_poss_len)
        used_frames_home_poss.extend(home_build_up_opp_filtered)
        used_frames_away_poss.extend(away_build_up_opp_filtered)
        #Progession - ensure ball is alive, frames are valid and not used
        home_progression = sorted(list(set(home_progression)))
        away_progression = sorted(list(set(away_progression)))
        home_progression_filtered = []
        away_progression_filtered = []
        for r in home_progression:
            if r not in used_frames_home_poss:
                home_progression_filtered.append(r)
        for r in away_progression:
            if r not in used_frames_away_poss:
                away_progression_filtered.append(r)

        home_progression_pct = round((len(home_progression_filtered) * 100) / alive_home_poss_len)
        away_progression_pct = round((len(away_progression_filtered) * 100) / alive_away_poss_len)
        used_frames_home_poss.extend(home_progression_filtered)
        used_frames_away_poss.extend(away_progression_filtered)

        #Final Third - ensure ball is alive, frames are valid and not used
        home_final_third = sorted(list(set(home_final_third)))
        away_final_third = sorted(list(set(away_final_third)))
        home_final_third_filtered = []
        away_final_third_filtered = []
        for r in home_final_third:
            if r not in used_frames_home_poss:
                home_final_third_filtered.append(r)
        for r in away_final_third:
            if r not in used_frames_away_poss:
                away_final_third_filtered.append(r)

        home_final_third_pct = round((len(home_final_third_filtered) * 100) / alive_home_poss_len)
        away_final_third_pct = round((len(away_final_third_filtered) * 100) / alive_away_poss_len)
        used_frames_home_poss.extend(home_final_third_filtered)
        used_frames_away_poss.extend(away_final_third_filtered)

        # High Press - ensure ball is alive, frames are valid and not used
        home_high_press= sorted(list(set(home_high_press)))
        away_high_press = sorted(list(set(away_high_press)))
        home_high_press_filtered = []
        away_high_press_filtered = []
        for r in home_high_press:
            if r not in used_frames_home_out_poss:
                home_high_press_filtered.append(r)
        for r in away_high_press:
            if r not in used_frames_away_out_poss:
                away_high_press_filtered.append(r)
        
        home_high_press_pct = round((len(home_high_press_filtered) * 100) / alive_away_poss_len)
        away_high_press_pct = round((len(away_high_press_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_high_press_filtered)
        used_frames_away_out_poss.extend(away_high_press_filtered)

        # Mid Press - ensure ball is alive, frames are valid and not used
        home_mid_press= sorted(list(set(home_mid_press)))
        away_mid_press = sorted(list(set(away_mid_press)))
        home_mid_press_filtered = []
        away_mid_press_filtered = []
        for r in home_mid_press:
            if r not in used_frames_home_out_poss:
                home_mid_press_filtered.append(r)
        for r in away_mid_press:
            if r not in used_frames_away_out_poss:
                away_mid_press_filtered.append(r)
        
        home_mid_press_pct = round((len(home_mid_press_filtered) * 100) / alive_away_poss_len)
        away_mid_press_pct = round((len(away_mid_press_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_mid_press_filtered)
        used_frames_away_out_poss.extend(away_mid_press_filtered)

        # Low Press - ensure ball is alive, frames are valid and not used
        home_low_press= sorted(list(set(home_low_press)))
        away_low_press = sorted(list(set(away_low_press)))
        home_low_press_filtered = []
        away_low_press_filtered = []
        for r in home_low_press:
            if r not in used_frames_home_out_poss:
                home_low_press_filtered.append(r)
        for r in away_low_press:
            if r not in used_frames_away_out_poss:
                away_low_press_filtered.append(r)
        
        home_low_press_pct = round((len(home_low_press_filtered) * 100) / alive_away_poss_len)
        away_low_press_pct = round((len(away_low_press_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_low_press_filtered)
        used_frames_away_out_poss.extend(away_low_press_filtered)

        # High Block - ensure ball is alive, frames are valid and not used
        home_high_block= sorted(list(set(home_high_block)))
        away_high_block = sorted(list(set(away_high_block)))
        home_high_block_filtered = []
        away_high_block_filtered = []
        for r in home_high_block:
            if r not in used_frames_home_out_poss:
                home_high_block_filtered.append(r)
        for r in away_high_block:
            if r not in used_frames_away_out_poss:
                away_high_block_filtered.append(r)
        
        home_high_block_pct = round((len(home_high_block_filtered) * 100) / alive_away_poss_len)
        away_high_block_pct = round((len(away_high_block_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_high_block_filtered)
        used_frames_away_out_poss.extend(away_high_block_filtered)
    
        # Mid Block - ensure ball is alive, frames are valid and not used
        home_mid_block= sorted(list(set(home_mid_block)))
        away_mid_block = sorted(list(set(away_mid_block)))
        home_mid_block_filtered = []
        away_mid_block_filtered = []
        for r in home_mid_block:
            if r not in used_frames_home_out_poss:
                home_mid_block_filtered.append(r)
        for r in away_mid_block:
            if r not in used_frames_away_out_poss:
                away_mid_block_filtered.append(r)
        
        home_mid_block_pct = round((len(home_mid_block_filtered) * 100) / alive_away_poss_len)
        away_mid_block_pct = round((len(away_mid_block_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_mid_block_filtered)
        used_frames_away_out_poss.extend(away_mid_block_filtered)

        # Low Block - ensure ball is alive, frames are valid and not used
        home_low_block= sorted(list(set(home_low_block)))
        away_low_block = sorted(list(set(away_low_block)))
        home_low_block_filtered = []
        away_low_block_filtered = []
        for r in home_low_block:
            if r not in used_frames_home_out_poss:
                home_low_block_filtered.append(r)
        for r in away_low_block:
            if r not in used_frames_away_out_poss:
                away_low_block_filtered.append(r)
        
        home_low_block_pct = round((len(home_low_block_filtered) * 100) / alive_away_poss_len)
        away_low_block_pct = round((len(away_low_block_filtered) * 100) / alive_home_poss_len)
        used_frames_home_out_poss.extend(home_low_block_filtered)
        used_frames_away_out_poss.extend(away_low_block_filtered)

        home_in_phases = [home_build_up_unopp_pct, home_build_up_opp_pct, home_progression_pct, home_final_third_pct, home_long_ball_pct, home_attacking_transition_pct,
                          home_counter_attack_pct, home_set_piece_pct]
        home_out_phases = [home_high_press_pct, home_mid_press_pct, home_low_press_pct, home_high_block_pct, home_mid_block_pct, home_low_block_pct, home_recovery_pct,
                          home_defensive_transition_pct, home_counter_press_pct]
        away_in_phases = [away_build_up_unopp_pct, away_build_up_opp_pct, away_progression_pct, away_final_third_pct, away_long_ball_pct, away_attacking_transition_pct,
                          away_counter_attack_pct, away_set_piece_pct]
        away_out_phases = [away_high_press_pct, away_mid_press_pct, away_low_press_pct, away_high_block_pct, away_mid_block_pct, away_low_block_pct, away_recovery_pct,
                          away_defensive_transition_pct, away_counter_press_pct]
        
        return home_in_phases, home_out_phases, away_in_phases, away_out_phases
        
    def __get_distance_to_goal(self,x_locations_col, y_locations_col, x_goal = 0, y_goal = 52.5):
        # Calculate the distance of shots to the goal
        distance_to_goal = ((x_goal - x_locations_col) ** 2 + (y_goal - y_locations_col) ** 2) ** 0.5
        return distance_to_goal
    def __get_angle(self,x_locations_col, y_locations_col, y_goal = 52.5, x_goal_right = 3.66, x_goal_left = -3.66):
        # Three points
        a = ((x_locations_col - x_goal_right) ** 2 + (y_locations_col - y_goal) ** 2) ** 0.5 
        b = ((x_locations_col - x_goal_left) ** 2 + (y_locations_col - y_goal) ** 2) ** 0.5 
        c = ((x_goal_left - x_goal_right) ** 2 + (y_goal - y_goal) ** 2) ** 0.5 
        # Calculate the angle opposite to side c
        C = np.arccos((a**2 + b**2 - c**2) / (2*a*b))

        # Convert the angle from radians to degrees
        C_degrees = np.degrees(C)
        return C_degrees
    def __get_goal_info(self,outcome_additional):
        outcome_additional = outcome_additional.values
        outcome_additional[np.where(outcome_additional != "goal")] = "no_goal"
        return outcome_additional
    def __event_to_tracking_coords(self, tracking_obj, adjusted_shot_frames, player_ids_event_shooter, player_ids_match, team_ids, home_id, away_id):
        # Helper function to retrieve defender and goal keeper coordinates at the time of shots taken.
        defender_team_coords = []
        defender_keeper_coords = []
        for i,a in enumerate(adjusted_shot_frames):
            shooter = player_ids_event_shooter[i]
            shooter_idx = player_ids_match.index(shooter)
            shooter_team = team_ids[shooter_idx]
            temp_defender_team_coords = []
            if shooter_team == home_id:
                if a < len(tracking_obj.away_coords_modified):
                    temp_defender_team_coords = -np.array(tracking_obj.away_coords_modified[a])
                else:
                    temp_defender_team_coords = -np.array(tracking_obj.away_coords_modified[len(tracking_obj.away_coords_modified)-1])
            elif shooter_team == away_id:
                if a < len(tracking_obj.home_coords_modified):
                    temp_defender_team_coords = -np.array(tracking_obj.home_coords_modified[a])
                else:
                    temp_defender_team_coords = -np.array(tracking_obj.home_coords_modified[len(tracking_obj.home_coords_modified)-1])
            else:
                continue
            keeper_idx = np.argmax(temp_defender_team_coords[:,1])
            defender_keeper_coords.append(temp_defender_team_coords[keeper_idx])
            defender_team_coords.append(temp_defender_team_coords)
        return np.array(defender_team_coords), np.array(defender_keeper_coords)

    def __area(self,x1, y1, x2, y2, x3, y3):
        # A utility function to calculate area
        # of triangle formed by (x1, y1),
        # (x2, y2) and (x3, y3)
    
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                    + x3 * (y1 - y2)) / 2.0)
    
    def __isInside(self,defenders, x1, y1, x2, y2, x3, y3):
        # A function to check whether point P(x, y)
        # lies inside the triangle formed by
        # A(x1, y1), B(x2, y2) and C(x3, y3)
        
        # Calculate area of triangle ABC
        A = self.__area(x1, y1, x2, y2, x3, y3)
        obst_def = []

        for d in defenders:
            x = d[0]
            y = d[1]
        
            # Calculate area of triangle PBC
            A1 = self.__area(x, y, x2, y2, x3, y3)
            
            # Calculate area of triangle PAC
            A2 = self.__area(x1, y1, x, y, x3, y3)
            
            # Calculate area of triangle PAB
            A3 = self.__area(x1, y1, x2, y2, x, y)
            
            # Check if sum of A1, A2 and A3
            # is same as A
            if(abs(A - (A1 + A2 + A3) )<= 0.05):
                obst_def.append(d)
                
        if len(obst_def) == 0:
            return 1
        else:
            return len(obst_def)
    
    def __xG_preprocess_data(self, match_objs, event_objs, tracking_objs):
        """ xG preprocess calculation.
        Parameters
        ---------- 
        match_objs: list
            match objects of the training set.
        event_objs: list
            event objects of the training set.
        tracking_objs: list
            tracking objects of the training set.

        Return 
        ---------- 
        xG_dfs: list
            recorded shots at the given matches.
        team_ids_shots: list
            ids of the teams that took the shots.
        """
        # PRE-PROCESS DATA
        xG_dfs = []
        team_ids_shots = []
        for i in range(len(match_objs)):
            tracking_obj = tracking_objs[i]
            match_obj = match_objs[i]
            event_obj = event_objs[i]

            # get shot rows from event data
            shots_df = event_obj.events[event_obj.events.event.isin(["attempt_at_goal"])][["team_id", "match_run_time_in_ms", "from_player_id", "x_location_start_mirrored","y_location_start_mirrored", "pressure", "body_type", "origin", "outcome_additional"]]
            shots_df = shots_df.reset_index()
            team_ids_shots.extend(list(shots_df["team_id"].values))
            # adjust shot locations according to tracking data
            x_location_start_mirrored = shots_df["x_location_start_mirrored"].values.copy()
            y_location_start_mirrored = shots_df["y_location_start_mirrored"].values.copy()
            shots_df["x_location_start_mirrored"] = - np.round((y_location_start_mirrored - 0.5) * 68)
            shots_df["y_location_start_mirrored"] = np.round((x_location_start_mirrored - 0.5) * 105)

            # calculate ball distance to goal
            distance_to_goal = self.__get_distance_to_goal(shots_df["x_location_start_mirrored"], shots_df["y_location_start_mirrored"])
            # calculate angle between ball and the goal line
            angle = self.__get_angle(shots_df["x_location_start_mirrored"], shots_df["y_location_start_mirrored"])
            # check if the shot is goal
            outcome_additional = self.__get_goal_info(shots_df["outcome_additional"])

            shots_df = shots_df.astype({'from_player_id': 'int64'})
            shots_df["distance_to_goal"] = distance_to_goal
            shots_df["angle"] = angle
            shots_df["outcome_additional"] = outcome_additional
            shots_df = shots_df.rename(columns={"outcome_additional":"goal_info"})

            # Find the shot frame from tracking data
            shots_df["match_run_time_in_ms"] = shots_df["match_run_time_in_ms"] // 40 # 1 frame = 40 ms
            shots_df = shots_df.rename(columns={"match_run_time_in_ms":"frame"})
            match_start_frame = list(tracking_obj.tracking["frame"])[0]
            shot_frames = np.array(shots_df["frame"]) + match_start_frame
            adjusted_shot_frames = []
            for p in shot_frames:
                old_p = p
                if old_p > match_obj.meta["Phase2StartFrame"][0]:
                    p -= match_obj.meta["Phase2StartFrame"][0] - match_obj.meta["Phase1EndFrame"][0]

                if tracking_obj.match_len > 2: # Over Time
                    if old_p > match_obj.meta["Phase3StartFrame"][0]: # end of match - start of first OT
                        p-= match_obj.meta["Phase3StartFrame"][0] - match_obj.meta["Phase2EndFrame"][0]
                    if old_p > match_obj.meta["Phase4StartFrame"][0]: # start of first OT - start of second OT
                        p-= match_obj.meta["Phase4StartFrame"][0] - match_obj.meta["Phase3EndFrame"][0]
                    
                    if tracking_obj.match_len > 4: # Penalties
                        if old_p > match_obj.meta["Phase5StartFrame"][0]: # end of match - start of first OT
                            p-= match_obj.meta["Phase5StartFrame"][0] - match_obj.meta["Phase4EndFrame"][0]
                
                if p - match_start_frame < len(tracking_obj.dirs):
                    adjusted_shot_frames.append(p - match_start_frame)
                else:
                    adjusted_shot_frames.append(len(tracking_obj.dirs) - 1)
            shots_df["frame"] = np.array(adjusted_shot_frames)
            
            # Get the appropriate defender team coords 
            player_ids_event_shooter = list(shots_df["from_player_id"])
            team_ids = list(match_obj.lineups["team_id"])
            player_ids_match = list(match_obj.lineups["player_id"])
            home_id = match_obj.meta["home_team_id"][0]
            away_id = match_obj.meta["away_team_id"][0]
            defender_team_coords, defender_keeper_coords = self.__event_to_tracking_coords(tracking_obj,adjusted_shot_frames,player_ids_event_shooter, 
                                                                                           player_ids_match, team_ids, home_id, away_id)
            # Goal keeper location
            shots_df["x_location_goal_keeper_mirrored"] = defender_keeper_coords[:,0]
            shots_df["y_location_goal_keeper_mirrored"] = defender_keeper_coords[:,1]
            
            number_of_obstructing_defenders = []
            for idx,row in shots_df.iterrows():
                temp_defender_team = defender_team_coords[idx]
                obstructing_defenders = self.__isInside(defenders = temp_defender_team, x1 = row["x_location_start_mirrored"], y1 =row["y_location_start_mirrored"], x2 = 3.66, y2 = 52.5, x3 = -3.66, y3 = 52.5)
                number_of_obstructing_defenders.append(obstructing_defenders)
            
            shots_df["number_of_obstructing_defenders"] = number_of_obstructing_defenders

            #delete unnecessary columns
            del shots_df['from_player_id']
            del shots_df['frame']
            del shots_df['index']
            shots_df = shots_df.dropna(subset=['x_location_start_mirrored'])
            xG_dfs.append(shots_df)
        
        return xG_dfs, team_ids_shots
    
    def xG_model(self, efi_objs):
        """ xG model calculation.
        Parameters
        ---------- 
        efi_objs: list
            EFI objects used in the training phase of xG model.
        Return round(sum(home_xG),2), round(sum(away_xG),2), sum(home_true), sum(away_true), probs, test_df, y_test
        ---------- 
        home_xG: float
            xG of the home team.
        away_xG: float
            xG of the away team.
        home_true: integer
            actual score of the home team.
        away_true: integer
            actual score of the away team.
        probs: list
            probabilities assigned to each shot.
        test_df: 
            shots in the evaluated match.
        y_test:
            goal information of the shots in the evaluated match.
        """
        match_objs = []
        event_objs = []
        tracking_objs = []
        for e in efi_objs:
            match_objs.append(e.match)
            event_objs.append(e.events)
            tracking_objs.append(e.tracking)
        
        train_dfs, _ = self.__xG_preprocess_data(match_objs, event_objs, tracking_objs)
        train_df = pd.concat(train_dfs, axis=0)
        test_df, test_team_ids = self.__xG_preprocess_data([self.match], [self.events], [self.tracking])
        test_df = test_df[0]
        home_id = self.match.meta["home_team_id"][0]
        away_id = self.match.meta["away_team_id"][0]

        #Train, Validation and Test sets
        y_train = train_df["goal_info"].values
        y_train[y_train == "no_goal"] = 0
        y_train[y_train == "goal"] = 1
        X_train = train_df.loc[:, train_df.columns != 'goal_info']

        y_test = test_df["goal_info"].values
        y_test[y_test == "no_goal"] = 0
        y_test[y_test == "goal"] = 1
        X_test = test_df.loc[:, test_df.columns != 'goal_info']
        
        # One-hot encode the categorical features
        cat_cols = ['pressure', 'body_type', 'origin']
        num_cols = []
        for c in X_train.columns:
            if c not in cat_cols:
                num_cols.append(c)
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(X_train[cat_cols])
        X_train_ohe = ohe.transform(X_train[cat_cols]).toarray()
        X_test_ohe = ohe.transform(X_test[cat_cols]).toarray()
        
       # Concatenate the one-hot encoded features with the numerical features
        X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names_out(cat_cols))
        X_test_ohe_df = pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names_out(cat_cols)) 
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        X_train = pd.concat([X_train.drop(cat_cols, axis=1), X_train_ohe_df], axis=1)
        X_test = pd.concat([X_test.drop(cat_cols, axis=1), X_test_ohe_df], axis=1)

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Scale the numerical features
        scaler = StandardScaler()
        scaler.fit(X_train[num_cols])
        X_train[num_cols] = scaler.transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        X_train = X_train.values
        X_test = X_test.values

        # Define the hyperparameters for each model
        logreg_params = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10], 'solver': ['liblinear']}
        rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
        gbt_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
        svm_params = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'probability': [True]}

        # Define the models
        logreg = LogisticRegression()
        rf = RandomForestClassifier()
        gbt = GradientBoostingClassifier()
        svm = SVC(verbose = True)

        # Train and optimize the models using cross-validation and validation set
        models = {'Logistic Regression': (logreg, logreg_params),
          'Random Forest': (rf, rf_params),
          'Gradient Boosted Trees': (gbt, gbt_params),
          'SVM': (svm, svm_params)}
        models_best = {}
        accuracy_scores = []
        for model_name, (model, params) in models.items():
            #print('Training and optimizing', model_name)
            grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            #print('Best parameters:', grid_search.best_params_)
            #print('Training accuracy:', best_model.score(X_train, y_train))
            #print('Validation accuracy:', best_model.score(X_val, y_val))
            # Test the best model
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(model_name, 'test accuracy:', test_accuracy)
            accuracy_scores.append(test_accuracy)
            #print()
            # Store the best model
            models_best[model_name] = best_model, grid_search.best_params_
        
        # Calculate the weighted probability of goal for each shot
        weights = {'Logistic Regression': accuracy_scores[0],
                'Random Forest': accuracy_scores[1],
                'Gradient Boosted Trees': accuracy_scores[2],
                'SVM': accuracy_scores[3]}
        
        # The weighted probability of goal for each shot
        probs = []
        for i in range(len(X_test)):
            prob_sum = 0
            for model_name, (model, params) in models_best.items():
                prob = model.predict_proba(X_test[[i]])[0][1]
                prob_sum += weights[model_name] * prob
            probs.append(prob_sum / sum(weights.values()))
        
        # Calculate the xG values of both teams
        home_xG = []
        away_xG = []
        home_true = []
        away_true = []
        for i,p in enumerate(probs):
            if test_team_ids[i] == home_id:
                home_xG.append(p)
                home_true.append(y_test[i])
            elif test_team_ids[i] == away_id:
                away_xG.append(p)
                away_true.append(y_test[i])
        return round(sum(home_xG),2), round(sum(away_xG),2), sum(home_true), sum(away_true), probs, test_df, y_test
    
    def line_breaks(self, event_x_final_up = 1, y_event_ch_left = 1, y_event_ch_right = 0, #pitch frame coordinates
                    tracking_x_range = 68, tracking_y_range = 105, #pitch frame coordinates
                    over_height_threshold = 1.67, over_duration = 27, frame_gap = 54): #over threshold
        
        """ Line breaks calculation.
        Parameters
        ---------- 
        over_height_threshold: float, optional
            required minimum height to ensure that the line was broken in the over direction category.
        over_duration: integer, optional
            required minimum frame interval to ensure that the line was broken in the over direction category.
        frame_gap: integer, optional
            number of frames used to add on the syncronized frames.
        Return round(sum(home_xG),2), round(sum(away_xG),2), sum(home_true), sum(away_true), probs, test_df, y_test
        ---------- 
        home_xG: float
            xG of the home team.
        away_xG: float
            xG of the away team.
        home_true: integer
            actual score of the home team.
        away_true: integer
            actual score of the away team.
        probs: list
            probabilities assigned to each shot.
        test_df: 
            shots in the evaluated match.
        y_test:
            goal information of the shots in the evaluated match.
        """

        # PASS and CROSS
        # Directions
        home_through = []
        home_around = []
        home_over = []
        away_through = []
        away_around = []
        away_over = []
        # Distribution Types
        home_pass = []
        home_cross = []
        home_prog = []
        away_pass = []
        away_cross = []
        away_prog = []
        
        # find passes and do pre-proccessing
        lb_pass_events = self.events.events[self.events.events["event"].isin(["pass", "cross"])][['event', 'x_location_start',"x_location_end", "y_location_start", "y_location_end", 'x_location_start_mirrored', 'x_location_end_mirrored', 'y_location_end_mirrored', "match_run_time_in_ms", "from_player_id", "to_player_id"]]
        lb_pass_events = lb_pass_events.dropna(subset = ["from_player_id"])
        lb_pass_events = lb_pass_events.astype({'from_player_id': 'int64'})
        lb_pass_events["match_run_time_in_ms"] = lb_pass_events["match_run_time_in_ms"] // 40 # 1 frame = 40 ms
        lb_pass_events = lb_pass_events.rename(columns={"match_run_time_in_ms":"frame"})
        lb_pass_events = lb_pass_events[lb_pass_events["x_location_end_mirrored"] <= event_x_final_up] # inside pitch
        lb_pass_events = lb_pass_events[lb_pass_events["y_location_end_mirrored"] <= y_event_ch_left] # inside pitch
        lb_pass_events = lb_pass_events[lb_pass_events["y_location_end_mirrored"] >= y_event_ch_right] #inside pitch
        lb_pass_events = lb_pass_events[lb_pass_events["x_location_start_mirrored"] <= lb_pass_events["x_location_end_mirrored"]] # only forward passes
        lb_pass_events = lb_pass_events[lb_pass_events['to_player_id'].notnull()] # only successful passes
        lb_pass_events = lb_pass_events.astype({'to_player_id': 'int64'})
        lb_pass_events["x_location_start_tracking"] = (lb_pass_events["y_location_start"] - 0.5) * tracking_x_range # to tracking coords 
        lb_pass_events["y_location_start_tracking"] = - (lb_pass_events["x_location_start"] - 0.5) * tracking_y_range # to tracking coords 
        lb_pass_events["x_location_end_tracking"] = (lb_pass_events["y_location_end"] - 0.5) * tracking_x_range # to tracking coords 
        lb_pass_events["y_location_end_tracking"] = - (lb_pass_events["x_location_end"] - 0.5) * tracking_y_range # to tracking coords 
        lb_pass_events = lb_pass_events.astype({'x_location_start_tracking': 'int64'})
        lb_pass_events = lb_pass_events.astype({'y_location_start_tracking': 'int64'})
        lb_pass_events = lb_pass_events.astype({'x_location_end_tracking': 'int64'})
        lb_pass_events = lb_pass_events.astype({'y_location_end_tracking': 'int64'})

        match_start_frame = list(self.tracking.tracking["frame"])[0]
        pass_frames = np.array(lb_pass_events["frame"]) + match_start_frame
        adjusted_pass_frames = []
        #adjust frames for passes
        for p in pass_frames:
            old_p = p
            if old_p > self.match.meta["Phase2StartFrame"][0]:
                p -= self.match.meta["Phase2StartFrame"][0] - self.match.meta["Phase1EndFrame"][0]

            if self.tracking.match_len > 2: # Over Time
                if old_p > self.match.meta["Phase3StartFrame"][0]: # end of match - start of first OT
                    p-= self.match.meta["Phase3StartFrame"][0] - self.match.meta["Phase2EndFrame"][0]
                if old_p > self.match.meta["Phase4StartFrame"][0]: # start of first OT - start of second OT
                    p-= self.match.meta["Phase4StartFrame"][0] - self.match.meta["Phase3EndFrame"][0]
                
                if self.tracking.match_len > 4: # Penalties
                    if old_p > self.match.meta["Phase5StartFrame"][0]: # end of match - start of first OT
                        p-= self.match.meta["Phase5StartFrame"][0] - self.match.meta["Phase4EndFrame"][0]
            
            adjusted_pass_frames.append(p - match_start_frame)

        lb_pass_events["frame"] = adjusted_pass_frames
        event_type = list(lb_pass_events["event"])
        player_ids_event_passer = list(lb_pass_events["from_player_id"])
        team_ids = list(self.match.lineups["team_id"])
        player_ids_match = list(self.match.lineups["player_id"])
        home_id = self.match.meta["home_team_id"][0]
        away_id = self.match.meta["away_team_id"][0]
        lb_pass_events["direction"] = 0
        lb_pass_events["team_id"] = 0
        lb_pass_events["is_line_break"] = 0
        lb_pass_events = lb_pass_events.reset_index(drop=True)
        for i,f in enumerate(adjusted_pass_frames): 
            passer = player_ids_event_passer[i]
            passer_idx = player_ids_match.index(passer)
            passer_team = team_ids[passer_idx]
            lb_pass_events.iloc[i, -2] = passer_team
            # Pass Location
            pass_start_x = lb_pass_events["x_location_start_tracking"].values[i]
            pass_start_y = lb_pass_events["y_location_start_tracking"].values[i]
            pass_end_x = lb_pass_events["x_location_end_tracking"].values[i]
            pass_end_y = lb_pass_events["y_location_end_tracking"].values[i]
            if passer_team == home_id: # Home Successful Pass
                if f < len(self.tracking.dirs) and self.tracking.dirs[f] == -1 : # Home Keeper Up - Away Keeper Down
                    # Away Team Units
                    sorted_indices = []
                    temp_away_coords = []
                    if len(self.tracking.away_coords) > f + frame_gap:
                        sorted_indices = np.argsort(self.tracking.away_coords[f + frame_gap], axis = 0)
                        temp_away_coords = self.tracking.away_coords[f + frame_gap][sorted_indices[:,1]] # sort by y coords
                    else:
                        sorted_indices = np.argsort(self.tracking.away_coords[f], axis = 0)
                        temp_away_coords = self.tracking.away_coords[f][sorted_indices[:,1]] # sort by y coords

                    temp_away_coords_x = temp_away_coords[1:,0]
                    temp_away_coords_y = temp_away_coords[1:,1]
                    away_deepest = np.min(temp_away_coords_y) # deepest player
                    away_farthest = np.max(temp_away_coords_y) # farthest player
                    away_interval = (away_farthest - away_deepest) / 3 # y - interval 
                    away_offensive_down = away_farthest - away_interval
                    away_defensive_up = away_deepest + away_interval
                    away_attacking_unit = temp_away_coords_y[(temp_away_coords_y > away_offensive_down) & (temp_away_coords_y <= away_farthest)] # attacking unit 
                    away_mid_unit = temp_away_coords_y[(temp_away_coords_y >= away_defensive_up) & (temp_away_coords_y <= away_offensive_down)] # midfield unit 
                    away_defensive_unit = temp_away_coords_y[(temp_away_coords_y >= away_deepest) & (temp_away_coords_y < away_defensive_up)]  # defensive unit 
                    away_attacking_unit_x = temp_away_coords_x[(temp_away_coords_y > away_offensive_down) & (temp_away_coords_y <= away_farthest)] # attacking unit 
                    away_mid_unit_x = temp_away_coords_x[(temp_away_coords_y >= away_defensive_up) & (temp_away_coords_y <= away_offensive_down)] # midfield unit 
                    away_defensive_unit_x = temp_away_coords_x[(temp_away_coords_y >= away_deepest) & (temp_away_coords_y < away_defensive_up)]  # defensive unit 
                    away_attacking_unit_x_min = 0
                    away_attacking_unit_x_max = 0
                    away_mid_unit_x_min = 0
                    away_mid_unit_x_max = 0
                    away_defensive_unit_x_min = 0
                    away_defensive_unit_x_max = 0
                    # Ensure existince of enough players in a unit
                    if len(away_attacking_unit) > 1:
                        away_attacking_unit = np.min(away_attacking_unit)
                        away_attacking_unit_x_min = np.min(away_attacking_unit_x)
                        away_attacking_unit_x_max = np.max(away_attacking_unit_x)
                    else:
                        away_attacking_unit = pass_start_y + 1
                    if len(away_mid_unit) > 1:
                        away_mid_unit = np.min(away_mid_unit)
                        away_mid_unit_x_min = np.min(away_mid_unit_x)
                        away_mid_unit_x_max = np.max(away_mid_unit_x)
                    else:
                        away_mid_unit = pass_start_y + 1
                    if len(away_defensive_unit) > 1:
                        away_defensive_unit = np.min(away_defensive_unit)
                        away_defensive_unit_x_min = np.min(away_defensive_unit_x)
                        away_defensive_unit_x_max = np.max(away_defensive_unit_x)
                    else:
                        away_defensive_unit = pass_start_y + 1

                    if pass_start_y >= away_attacking_unit: # check attacking unit break
                        if pass_end_y < away_attacking_unit: # attacking unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                home_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                home_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                home_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < away_attacking_unit_x_min or pass_end_x > away_attacking_unit_x_max: # around
                                home_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0


                    elif pass_start_y < away_attacking_unit and pass_start_y >= away_mid_unit: # check midfield unit break
                        if pass_end_y < away_mid_unit: # midfield unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                home_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                home_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                home_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < away_mid_unit_x_min or pass_end_x > away_mid_unit_x_max: # around
                                home_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

                    elif pass_start_y < away_mid_unit and pass_start_y >= away_defensive_unit: # check defensive unit break
                        if pass_end_y < away_defensive_unit: # defensive unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                home_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                home_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                home_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < away_defensive_unit_x_min or pass_end_x > away_defensive_unit_x_max: # around
                                home_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

                elif f < len(self.tracking.dirs) and self.tracking.dirs[f] == 1 : # Home Keeper Down - Away Keeper Up
                    # Away Team Units
                    sorted_indices = []
                    temp_away_coords = []
                    if len(self.tracking.away_coords) > f + frame_gap:
                        sorted_indices = np.argsort(self.tracking.away_coords[f + frame_gap], axis = 0)
                        temp_away_coords = self.tracking.away_coords[f + frame_gap][sorted_indices[:,1]] # sort by y coords
                    else:
                        sorted_indices = np.argsort(self.tracking.away_coords[f], axis = 0)
                        temp_away_coords = self.tracking.away_coords[f][sorted_indices[:,1]] # sort by y coords

                    temp_away_coords_x = temp_away_coords[:-1,0]
                    temp_away_coords_y = temp_away_coords[:-1,1]
                    away_deepest = np.max(temp_away_coords_y) # deepest player
                    away_farthest = np.min(temp_away_coords_y) # farthest player
                    away_interval = (away_deepest - away_farthest) / 3 # y - interval 
                    away_offensive_down = away_farthest + away_interval
                    away_defensive_up = away_deepest - away_interval
                    away_attacking_unit = temp_away_coords_y[(temp_away_coords_y < away_offensive_down) & (temp_away_coords_y >= away_farthest)] # attacking unit 
                    away_mid_unit = temp_away_coords_y[(temp_away_coords_y <= away_defensive_up) & (temp_away_coords_y >= away_offensive_down)] # midfield unit 
                    away_defensive_unit = temp_away_coords_y[(temp_away_coords_y <= away_deepest) & (temp_away_coords_y > away_defensive_up)] # defensive unit 
                    away_attacking_unit_x = temp_away_coords_x[(temp_away_coords_y < away_offensive_down) & (temp_away_coords_y >= away_farthest)] # attacking unit 
                    away_mid_unit_x = temp_away_coords_x[(temp_away_coords_y <= away_defensive_up) & (temp_away_coords_y >= away_offensive_down)] # midfield unit 
                    away_defensive_unit_x = temp_away_coords_x[(temp_away_coords_y <= away_deepest) & (temp_away_coords_y > away_defensive_up)] # defensive unit 
                    away_attacking_unit_x_min = 0
                    away_attacking_unit_x_max = 0
                    away_mid_unit_x_min = 0
                    away_mid_unit_x_max = 0
                    away_defensive_unit_x_min = 0
                    away_defensive_unit_x_max = 0
                    # Ensure existince of enough players in a unit
                    if len(away_attacking_unit) > 1:
                        away_attacking_unit = np.max(away_attacking_unit)
                        away_attacking_unit_x_min = np.min(away_attacking_unit_x)
                        away_attacking_unit_x_max = np.max(away_attacking_unit_x)
                    else:
                        away_attacking_unit = pass_start_y - 1
                    if len(away_mid_unit) > 1:
                        away_mid_unit = np.max(away_mid_unit)
                        away_mid_unit_x_min = np.min(away_attacking_unit_x)
                        away_mid_unit_x_max = np.max(away_attacking_unit_x)
                    else:
                        away_mid_unit = pass_start_y - 1
                    if len(away_defensive_unit) > 1:
                        away_defensive_unit = np.max(away_defensive_unit)
                        away_defensive_unit_x_min = np.min(away_defensive_unit_x)
                        away_defensive_unit_x_max = np.max(away_defensive_unit_x)
                    else:
                        away_defensive_unit = pass_start_y - 1

                    if pass_start_y <= away_attacking_unit: # check attacking unit break
                        if pass_end_y > away_attacking_unit: # attacking unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                home_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                home_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                home_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < away_attacking_unit_x_min or pass_end_x > away_attacking_unit_x_max: # around
                                home_around.append(f) 
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f) 
                                lb_pass_events.iloc[i,-3] = 0

                    elif pass_start_y > away_attacking_unit and pass_start_y <= away_mid_unit: # check midfield unit break
                        if pass_end_y > away_mid_unit: # midfield unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                home_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                home_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                home_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < away_defensive_unit_x_min or pass_end_x > away_defensive_unit_x_max: # around
                                home_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

                    elif pass_start_y > away_mid_unit and pass_start_y <= away_defensive_unit: # check defensive unit break
                        if pass_end_y > away_defensive_unit: # defensive unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                home_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                home_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                home_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < away_defensive_unit_x_min or pass_end_x > away_defensive_unit_x_max: # around
                                home_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

            elif passer_team == away_id: # Away Successful Pass
                if f < len(self.tracking.dirs) and self.tracking.dirs[f] == -1: # Home Keeper Up - Away Keeper Down
                    # Home Team Units
                    sorted_indices = []
                    temp_home_coords = []
                    if len(self.tracking.home_coords) > f + frame_gap:
                        sorted_indices = np.argsort(self.tracking.home_coords[f + frame_gap], axis = 0)
                        temp_home_coords = self.tracking.home_coords[f + frame_gap][sorted_indices[:,1]] # sort by y coords
                    else:
                        sorted_indices = np.argsort(self.tracking.home_coords[f], axis = 0)
                        temp_home_coords = self.tracking.home_coords[f][sorted_indices[:,1]] # sort by y coords

                    temp_home_coords_x = temp_home_coords[:-1,0]
                    temp_home_coords_y = temp_home_coords[:-1,1]
                    home_deepest = np.max(temp_home_coords_y) # deepest player
                    home_farthest = np.min(temp_home_coords_y) # farthest player
                    home_interval = (home_deepest - home_farthest) / 3 # y - interval 
                    home_offensive_down = home_farthest + home_interval
                    home_defensive_up = home_deepest - home_interval
                    home_attacking_unit = temp_home_coords_y[(temp_home_coords_y < home_offensive_down) & (temp_home_coords_y >= home_farthest)] # attacking unit 
                    home_mid_unit = temp_home_coords_y[(temp_home_coords_y <= home_defensive_up) & (temp_home_coords_y >= home_offensive_down)] # midfield unit 
                    home_defensive_unit = temp_home_coords_y[(temp_home_coords_y <= home_deepest) & (temp_home_coords_y > home_defensive_up)] # defensive unit 
                    home_attacking_unit_x = temp_home_coords_x[(temp_home_coords_y < home_offensive_down) & (temp_home_coords_y >= home_farthest)] # attacking unit 
                    home_mid_unit_x = temp_home_coords_x[(temp_home_coords_y <= home_defensive_up) & (temp_home_coords_y >= home_offensive_down)] # midfield unit 
                    home_defensive_unit_x = temp_home_coords_x[(temp_home_coords_y <= home_deepest) & (temp_home_coords_y > home_defensive_up)] # defensive unit 
                    home_attacking_unit_x_min = 0
                    home_attacking_unit_x_max = 0
                    home_mid_unit_x_min = 0
                    home_mid_unit_x_max = 0
                    home_defensive_unit_x_min = 0
                    home_defensive_unit_x_max = 0
                    #Ensure existince of enough players in a unit
                    if len(home_attacking_unit) > 1:
                        home_attacking_unit = np.max(home_attacking_unit)
                        home_attacking_unit_x_min = np.min(home_attacking_unit_x)
                        home_attacking_unit_x_max = np.max(home_attacking_unit_x)
                    else:
                        home_attacking_unit = pass_start_y - 1
                    if len(home_mid_unit) > 1:
                        home_mid_unit = np.max(home_mid_unit)
                        home_mid_unit_x_min = np.min(home_mid_unit_x)
                        home_mid_unit_x_max = np.max(home_mid_unit_x)
                    else:
                        home_mid_unit = pass_start_y - 1
                    if len(home_defensive_unit) > 1:
                        home_defensive_unit = np.max(home_defensive_unit)
                        home_defensive_unit_x_min = np.min(home_defensive_unit_x)
                        home_defensive_unit_x_max = np.max(home_defensive_unit_x)
                    else:
                        home_defensive_unit = pass_start_y - 1

                    if pass_start_y <= home_attacking_unit: # check attacking unit break
                        if pass_end_y > home_attacking_unit: # attacking unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                away_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                away_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                away_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < home_attacking_unit_x_min or pass_end_x > home_attacking_unit_x_max: # around
                                away_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

                    elif pass_start_y > home_attacking_unit and pass_start_y <= home_mid_unit: # check midfield unit break
                        if pass_end_y > home_mid_unit: # midfield unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                away_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                away_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                away_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < home_mid_unit_x_min or pass_end_x > home_mid_unit_x_max: # around
                                away_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

                    elif pass_start_y > home_mid_unit and pass_start_y <= home_defensive_unit: # check defensive unit break
                        if pass_end_y > home_defensive_unit: # defensive unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass': # pass
                                away_pass.append(f)
                            elif event_type[i] == 'cross': # cross
                                away_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                away_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < home_defensive_unit_x_min or pass_end_x > home_defensive_unit_x_max: # around
                                away_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

                elif f < len(self.tracking.dirs) and self.tracking.dirs[f] == 1: # Home Keeper Down - Away Keeper Up
                    # Home Team Units
                    sorted_indices = []
                    temp_home_coords = []
                    if len(self.tracking.home_coords) > f + frame_gap:
                        sorted_indices = np.argsort(self.tracking.home_coords[f + frame_gap], axis = 0)
                        temp_home_coords = self.tracking.home_coords[f + frame_gap][sorted_indices[:,1]] # sort by y coords
                    else:
                        sorted_indices = np.argsort(self.tracking.home_coords[f], axis = 0)
                        temp_home_coords = self.tracking.home_coords[f][sorted_indices[:,1]] # sort by y coords
                    
                    temp_home_coords_x = temp_home_coords[1:,0]
                    temp_home_coords_y = temp_home_coords[1:,1]
                    home_deepest = np.min(temp_home_coords_y) # deepest player
                    home_farthest = np.max(temp_home_coords_y) # farthest player
                    home_interval = (home_farthest - home_deepest) / 3 # y - interval 
                    home_offensive_down = home_farthest - home_interval
                    home_defensive_up = home_deepest + home_interval
                    home_attacking_unit = temp_home_coords_y[(temp_home_coords_y > home_offensive_down) & (temp_home_coords_y <= home_farthest)] # attacking unit 
                    home_mid_unit = temp_home_coords_y[(temp_home_coords_y >= home_defensive_up) & (temp_home_coords_y <= home_offensive_down)] # midfield unit 
                    home_defensive_unit = temp_home_coords_y[(temp_home_coords_y >= home_deepest) & (temp_home_coords_y < home_defensive_up)]  # defensive unit 
                    home_attacking_unit_x = temp_home_coords_x[(temp_home_coords_y > home_offensive_down) & (temp_home_coords_y <= home_farthest)] # attacking unit 
                    home_mid_unit_x = temp_home_coords_x[(temp_home_coords_y >= home_defensive_up) & (temp_home_coords_y <= home_offensive_down)] # midfield unit 
                    home_defensive_unit_x = temp_home_coords_x[(temp_home_coords_y >= home_deepest) & (temp_home_coords_y < home_defensive_up)]  # defensive unit 
                    home_attacking_unit_x_min = 0
                    home_attacking_unit_x_max = 0 
                    home_mid_unit_x_min = 0
                    home_mid_unit_x_max = 0
                    home_defensive_unit_x_min = 0
                    home_defensive_unit_x_max = 0
                    #Ensure existince of enough players in a unit
                    if len(home_attacking_unit) > 1:
                        home_attacking_unit = np.min(home_attacking_unit)
                        home_attacking_unit_x_min = np.min(home_attacking_unit_x)
                        home_attacking_unit_x_max = np.max(home_attacking_unit_x)
                    else:
                        home_attacking_unit = pass_start_y + 1
                    if len(home_mid_unit) > 1:
                        home_mid_unit = np.min(home_mid_unit)
                        home_mid_unit_x_min = np.min(home_mid_unit_x)
                        home_mid_unit_x_max = np.max(home_mid_unit_x)
                    else:
                        home_mid_unit = pass_start_y + 1
                    if len(home_defensive_unit) > 1:
                        home_defensive_unit = np.min(home_defensive_unit)
                        home_defensive_unit_x_min = np.min(home_defensive_unit_x)
                        home_defensive_unit_x_max = np.max(home_defensive_unit_x)
                    else:
                        home_defensive_unit = pass_start_y + 1

                    if pass_start_y >= home_attacking_unit: # check attacking unit break
                        if pass_end_y < home_attacking_unit: # attacking unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass':
                                away_pass.append(f)
                            elif event_type[i] == 'cross':
                                away_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                away_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < home_attacking_unit_x_min or pass_end_x > home_attacking_unit_x_max: # around
                                away_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

                    elif pass_start_y < home_attacking_unit and pass_start_y >= home_mid_unit: # check midfield unit break
                        if pass_end_y < home_mid_unit: # midfield unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass':
                                away_pass.append(f)
                            elif event_type[i] == 'cross':
                                away_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                away_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < home_attacking_unit_x_min or pass_end_x > home_attacking_unit_x_max: # around
                                away_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0

                    elif pass_start_y < home_mid_unit and pass_start_y >= home_defensive_unit: # check defensive unit break
                        if pass_end_y < home_defensive_unit: # defensive unit broken
                            lb_pass_events.iloc[i,-1] = 1
                            # distribution type classification
                            if event_type[i] == 'pass':
                                away_pass.append(f)
                            elif event_type[i] == 'cross':
                                away_cross.append(f)
                            # direction classification
                            if len(np.where(self.tracking.ball_height[f - over_duration//4:  f + 3 * over_duration //4] > over_height_threshold)[0]) > 0: # over 
                                away_over.append(f)
                                lb_pass_events.iloc[i,-3] = 2
                            elif pass_end_x < home_attacking_unit_x_min or pass_end_x > home_attacking_unit_x_max: # around
                                away_around.append(f)
                                lb_pass_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_pass_events.iloc[i,-3] = 0
            

            
        # BALL PROGRESSION
        lb_prog_events = self.events.events[self.events.events["event"].isin(["ball_progression"])][['x_location_start',"x_location_end", "y_location_start", "y_location_end", 'x_location_start_mirrored', 'x_location_end_mirrored', 'y_location_end_mirrored', "match_run_time_in_ms", "from_player_id"]]
        lb_prog_events = lb_prog_events.astype({'from_player_id': 'int64'})
        lb_prog_events["match_run_time_in_ms"] = lb_prog_events["match_run_time_in_ms"] // 40 # 1 frame = 40 ms
        lb_prog_events = lb_prog_events.rename(columns={"match_run_time_in_ms":"frame"})
        lb_prog_events = lb_prog_events[lb_prog_events["x_location_end_mirrored"] <= event_x_final_up] # inside pitch
        lb_prog_events = lb_prog_events[lb_prog_events["y_location_end_mirrored"] <= y_event_ch_left] # inside pitch
        lb_prog_events = lb_prog_events[lb_prog_events["y_location_end_mirrored"] >= y_event_ch_right] #inside pitch
        lb_prog_events = lb_prog_events[lb_prog_events["x_location_start_mirrored"] <= lb_prog_events["x_location_end_mirrored"]] # only forward progressions
        lb_prog_events["x_location_start_tracking"] = (lb_prog_events["y_location_start"] - 0.5) * tracking_x_range # to tracking coords 
        lb_prog_events["y_location_start_tracking"] = - (lb_prog_events["x_location_start"] - 0.5) * tracking_y_range # to tracking coords 
        lb_prog_events["x_location_end_tracking"] = (lb_prog_events["y_location_end"] - 0.5) * tracking_x_range # to tracking coords 
        lb_prog_events["y_location_end_tracking"] = - (lb_prog_events["x_location_end"] - 0.5) * tracking_y_range # to tracking coords 
        lb_prog_events = lb_prog_events.astype({'x_location_start_tracking': 'int64'})
        lb_prog_events = lb_prog_events.astype({'y_location_start_tracking': 'int64'})
        lb_prog_events = lb_prog_events.astype({'x_location_end_tracking': 'int64'})
        lb_prog_events = lb_prog_events.astype({'y_location_end_tracking': 'int64'})

        prog_frames = np.array(lb_prog_events["frame"]) + match_start_frame
        adjusted_prog_frames = []
        #adjust frames for ball progression
        for p in prog_frames:
            old_p = p
            if old_p > self.match.meta["Phase2StartFrame"][0]:
                p -= self.match.meta["Phase2StartFrame"][0] - self.match.meta["Phase1EndFrame"][0]

            if self.tracking.match_len > 2: # Over Time
                if old_p > self.match.meta["Phase3StartFrame"][0]: # end of match - start of first OT
                    p-= self.match.meta["Phase3StartFrame"][0] - self.match.meta["Phase2EndFrame"][0]
                if old_p > self.match.meta["Phase4StartFrame"][0]: # start of first OT - start of second OT
                    p-= self.match.meta["Phase4StartFrame"][0] - self.match.meta["Phase3EndFrame"][0]
                
                if self.tracking.match_len > 4: # Penalties
                    if old_p > self.match.meta["Phase5StartFrame"][0]: # end of match - start of first OT
                        p-= self.match.meta["Phase5StartFrame"][0] - self.match.meta["Phase4EndFrame"][0]
            
            adjusted_prog_frames.append(p - match_start_frame)

        lb_prog_events["frame"] = adjusted_prog_frames
        player_ids_event_prog = list(lb_prog_events["from_player_id"])
        lb_prog_events["direction"] = 0
        lb_prog_events["team_id"] = 0
        lb_prog_events["is_line_break"] = 0
        lb_prog_events = lb_prog_events.reset_index(drop=True)
        for i,f in enumerate(adjusted_prog_frames): 
            progressor = player_ids_event_prog[i]
            progressor_idx = player_ids_match.index(progressor)
            progressor_team = team_ids[progressor_idx]
            lb_prog_events.iloc[i, -2] = progressor_team
            # Progression Location
            prog_start_x = lb_prog_events["x_location_start_tracking"].values[i]
            prog_start_y = lb_prog_events["y_location_start_tracking"].values[i]
            prog_end_x = lb_prog_events["x_location_end_tracking"].values[i]
            prog_end_y = lb_prog_events["y_location_end_tracking"].values[i]

            if progressor_team == home_id: # Home Successful Progression
                if f < len(self.tracking.dirs) and self.tracking.dirs[f] == -1: # Home Keeper Up - Away Keeper Down
                    # Away Team Units
                    sorted_indices = np.argsort(self.tracking.away_coords[f], axis = 0)
                    temp_away_coords = self.tracking.away_coords[f][sorted_indices[:,1]] # sort by y coords
                    temp_away_coords_x = temp_away_coords[1:,0]
                    temp_away_coords_y = temp_away_coords[1:,1]
                    away_deepest = np.min(temp_away_coords_y) # deepest player
                    away_farthest = np.max(temp_away_coords_y) # farthest player
                    away_interval = (away_farthest - away_deepest) / 3 # y - interval 
                    away_offensive_down = away_farthest - away_interval
                    away_defensive_up = away_deepest + away_interval
                    away_attacking_unit = temp_away_coords_y[(temp_away_coords_y > away_offensive_down) & (temp_away_coords_y <= away_farthest)] # attacking unit 
                    away_mid_unit = temp_away_coords_y[(temp_away_coords_y >= away_defensive_up) & (temp_away_coords_y <= away_offensive_down)] # midfield unit 
                    away_defensive_unit = temp_away_coords_y[(temp_away_coords_y >= away_deepest) & (temp_away_coords_y < away_defensive_up)]  # defensive unit 
                    away_attacking_unit_x = temp_away_coords_x[(temp_away_coords_y > away_offensive_down) & (temp_away_coords_y <= away_farthest)] # attacking unit 
                    away_mid_unit_x = temp_away_coords_x[(temp_away_coords_y >= away_defensive_up) & (temp_away_coords_y <= away_offensive_down)] # midfield unit 
                    away_defensive_unit_x = temp_away_coords_x[(temp_away_coords_y >= away_deepest) & (temp_away_coords_y < away_defensive_up)]  # defensive unit 
                    away_attacking_unit_x_min = 0
                    away_attacking_unit_x_max = 0
                    away_mid_unit_x_min = 0
                    away_mid_unit_x_max = 0
                    away_defensive_unit_x_min = 0
                    away_defensive_unit_x_max = 0

                    # Ensure existince of enough players in a unit
                    if len(away_attacking_unit) > 1:
                        away_attacking_unit = np.min(away_attacking_unit)
                        away_attacking_unit_x_min = np.min(away_attacking_unit_x)
                        away_attacking_unit_x_max = np.max(away_attacking_unit_x)
                    else:
                        away_attacking_unit = prog_start_y + 1
                    if len(away_mid_unit) > 1:
                        away_mid_unit = np.min(away_mid_unit)
                        away_mid_unit_x_min = np.min(away_mid_unit_x)
                        away_mid_unit_x_max = np.max(away_mid_unit_x)
                    else:
                        away_mid_unit = prog_start_y + 1
                    if len(away_defensive_unit) > 1:
                        away_defensive_unit = np.min(away_defensive_unit)
                        away_defensive_unit_x_min = np.min(away_defensive_unit_x)
                        away_defensive_unit_x_max = np.max(away_defensive_unit_x)
                    else:
                        away_defensive_unit = prog_start_y + 1
                    
                    if prog_start_y >= away_attacking_unit: # check attacking unit break
                        if prog_end_y < away_attacking_unit: # attacking unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            home_prog.append(f)
                            # direction classification
                            if prog_end_x < away_attacking_unit_x_min or prog_end_x > away_attacking_unit_x_max: # around
                                home_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0


                    elif prog_start_y < away_attacking_unit and prog_start_y >= away_mid_unit: # check midfield unit break
                        if prog_end_y < away_mid_unit: # midfield unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            home_prog.append(f)
                            # direction classification
                            if prog_end_x < away_mid_unit_x_min or prog_end_x > away_mid_unit_x_max: # around
                                home_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0

                    elif prog_start_y < away_mid_unit and prog_start_y >= away_defensive_unit: # check defensive unit break
                        if prog_end_y < away_defensive_unit: # defensive unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            home_prog.append(f)
                            # direction classification
                            if prog_end_x < away_defensive_unit_x_min or prog_end_x > away_defensive_unit_x_max: # around
                                home_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0
                
                elif f < len(self.tracking.dirs) and self.tracking.dirs[f] == 1: # Home Keeper Down - Away Keeper Up
                    # Away Team Units
                    sorted_indices = np.argsort(self.tracking.away_coords[f], axis = 0)
                    temp_away_coords = self.tracking.away_coords[f][sorted_indices[:,1]] # sort by y coords
                    temp_away_coords_x = temp_away_coords[:-1,0]
                    temp_away_coords_y = temp_away_coords[:-1,1]
                    away_deepest = np.max(temp_away_coords_y) # deepest player
                    away_farthest = np.min(temp_away_coords_y) # farthest player
                    away_interval = (away_deepest - away_farthest) / 3 # y - interval 
                    away_offensive_down = away_farthest + away_interval
                    away_defensive_up = away_deepest - away_interval
                    away_attacking_unit = temp_away_coords_y[(temp_away_coords_y < away_offensive_down) & (temp_away_coords_y >= away_farthest)] # attacking unit 
                    away_mid_unit = temp_away_coords_y[(temp_away_coords_y <= away_defensive_up) & (temp_away_coords_y >= away_offensive_down)] # midfield unit 
                    away_defensive_unit = temp_away_coords_y[(temp_away_coords_y <= away_deepest) & (temp_away_coords_y > away_defensive_up)] # defensive unit 
                    away_attacking_unit_x = temp_away_coords_x[(temp_away_coords_y < away_offensive_down) & (temp_away_coords_y >= away_farthest)] # attacking unit 
                    away_mid_unit_x = temp_away_coords_x[(temp_away_coords_y <= away_defensive_up) & (temp_away_coords_y >= away_offensive_down)] # midfield unit 
                    away_defensive_unit_x = temp_away_coords_x[(temp_away_coords_y <= away_deepest) & (temp_away_coords_y > away_defensive_up)] # defensive unit 
                    away_attacking_unit_x_min = 0
                    away_attacking_unit_x_max = 0
                    away_mid_unit_x_min = 0
                    away_mid_unit_x_max = 0
                    away_defensive_unit_x_min = 0
                    away_defensive_unit_x_max = 0

                    # Ensure existince of enough players in a unit
                    if len(away_attacking_unit) > 1:
                        away_attacking_unit = np.max(away_attacking_unit)
                        away_attacking_unit_x_min = np.min(away_attacking_unit_x)
                        away_attacking_unit_x_max = np.max(away_attacking_unit_x)
                    else:
                        away_attacking_unit = prog_start_y - 1
                    if len(away_mid_unit) > 1:
                        away_mid_unit = np.max(away_mid_unit)
                        away_mid_unit_x_min = np.min(away_attacking_unit_x)
                        away_mid_unit_x_max = np.max(away_attacking_unit_x)
                    else:
                        away_mid_unit = prog_start_y - 1
                    if len(away_defensive_unit) > 1:
                        away_defensive_unit = np.max(away_defensive_unit)
                        away_defensive_unit_x_min = np.min(away_defensive_unit_x)
                        away_defensive_unit_x_max = np.max(away_defensive_unit_x)
                    else:
                        away_defensive_unit = prog_start_y - 1
                    
                    if prog_start_y <= away_attacking_unit: # check attacking unit break
                        if prog_end_y > away_attacking_unit: # attacking unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            home_prog.append(f)
                            # direction classification
                            if prog_end_x < away_attacking_unit_x_min or prog_end_x > away_attacking_unit_x_max: # around
                                home_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else:  # through
                                home_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0

                    elif prog_start_y > away_attacking_unit and prog_start_y <= away_mid_unit: # check midfield unit break
                        if prog_end_y > away_mid_unit: # midfield unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            home_prog.append(f)
                            # direction classification
                            if prog_end_x < away_defensive_unit_x_min or prog_end_x > away_defensive_unit_x_max: # around
                                home_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0

                    elif prog_start_y > away_mid_unit and prog_start_y <= away_defensive_unit: # check defensive unit break
                        if prog_end_y > away_defensive_unit: # defensive unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            home_prog.append(f)
                            # direction classification
                            if prog_end_x < away_defensive_unit_x_min or prog_end_x > away_defensive_unit_x_max: # around
                                home_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                home_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0

            elif progressor_team == away_id: # Away Succesful Progression
                if f < len(self.tracking.dirs) and self.tracking.dirs[f] == -1: # Home Keeper Up - Away Keeper Down
                    # Home Team Units
                    sorted_indices = np.argsort(self.tracking.home_coords[f], axis = 0)
                    temp_home_coords = self.tracking.home_coords[f][sorted_indices[:,1]] # sort by y coords
                    temp_home_coords_x = temp_home_coords[:-1,0]
                    temp_home_coords_y = temp_home_coords[:-1,1]
                    home_deepest = np.max(temp_home_coords_y) # deepest player
                    home_farthest = np.min(temp_home_coords_y) # farthest player
                    home_interval = (home_deepest - home_farthest) / 3 # y - interval 
                    home_offensive_down = home_farthest + home_interval
                    home_defensive_up = home_deepest - home_interval
                    home_attacking_unit = temp_home_coords_y[(temp_home_coords_y < home_offensive_down) & (temp_home_coords_y >= home_farthest)] # attacking unit 
                    home_mid_unit = temp_home_coords_y[(temp_home_coords_y <= home_defensive_up) & (temp_home_coords_y >= home_offensive_down)] # midfield unit 
                    home_defensive_unit = temp_home_coords_y[(temp_home_coords_y <= home_deepest) & (temp_home_coords_y > home_defensive_up)] # defensive unit 
                    home_attacking_unit_x = temp_home_coords_x[(temp_home_coords_y < home_offensive_down) & (temp_home_coords_y >= home_farthest)] # attacking unit 
                    home_mid_unit_x = temp_home_coords_x[(temp_home_coords_y <= home_defensive_up) & (temp_home_coords_y >= home_offensive_down)] # midfield unit 
                    home_defensive_unit_x = temp_home_coords_x[(temp_home_coords_y <= home_deepest) & (temp_home_coords_y > home_defensive_up)] # defensive unit 
                    home_attacking_unit_x_min = 0
                    home_attacking_unit_x_max = 0
                    home_mid_unit_x_min = 0
                    home_mid_unit_x_max = 0
                    home_defensive_unit_x_min = 0
                    home_defensive_unit_x_max = 0

                    # Ensure existince of enough players in a unit
                    if len(home_attacking_unit) > 1:
                        home_attacking_unit = np.max(home_attacking_unit)
                        home_attacking_unit_x_min = np.min(home_attacking_unit_x)
                        home_attacking_unit_x_max = np.max(home_attacking_unit_x)
                    else:
                        home_attacking_unit = prog_start_y - 1
                    if len(home_mid_unit) > 1:
                        home_mid_unit = np.max(home_mid_unit)
                        home_mid_unit_x_min = np.min(home_mid_unit_x)
                        home_mid_unit_x_max = np.max(home_mid_unit_x)
                    else:
                        home_mid_unit = prog_start_y - 1
                    if len(home_defensive_unit) > 1:
                        home_defensive_unit = np.max(home_defensive_unit)
                        home_defensive_unit_x_min = np.min(home_defensive_unit_x)
                        home_defensive_unit_x_max = np.max(home_defensive_unit_x)
                    else:
                        home_defensive_unit = prog_start_y - 1
                    
                    if prog_start_y <= home_attacking_unit: # check attacking unit break
                        if prog_end_y > home_attacking_unit: # attacking unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            away_prog.append(f)
                            # direction classification
                            if prog_end_x < home_attacking_unit_x_min or prog_end_x > home_attacking_unit_x_max: # around
                                away_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0

                    elif prog_start_y > home_attacking_unit and prog_start_y <= home_mid_unit: # check midfield unit break
                        if prog_end_y > home_mid_unit: # midfield unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            away_prog.append(f)
                            # direction classification
                            if prog_end_x < home_mid_unit_x_min or prog_end_x > home_mid_unit_x_max: # around
                                away_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0

                    elif prog_start_y > home_mid_unit and prog_start_y <= home_defensive_unit: # check defensive unit break
                        if prog_end_y > home_defensive_unit: # defensive unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            away_prog.append(f)
                            # direction classification
                            if prog_end_x < home_defensive_unit_x_min or prog_end_x > home_defensive_unit_x_max: # around
                                away_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0


                elif f < len(self.tracking.dirs) and self.tracking.dirs[f] == 1: # Home Keeper Up - Away Keeper Down
                    # Home Team Units
                    sorted_indices = np.argsort(self.tracking.home_coords[f], axis = 0)
                    temp_home_coords = self.tracking.home_coords[f][sorted_indices[:,1]] # sort by y coords
                    temp_home_coords_x = temp_home_coords[1:,0]
                    temp_home_coords_y = temp_home_coords[1:,1]
                    home_deepest = np.min(temp_home_coords_y)
                    home_farthest = np.max(temp_home_coords_y)
                    home_interval = (home_farthest - home_deepest) / 3 # y - interval 
                    home_offensive_down = home_farthest - home_interval
                    home_defensive_up = home_deepest + home_interval
                    home_attacking_unit = temp_home_coords_y[(temp_home_coords_y > home_offensive_down) & (temp_home_coords_y <= home_farthest)] # attacking unit 
                    home_mid_unit = temp_home_coords_y[(temp_home_coords_y >= home_defensive_up) & (temp_home_coords_y <= home_offensive_down)] # midfield unit 
                    home_defensive_unit = temp_home_coords_y[(temp_home_coords_y >= home_deepest) & (temp_home_coords_y < home_defensive_up)]  # defensive unit 
                    home_attacking_unit_x = temp_home_coords_x[(temp_home_coords_y > home_offensive_down) & (temp_home_coords_y <= home_farthest)] # attacking unit 
                    home_mid_unit_x = temp_home_coords_x[(temp_home_coords_y >= home_defensive_up) & (temp_home_coords_y <= home_offensive_down)] # midfield unit 
                    home_defensive_unit_x = temp_home_coords_x[(temp_home_coords_y >= home_deepest) & (temp_home_coords_y < home_defensive_up)]  # defensive unit 
                    home_attacking_unit_x_min = 0
                    home_attacking_unit_x_max = 0 
                    home_mid_unit_x_min = 0
                    home_mid_unit_x_max = 0
                    home_defensive_unit_x_min = 0
                    home_defensive_unit_x_max = 0
                    # Ensure existince of enough players in a unit
                    if len(home_attacking_unit) > 1:
                        home_attacking_unit = np.min(home_attacking_unit)
                        home_attacking_unit_x_min = np.min(home_attacking_unit_x)
                        home_attacking_unit_x_max = np.max(home_attacking_unit_x)
                    else:
                        home_attacking_unit = prog_start_y + 1
                    if len(home_mid_unit) > 1:
                        home_mid_unit = np.min(home_mid_unit)
                        home_mid_unit_x_min = np.min(home_mid_unit_x)
                        home_mid_unit_x_max = np.max(home_mid_unit_x)
                    else:
                        home_mid_unit = prog_start_y + 1
                    if len(home_defensive_unit) > 1:
                        home_defensive_unit = np.min(home_defensive_unit)
                        home_defensive_unit_x_min = np.min(home_defensive_unit_x)
                        home_defensive_unit_x_max = np.max(home_defensive_unit_x)
                    else:
                        home_defensive_unit = prog_start_y + 1

                    if prog_start_y >= home_attacking_unit: # check attacking unit break
                        if prog_end_y < home_attacking_unit: # attacking unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            away_prog.append(f)
                            # direction classification
                            if prog_end_x < home_attacking_unit_x_min or prog_end_x > home_attacking_unit_x_max: # around
                                away_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0

                    elif prog_start_y < home_attacking_unit and prog_start_y >= home_mid_unit: # check midfield unit break
                        if prog_end_y < home_mid_unit: # midfield unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            away_prog.append(f)
                            # direction classification
                            if prog_end_x < home_attacking_unit_x_min or prog_end_x > home_attacking_unit_x_max: # around
                                away_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0

                    elif prog_start_y < home_mid_unit and prog_start_y >= home_defensive_unit: # check defensive unit break
                        if prog_end_y < home_defensive_unit: # defensive unit broken
                            lb_prog_events.iloc[i,-1] = 1
                            # distribution type classification
                            away_prog.append(f)
                            # direction classification
                            if prog_end_x < home_attacking_unit_x_min or prog_end_x > home_attacking_unit_x_max: # around
                                away_around.append(f)
                                lb_prog_events.iloc[i,-3] = 1
                            else: # through
                                away_through.append(f)
                                lb_prog_events.iloc[i,-3] = 0


        return home_pass, home_cross, home_prog, away_pass, away_cross, away_prog,\
               home_through, home_around, home_over, away_through, away_around, away_over,\
               lb_pass_events, lb_prog_events

    def receptions(self, event_x_final_up = 1, y_event_ch_left = 1, y_event_ch_right = 0, #pitch frame coordinates
                   tracking_x_range = 68, tracking_y_range = 105, #pitch frame coordinates
                   frame_gap = -5): # frame adjustment heuristic
        """ Receptions calculation.
        Parameters
        ---------- 
        frame_gap: integer, optional
            number of frames used to add on the syncronized frames.

        Return 
        home_between_mid_def: list
            frames at which the home team made receptions between midfield and defensive lines.
        home_behind_def: float
            frames at which the home team made receptions behind defensive line.
        away_between_mid_def: integer
            frames at which the away team made receptions between midfield and defensive lines.
        away_behind_def: integer
            frames at which the away team made receptions behind defensive line.
        recep_events: list
            recorded reception events behind midfield and defensive lines.
        """
        
        home_behind_def = []
        away_behind_def = []
        home_between_mid_def = []
        away_between_mid_def = []
         # find receptions and do pre-proccessing
        recep_events = self.events.events[self.events.events["event"].isin(["reception"])][['event', 'x', "y", 'x_mirrored', 'y_mirrored', "match_run_time_in_ms", "from_player_id"]]
        recep_events = recep_events.dropna(subset = ["from_player_id"])
        recep_events = recep_events.astype({'from_player_id': 'int64'})
        recep_events["match_run_time_in_ms"] = recep_events["match_run_time_in_ms"] // 40 # 1 frame = 40 ms
        recep_events = recep_events.rename(columns={"match_run_time_in_ms":"frame"})
        recep_events = recep_events[recep_events["x_mirrored"] <= event_x_final_up] # inside pitch
        recep_events = recep_events[recep_events["y_mirrored"] <= y_event_ch_left] # inside pitch
        recep_events = recep_events[recep_events["y_mirrored"] >= y_event_ch_right] #inside pitch
        recep_events["x_tracking"] = (recep_events["y"] - 0.5) * tracking_x_range # to tracking coords 
        recep_events["y_tracking"] = - (recep_events["x"] - 0.5) * tracking_y_range # to tracking coords 
        recep_events = recep_events.astype({'x_tracking': 'int64'})
        recep_events = recep_events.astype({'y_tracking': 'int64'})

        match_start_frame = list(self.tracking.tracking["frame"])[0]
        recep_frames = np.array(recep_events["frame"]) + match_start_frame
        adjusted_recep_frames = []
        #adjust frames for passes
        for p in recep_frames:
            old_p = p 
            if old_p > self.match.meta["Phase2StartFrame"][0]:
                p -= self.match.meta["Phase2StartFrame"][0] - self.match.meta["Phase1EndFrame"][0]

            if self.tracking.match_len > 2: # Over Time
                if old_p > self.match.meta["Phase3StartFrame"][0]: # end of match - start of first OT
                    p-= self.match.meta["Phase3StartFrame"][0] - self.match.meta["Phase2EndFrame"][0]
                if old_p > self.match.meta["Phase4StartFrame"][0]: # start of first OT - start of second OT
                    p-= self.match.meta["Phase4StartFrame"][0] - self.match.meta["Phase3EndFrame"][0]
                
                if self.tracking.match_len > 4: # Penalties
                    if old_p > self.match.meta["Phase5StartFrame"][0]: # end of match - start of first OT
                        p-= self.match.meta["Phase5StartFrame"][0] - self.match.meta["Phase4EndFrame"][0]
            
            adjusted_recep_frames.append(p - match_start_frame)

        recep_events["frame"] = adjusted_recep_frames
        player_ids_receiver = list(recep_events["from_player_id"])
        team_ids = list(self.match.lineups["team_id"])
        player_ids_match = list(self.match.lineups["player_id"])
        home_id = self.match.meta["home_team_id"][0]
        away_id = self.match.meta["away_team_id"][0]
        recep_events["need_mirror"] = 0
        recep_events["is_reception"] = 0
        recep_events["team_id"] = 0
        recep_events["reception_unit"] = 0
        recep_events = recep_events.reset_index(drop = True)

        for i,f in enumerate(adjusted_recep_frames):
            receiver = player_ids_receiver[i]
            receiver_idx = player_ids_match.index(receiver)
            receiver_team = team_ids[receiver_idx]
            recep_events.iloc[i,-2] = receiver_team
            # Reception Location
            recep_y = recep_events["y_tracking"].values[i]
            if receiver_team == home_id: # Home Successful Reception
                if f < len(self.tracking.dirs) and self.tracking.dirs[f] == -1: # Home Keeper Up - Away Keeper Down
                    # Away Team Units
                    sorted_indices = []
                    temp_away_coords = []
                    if len(self.tracking.away_coords) > f + frame_gap:
                        sorted_indices = np.argsort(self.tracking.away_coords[f + frame_gap], axis = 0)
                        temp_away_coords = self.tracking.away_coords[f+ frame_gap][sorted_indices[:,1]] # sort by y coords
                    else:
                        sorted_indices = np.argsort(self.tracking.away_coords[f], axis = 0)
                        temp_away_coords = self.tracking.away_coords[f][sorted_indices[:,1]] # sort by y coords

                    temp_away_coords_y = temp_away_coords[1:,1]
                    away_deepest = np.min(temp_away_coords_y) # deepest player
                    away_farthest = np.max(temp_away_coords_y) # farthest player
                    away_interval = (away_farthest - away_deepest) / 3 # y - interval 
                    away_offensive_down = away_farthest - away_interval
                    away_defensive_up = away_deepest + away_interval
                    away_mid_unit = temp_away_coords_y[(temp_away_coords_y >= away_defensive_up) & (temp_away_coords_y <= away_offensive_down)] # midfield unit 
                    away_defensive_unit = np.min(temp_away_coords_y[(temp_away_coords_y >= away_deepest) & (temp_away_coords_y < away_defensive_up)])  # defensive unit 
                    # Ensure existince of enough players in mid unit
                    if len(away_mid_unit) > 0:
                        away_mid_unit = np.min(away_mid_unit)
                    else:
                        away_mid_unit = recep_y + 1

                    # Reception Between Midfield - Defensive Line
                    if recep_y < away_mid_unit and recep_y >= away_defensive_unit:
                        home_between_mid_def.append(f)
                        recep_events.iloc[i,-1] = 0
                        recep_events.iloc[i,-3] = 1
                        recep_events.iloc[i,-4] = 1
                    # Reception Behind Defensive Line
                    elif recep_y < away_defensive_unit:
                        home_behind_def.append(f)
                        recep_events.iloc[i,-1] = 1
                        recep_events.iloc[i,-3] = 1
                        recep_events.iloc[i,-4] = 1

                elif f < len(self.tracking.dirs) and self.tracking.dirs[f] == 1: # Home Keeper Down - Away Keeper Up
                    # Away Team Units
                    sorted_indices = []
                    temp_away_coords = []
                    if len(self.tracking.away_coords) > f + frame_gap:
                        sorted_indices = np.argsort(self.tracking.away_coords[f + frame_gap], axis = 0)
                        temp_away_coords = self.tracking.away_coords[f+ frame_gap][sorted_indices[:,1]] # sort by y coords
                    else:
                        sorted_indices = np.argsort(self.tracking.away_coords[f], axis = 0)
                        temp_away_coords = self.tracking.away_coords[f][sorted_indices[:,1]] # sort by y coords

                    temp_away_coords_y = temp_away_coords[:-1,1]
                    away_deepest = np.max(temp_away_coords_y) # deepest player
                    away_farthest = np.min(temp_away_coords_y) # farthest player
                    away_interval = (away_deepest - away_farthest) / 3 # y - interval 
                    away_offensive_down = away_farthest + away_interval
                    away_defensive_up = away_deepest - away_interval
                    away_mid_unit = temp_away_coords_y[(temp_away_coords_y <= away_defensive_up) & (temp_away_coords_y >= away_offensive_down)] # midfield unit 
                    away_defensive_unit = np.max(temp_away_coords_y[(temp_away_coords_y <= away_deepest) & (temp_away_coords_y > away_defensive_up)]) # defensive unit 
                    # Ensure existince of enough players in mid unit
                    if len(away_mid_unit) > 0:
                        away_mid_unit = np.max(away_mid_unit)
                    else:
                        away_mid_unit = recep_y - 1

                    # Reception Between Midfield - Defensive line
                    if recep_y > away_mid_unit and recep_y <= away_defensive_unit:
                        home_between_mid_def.append(f)
                        recep_events.iloc[i,-1] = 0
                        recep_events.iloc[i,-3] = 1
                    # Reception Behind Defensive Line
                    elif recep_y > away_defensive_unit:
                        home_behind_def.append(f)
                        recep_events.iloc[i,-1] = 1
                        recep_events.iloc[i,-3] = 1
                        
            elif receiver_team == away_id: # Away Succesful Reception
                if f < len(self.tracking.dirs) and self.tracking.dirs[f] == -1: # Home Keeper Up - Away Keeper Down
                    # Home Team Units
                    sorted_indices = []
                    temp_home_coords = []
                    if len(self.tracking.home_coords) > f + frame_gap:
                        sorted_indices = np.argsort(self.tracking.home_coords[f + frame_gap], axis = 0)
                        temp_home_coords = self.tracking.home_coords[f+ frame_gap][sorted_indices[:,1]] # sort by y coords
                    else:
                        sorted_indices = np.argsort(self.tracking.home_coords[f], axis = 0)
                        temp_home_coords = self.tracking.home_coords[f][sorted_indices[:,1]] # sort by y coords

                    temp_home_coords_y = temp_home_coords[:-1,1]
                    home_deepest = np.max(temp_home_coords_y) # deepest player
                    home_farthest = np.min(temp_home_coords_y) # farthest player
                    home_interval = (home_deepest - home_farthest) / 3 # y - interval 
                    home_offensive_down = home_farthest + home_interval
                    home_defensive_up = home_deepest - home_interval
                    home_mid_unit = temp_home_coords_y[(temp_home_coords_y <= home_defensive_up) & (temp_home_coords_y >= home_offensive_down)] # midfield unit 
                    home_defensive_unit = np.max(temp_home_coords_y[(temp_home_coords_y <= home_deepest) & (temp_home_coords_y > home_defensive_up)]) # defensive unit 
                    # Ensure existince of enough players in mid unit
                    if len(home_mid_unit) > 0:
                        home_mid_unit = np.max(home_mid_unit)
                    else:
                        home_mid_unit = recep_y - 1

                    # Reception Between Midfield - Defensive line
                    if recep_y > home_mid_unit and recep_y <= home_defensive_unit:
                        away_between_mid_def.append(f)
                        recep_events.iloc[i,-1] = 0
                        recep_events.iloc[i,-3] = 1
                    # Reception Behind Defensive Line
                    elif recep_y > home_defensive_unit:
                        away_behind_def.append(f)
                        recep_events.iloc[i,-1] = 1
                        recep_events.iloc[i,-3] = 1
                        
                elif f < len(self.tracking.dirs) and self.tracking.dirs[f] == 1: # Home Keeper Down - Away Keeper Up
                    # Home Team Units
                    sorted_indices = []
                    temp_home_coords = []
                    if len(self.tracking.home_coords) > f + frame_gap:
                        sorted_indices = np.argsort(self.tracking.home_coords[f + frame_gap], axis = 0)
                        temp_home_coords = self.tracking.home_coords[f+ frame_gap][sorted_indices[:,1]] # sort by y coords
                    else:
                        sorted_indices = np.argsort(self.tracking.home_coords[f], axis = 0)
                        temp_home_coords = self.tracking.home_coords[f][sorted_indices[:,1]] # sort by y coords
                        
                    temp_home_coords_y = temp_home_coords[1:,1]
                    home_deepest = np.min(temp_home_coords_y) # deepest player
                    home_farthest = np.max(temp_home_coords_y) # farthest player
                    home_interval = (home_farthest - home_deepest) / 3 # y - interval 
                    home_offensive_down = home_farthest - home_interval
                    home_defensive_up = home_deepest + home_interval 
                    home_mid_unit = temp_home_coords_y[(temp_home_coords_y >= home_defensive_up) & (temp_home_coords_y <= home_offensive_down)] # midfield unit 
                    home_defensive_unit = np.min(temp_home_coords_y[(temp_home_coords_y >= home_deepest) & (temp_home_coords_y < home_defensive_up)])  # defensive unit 
                    # Ensure existince of enough players in mid unit
                    if len(home_mid_unit) > 0:
                        home_mid_unit = np.min(home_mid_unit)
                    else:
                        home_mid_unit = recep_y + 1

                    # Reception Between Midfield - Defensive Line
                    if recep_y < home_mid_unit and recep_y >= home_defensive_unit:
                        away_between_mid_def.append(f)
                        recep_events.iloc[i,-1] = 0
                        recep_events.iloc[i,-3] = 1
                        recep_events.iloc[i,-4] = 1
                    # Reception Behind Defensive Line
                    elif recep_y < home_defensive_unit:
                        away_behind_def.append(f)
                        recep_events.iloc[i,-1] = 1
                        recep_events.iloc[i,-3] = 1
                        recep_events.iloc[i,-4] = 1
                    
        return home_between_mid_def, home_behind_def, away_between_mid_def, away_behind_def, recep_events