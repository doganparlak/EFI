from source.Match import Match
from source.Event import Event
from source.Tracking import Tracking
from source.EFI import EFI
from source.Visualizer import Visualizer

#-------- Object initializations --------
#Match object
match = Match(path= "match_data.json") # the path is given as an example and does not direct to the actual data
print("Match object is created")

#Event object
event = Event(path= "event_data.json") # the path is given as an example and does not direct to the actual data
print("Event object is created")

#Tracking object
tracking = Tracking(path= "tracking_data.dat", match= match) # the path is given as an example and does not direct to the actual data
print("Tracking object is created")

#EFI object
efi = EFI(match= match, events= event, tracking= tracking)
print("EFI object is created")

#Visualizer object
vis = Visualizer()

#-------- Example use of the EFI class --------
#Possession control
home_possession, in_contest, away_possession = efi.possession_control()
print("Possession control analysis is completed")

#Ball recovery time
home_recovery_time, away_recovery_time = efi.ball_recovery_time()
print("Ball recovery time analysis is completed")

#Pressure on the ball
pressure_home_cnt, pressure_away_cnt, pressure_home, pressure_away, pressure_index_home, pressure_index_away = efi.pressure_on_ball()
print("Pressure on the ball analysis is completed")

#Forced turnovers
home_forced_turnover_cnt, away_forced_turonver_cnt, home_forced_turnovers, away_forced_turnovers = efi.forced_turnover()
print("Forced turnover analysis is completed")

#Defensive line height and team length 
avg_home_defensive_line_heights, avg_home_offensive_line_heights, avg_away_defensive_line_heights, avg_away_offensive_line_heights, \
avg_home_defensive_team_lengths, avg_home_offensive_team_lengths, avg_away_defensive_team_lengths, avg_away_offensive_team_lengths, \
avg_home_defensive_team_widths, avg_home_offensive_team_widths, avg_away_defensive_team_widths, avg_away_offensive_team_widths = efi.line_height_team_length()
print("Defensive line height and team length analysis is completed")

#Final third entries 
home_entries, away_entries, home_entries_idx, away_entries_idx = efi.final_third_entries()
print("Final third entries analysis is completed")

#Team shape
home_shape, away_shape, home_in_pos_shape, away_in_pos_shape, home_out_pos_shape, away_out_pos_shape  = efi.team_shape()
print("Team shape analysis is completed")

#Phases of play
home_in_phases, home_out_phases, away_in_phases, away_out_phases = efi.phases_of_play()
print("Phases of play analysis is completed")

#Line breaks
home_pass, home_cross, home_prog, away_pass, away_cross, away_prog,\
home_through, home_around, home_over, away_through, away_around, away_over,\
lb_pass_events, lb_prog_events = efi.line_breaks()
print("Line breaks analysis is completed")

#Receptions behind midfield and defensive lines
home_between_mid_def, home_behind_def, away_between_mid_def, away_behind_def, recep_events = efi.receptions()
print("Receptions analysis is completed")

#Expected goal (xG)
#Games encompassing the training data for building the xG model
match_1 = Match(path= "match_data1.json") # the path is given as an example and does not direct to the actual data
event_1 = Event(path= "event_data1.json") # the path is given as an example and does not direct to the actual data
tracking_1 = Tracking(path= "tracking_data1.dat", match= match_1) # the path is given as an example and does not direct to the actual data
match_2 = Match(path= "match_data2.json") # the path is given as an example and does not direct to the actual data
event_2 = Event(path= "event_data2.json") # the path is given as an example and does not direct to the actual data
tracking_2 = Tracking(path= "tracking_data2.dat", match= match_2) # the path is given as an example and does not direct to the actual data
match_3 = Match(path= "match_data3.json") # the path is given as an example and does not direct to the actual data
event_3 = Event(path= "event_data3.json") # the path is given as an example and does not direct to the actual data
tracking_3 = Tracking(path= "tracking_data3.dat", match= match_3) # the path is given as an example and does not direct to the actual data

efi1 = EFI(match= match_1, events= event_1, tracking= tracking_1)
efi2 = EFI(match= match_2, events= event_2, tracking= tracking_2)
efi3 = EFI(match= match_3, events= event_3, tracking= tracking_3)

home_xG, away_xG, home_score, away_score, probs, shots_df, y_test  = efi.xG_model(efi_objs= [efi1, efi2, efi3])
print("Expected goal (xG) analysis is completed")

#-------- Example use of the Visualizer class --------
#Possession control
vis.possession_control_plotter(match, home_possession, in_contest, away_possession)
print("Possession control plot is generated")

#Ball recovery time
vis.ball_recovery_plotter(match, home_recovery_time, away_recovery_time)
print("Ball recovery time plot is generated")

#Pressure on the ball
vis.pressure_on_ball_plotter(match, tracking, pressure_index_home, pressure_index_away)
print("Pressure on the ball plot is generated")

#Forced turnovers
vis.forced_turnover_plotter(match, tracking, home_forced_turnovers, away_forced_turnovers)
print("Forced turnover plot is generated")

#Defensive line height and team length 
vis.line_height_team_length_plotter(match, avg_home_defensive_line_heights, avg_home_offensive_line_heights, avg_away_defensive_line_heights, avg_away_offensive_line_heights, \
                                           avg_home_defensive_team_lengths, avg_home_offensive_team_lengths, avg_away_defensive_team_lengths, avg_away_offensive_team_lengths, \
                                           avg_home_defensive_team_widths, avg_home_offensive_team_widths, avg_away_defensive_team_widths, avg_away_offensive_team_widths)
print("Defensive line height and team length plot is generated")

#Final third entries 
vis.final_third_entries_plotter(match, home_entries, away_entries)
print("Final third entries plot is generated")

#Team Shape
vis.team_shape_plotter(match, tracking, home_shape, away_shape, home_in_pos_shape, away_in_pos_shape, home_out_pos_shape, away_out_pos_shape)
print("Team shape plot is generated")

#Phases of play
vis.phases_of_play_plotter(match, home_in_phases, home_out_phases, away_in_phases, away_out_phases)
print("Phases of play plot is generated")

#Line breaks
vis.line_breaks_plotter(match, lb_pass_events, lb_prog_events)
print("Line breaks plot is generated")

#Receptions behind midfield and defensive lines
vis.reception_plotter(match, recep_events)
print("Receptions behind midfield and defensive lines plot is generated")