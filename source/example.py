from source.Match import Match
from source.Event import Event
from source.Tracking import Tracking
from source.EFI import EFI
from source.Visualizer import Visualizer

#-------- Object initializations --------
#Match object
match = Match(path= "data/64/64_128083_meta.json")
print("Match object is created")

#Event object
event = Event(path= "data/64/64_128083_events.json")
print("Event object is created")

#Tracking object
tracking = Tracking(path= "data/64/64_128083_tracking.dat", match= match, home_keeper_jersey= "23", away_keeper_jersey= "1", duel_zone= 2)
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

#Phases of play
home_in_phases, home_out_phases, away_in_phases, away_out_phases = efi.phases_of_play()
print("Phases of play analysis is completed")

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

#Phases of play
vis.phases_of_play_plotter(match, home_in_phases, home_out_phases, away_in_phases, away_out_phases)
print("Phases of play plot is generated")