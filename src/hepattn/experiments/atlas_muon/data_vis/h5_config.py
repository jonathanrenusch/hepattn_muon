"""
Configuration and constants for muon tracking analysis.
"""

# File paths
from networkx import hits


# H5_FILEPATH = "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/gut_check_test_data"
# H5_FILEPATH = "/scratch/ml_test_data_156000_hdf5"
# H5_FILEPATH = "/scratch/ml_validation_data_144000_hdf5_no-NSW_no-RPC"
# H5_FILEPATH = "/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500"
# H5_FILEPATH = "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts"
# H5_FILEPATH = "/scratch/ml_training_data_2694000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
# H5_FILEPATH = "/scratch/ml_training_data_2694000_hdf5"
# H5_FILEPATH = "/scratch/ml_validation_data_144000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
# H5_FILEPATH = "/scratch/ml_validation_data_144000_hdf5"
# H5_FILEPATH = "/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
H5_FILEPATH = "/scratch/ml_test_data_156000_hdf5"
# H5_FILEPATH = "/scratch/ml_training_data_2694000_hdf5"
# H5_FILEPATH = "/scratch/ml_training_data_2694000_hdf5_filtered_wp0990_maxtrk2_maxhit500"
# H5_FILEPATH = "/scratch/ml_validation_data_144000_hdf5_filtered_wp0990_maxtrk2_maxhit500"
# H5_FILEPATH = "/scratch/ml_validation_data_144000_hdf5"
# H5_FILEPATH = "/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500"
# H5_FILEPATH = "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts"
# H5_FILEPATH = "/eos/project/e/end-to-end-muon-tracking/tracking/data/noNSW/ml_validation_data_144000_no-NSW"
# H5_FILEPATH = "/scratch/ml_test_data_156000_hdf5_no-NSW"
# H5_FILEPATH = "/scratch/ml_validation_data_144000_hdf5"
# H5_FILEPATH = "/scratch/ml_validation_data_144000_hdf5_no-NSW"
# H5_FILEPATH = "/scratch/_no-NSW"
# H5_FILEPATH = "/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit500"

# HIT_EVAL_FILEPATH = "/scratch/epoch=021-val_auc=0.99969_ml_test_data_156000_hdf5_eval.h5"
HIT_EVAL_FILEPATH = None
# HIT_EVAL_FILEPATH = "/home/iwsatlas1/jrenusch/master_thesis/tracking/hepattn_muon/logs/ATLAS-Muon-v1_20250814-T223545/ckpts/epoch=049-val_acc=0.99711_gut_check_test_data_eval.h5"
# H5_FILEPATH = "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/valid_PU200_processed",
# HIT_EVAL_FILEPATH = "/home/iwsatlas1/jrenusch/master_thesis/tracking/hepattn_muon/logs/ATLAS-Muon-v1_20250814-T223545/ckpts/epoch=049-val_acc=0.99711_valid_PU200_processed_eval.h5"
# H5_FILEPATH = "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/train_PU200_processed",
# Tree names
DEFAULT_TREE_NAME = "MuonHitDump"

# Branch names
BRANCH_NAMES = {
    "event_number": "eventNumber",
    "truth_link": "spacePoint_truthLink",
    "pos_x": "spacePoint_globPosX",
    "pos_y": "spacePoint_globPosY",
    "pos_z": "spacePoint_globPosZ",
    "position_x": "spacePoint_PositionX",  # Alternative naming
}

# Visualization settings
TRACK_COLORS = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFF00",
    "#FF8000",
    "#8000FF",
]

PLOT_SETTINGS = {
    "figure_size": (18, 6),
    "background_color": "gray",
    "background_alpha": 0.3,
    "background_size": 10,
    "track_alpha": 0.9,
    "track_size": 30,
    "track_marker": "x",
    "track_linewidth": 2,
    "grid_alpha": 0.3,
    "dpi": 300,
}

# Analysis settings
ANALYSIS_SETTINGS = {
    "max_track_id": 1e6,  # Filter out invalid track IDs above this value
    "min_track_id": 0,  # Filter out background hits (typically -1)
    "histogram_bins": 50,
}

# HISTOGRAM_SETTINGS = {"hits":
#                          {
#                          "hit_spacePoint_globEdgeHighX": {"bins": 100, "range": (-15000, 15000)},
#                          "hit_spacePoint_globEdgeHighY": {"bins": 100, "range": (-15000, 15000)},
#                          "hit_spacePoint_globEdgeHighZ": {"bins": 100, "range": (-25000, 25000)},
#                          "hit_spacePoint_globEdgeLowX": {"bins": 100, "range": (-15000, 15000)},
#                          "hit_spacePoint_globEdgeLowY": {"bins": 100, "range": (-15000, 15000)},
#                          "hit_spacePoint_globEdgeLowZ": {"bins": 100, "range": (-25000, 25000)},
#                          "hit_spacePoint_time": {"bins": 100, "range": (-1000, 20000)},
#                          "hit_spacePoint_driftR": {"bins": 100, "range": (0, 15)},
#                          "hit_spacePoint_readOutSide": {"bins": None, "range": (-2, 2)},
#                         #  "hit_spacePoint_covXX": {"bins": 100, "range": (-1000, 10000000), "scale_factor": 1.0},
#                         #  "hit_spacePoint_covXY": {"bins": 100, "range": (-300000, 300000), "scale_factor": 1.0},
#                         #  "hit_spacePoint_covYX": {"bins": 100, "range": (-150000, 150000), "scale_factor": 1.0},
#                         #  "hit_spacePoint_covYY": {"bins": 100, "range": (0, 1300000), "scale_factor": 1.0},
#                          "hit_spacePoint_covXX": {"bins": 100, "range": (-1000, 10000000)},
#                          "hit_spacePoint_covXY": {"bins": 100, "range": (-300000, 300000)},
#                          "hit_spacePoint_covYX": {"bins": 100, "range": (-150000, 150000)},
#                          "hit_spacePoint_covYY": {"bins": 100, "range": (0, 1300000)},
#                          "hit_spacePoint_channel": {"bins": 100, "range": (0, 6000)},
#                          "hit_spacePoint_layer": {"bins": None, "range": (0, 10)},
#                          "hit_spacePoint_stationPhi": { "bins": None, "range": (0, 50)},
#                          "hit_spacePoint_stationEta": {"bins": None, "range": (-10, 10)},
#                          "hit_spacePoint_technology": {"bins": None, "range": (-1, 10)},
#                          "hit_spacePoint_truthLink": {"bins": None, "range": (-1, 8)},
#                          "hit_r": {"bins": 100, "range": (0, 15000)},
#                          "hit_s": {"bins": 100, "range": (0, 300000)},
#                          "hit_theta": {"bins": 100, "range": (0, 3.2)},
#                          "hit_phi": {"bins": 100, "range": (-3.2, 3.2)}}, 
#                          "tragets":{
#                          "particle_truthMuon_pt": {"bins": 100, "range": (0, 200)},
#                          "particle_truthMuon_eta": {"bins": 100, "range": (-3, 3),},
#                          "particle_truthMuon_phi": {"bins": 100, "range": (-3.2, 3.2)},
#                          "particle_truthMuon_q": {"bins": None, "range": (-2, 2)}},
# }
HISTOGRAM_SETTINGS = {"hits":
                         {
                         "hit_spacePoint_globEdgeHighX": {"bins": 100, "range": (-15000, 15000), "scale_factor": 0.001},
                         "hit_spacePoint_globEdgeHighY": {"bins": 100, "range": (-15000, 15000), "scale_factor": 0.001},
                         "hit_spacePoint_globEdgeHighZ": {"bins": 100, "range": (-25000, 25000), "scale_factor": 0.001},
                         "hit_spacePoint_globEdgeLowX": {"bins": 100, "range": (-15000, 15000), "scale_factor": 0.001},
                         "hit_spacePoint_globEdgeLowY": {"bins": 100, "range": (-15000, 15000), "scale_factor": 0.001},
                         "hit_spacePoint_globEdgeLowZ": {"bins": 100, "range": (-25000, 25000), "scale_factor": 0.001},
                         "hit_spacePoint_time": {"bins": 100, "range": (-1000, 20000), "scale_factor": 0.00001},
                         "hit_spacePoint_driftR": {"bins": 100, "range": (0, 15), "scale_factor": 1.0},
                         "hit_spacePoint_readOutSide": {"bins": None, "range": (-2, 2), "scale_factor": 1.0},
                         "hit_spacePoint_covXX": {"bins": 100, "range": (-1000, 10000000), "scale_factor": 0.000001},
                         "hit_spacePoint_covXY": {"bins": 100, "range": (-300000, 300000), "scale_factor": 0.000001},
                         "hit_spacePoint_covYX": {"bins": 100, "range": (-150000, 150000), "scale_factor": 0.000001},
                         "hit_spacePoint_covYY": {"bins": 100, "range": (0, 1300000), "scale_factor": 0.000001},
                         "hit_spacePoint_channel": {"bins": 100, "range": (0, 6000), "scale_factor": 0.001},
                         "hit_spacePoint_layer": {"bins": None, "range": (0, 10), "scale_factor": 1.0},
                         "hit_spacePoint_stationPhi": { "bins": None, "range": (0, 50), "scale_factor": 1.0},
                         "hit_spacePoint_stationEta": {"bins": None, "range": (-10, 10), "scale_factor": 1.0},
                         "hit_spacePoint_technology": {"bins": None, "range": (-1, 10), "scale_factor": 1.0},
                         "hit_spacePoint_stationIndex": {"bins": None, "range": (-1000, 1000), "scale_factor": 0.1},
                         "hit_spacePoint_truthLink": {"bins": None, "range": (-1, 8), "scale_factor": 1.0},
                         "hit_r": {"bins": 100, "range": (0, 15000), "scale_factor": 0.001},
                         "hit_s": {"bins": 100, "range": (0, 30000), "scale_factor": 0.001},
                         "hit_theta": {"bins": 100, "range": (0, 3.2), "scale_factor": 1.0},
                         "hit_phi": {"bins": 100, "range": (-3.2, 3.2), "scale_factor": 1.0}
                        }, 
                         "tragets":{
                         "particle_truthMuon_pt": {"bins": 100, "range": (0, 200), "scale_factor": 1.0},
                         "particle_truthMuon_eta": {"bins": 100, "range": (-3, 3), "scale_factor": 1.0},
                         "particle_truthMuon_phi": {"bins": 100, "range": (-3.2, 3.2), "scale_factor": 1.0},
                         "particle_truthMuon_q": {"bins": None, "range": (-2, 2), "scale_factor": 1.0},
                         "particle_truthMuon_qpt": {"bins": 100, "range": (-0.25, 0.25), "scale_factor": 1.0}
                        },
}

