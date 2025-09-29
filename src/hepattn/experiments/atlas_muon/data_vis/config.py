"""
Configuration and constants for muon tracking analysis.
"""

# File paths
ROOT_FILE_PATHS = {
    # "JpsiPU60R315000": "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/JpsiPU60R315000.root",
    # "JpsiPU200R415000": "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/JpsiPU200R415000.root",
    # "ttbarPU60R315000": "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/ttbarPU60R315000.root",
    # "ttbarPU200R415000": "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/ttbarPU200R415000.root",
    # "ZmumuPU60R315000": "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/ZmumuPU60R315000.root",
    # "ZmumuPU200R415000": "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/ZmumuPU200R415000.root",
    # "JpsiNOPUR415000": "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/JpsiNOPUR415000.root",
    "PhPy8EG_A14_ttbar_hdamp258p75_dil_PU200": "/home/iwsatlas1/jrenusch/master_thesis/tracking/data/root/ml_training_data/PhPy8EG_A14_ttbar_hdamp258p75_dil_PU200_skip5000_n2500.root",
}

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

HISTOGRAM_SETTINGS = {   "spacePoint_globEdgeHighX": {"bins": 100, "range": (-15000, 15000)},
                         "spacePoint_globEdgeHighY": {"bins": 100, "range": (-15000, 15000)},
                         "spacePoint_globEdgeHighZ": {"bins": 100, "range": (-25000, 25000)},
                         "spacePoint_globEdgeLowX": {"bins": 100, "range": (-15000, 15000)},
                         "spacePoint_globEdgeLowY": {"bins": 100, "range": (-15000, 15000)},
                         "spacePoint_globEdgeLowZ": {"bins": 100, "range": (-25000, 25000)},
                         "spacePoint_time": {"bins": 100, "range": (-1000, 20000)},
                         "spacePoint_driftR": {"bins": None, "range": (0, 15)},
                         "spacePoint_readOutSide": {"bins": None, "range": (-2, 2)},
                         "spacePoint_covXX": {"bins": 100, "range": (-1000, 10000000)},
                         "spacePoint_covXY": {"bins": 100, "range": (-300000, 300000)},
                         "spacePoint_covYX": {"bins": 100, "range": (-150000, 150000)},
                         "spacePoint_covYY": {"bins": 100, "range": (0, 1300000)},
                         "spacePoint_channel": {"bins": 100, "range": (0, 6000)},
                         "spacePoint_layer": {"bins": None, "range": (0, 10)},
                         "spacePoint_stationPhi": { "bins": None, "range": (0, 50)},
                         "spacePoint_stationEta": {"bins": None, "range": (-10, 10)},
                         "spacePoint_stationIndex": {"bins": None, "range": (-100, 100)},
                         "spacePoint_technology": {"bins": None, "range": (-1, 10)},
                         "spacePoint_truthLink": {"bins": None, "range": (-1, 8)},
                         "truthMuon_pt": {"bins": 100, "range": (0, 200)},
                         "truthMuon_eta": {"bins": 100, "range": (-4, 4)},
                         "truthMuon_phi": {"bins": 100, "range": (-3.2, 3.2)},
                         "truthMuon_q": {"bins": None, "range": (-2, 1)},
}