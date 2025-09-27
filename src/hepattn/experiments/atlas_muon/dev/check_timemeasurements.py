"""
Quick check of negative time measurements in a ROOT TTree.

Opens a ROOT file, reads the tree "MuonHitDump", inspects branch
"spacePoint_time" for values < 0, prints the percentage of negatives
relative to the total number of time measurements, and lists the unique
detector technologies (via branch "spacePoint_technology") associated
with those negative times.

Dependencies: uproot, awkward
"""


import argparse
from pathlib import Path
import numpy as np
try:
    import uproot
except Exception as e:
    raise SystemExit(
        "This script requires the 'uproot' package.\n"
        "Install it, e.g.: pip install uproot\n"
        f"Import error: {e}"
    )



# Provided lookup: name -> code
TECH_TO_CODE = {"MDT": 0, "RPC": 2, "TGC": 3, "STGC": 4, "MM": 5}
CODE_TO_TECH = {v: k for k, v in TECH_TO_CODE.items()}



def analyze(file_path: Path, tree_name: str = "MuonHitDump"):
    if not file_path.exists():
        raise FileNotFoundError(f"ROOT file not found: {file_path}")

    with uproot.open(str(file_path)) as fh:
        if tree_name not in fh:
            raise KeyError(f"Tree '{tree_name}' not found in file. Available keys: {list(fh.keys())}")
        tree = fh[tree_name]
        # Load branches as numpy arrays (flattened)
        times = tree["spacePoint_time"].array(library="np")
        techs = tree["spacePoint_technology"].array(library="np")

    # Flatten arrays if jagged (object dtype)
    times_flat = np.concatenate(times) if times.dtype == object else times
    techs_flat = np.concatenate(techs) if techs.dtype == object else techs

    # Remove NaNs from times, keep alignment
    valid_mask = ~np.isnan(times_flat)
    times_valid = times_flat[valid_mask]
    techs_valid = techs_flat[valid_mask]

    neg_mask = times_valid < 0
    num_neg = np.sum(neg_mask)
    total = times_valid.size
    pct_neg = (100.0 * num_neg / total) if total > 0 else 0.0

    techs_neg = techs_valid[neg_mask]
    techs_neg = techs_neg.astype(np.int64, copy=False)
    if techs_neg.size > 0:
        unique_codes, counts = np.unique(techs_neg, return_counts=True)
        tech_counts = sorted(
            ((CODE_TO_TECH.get(int(code), str(int(code))), int(cnt)) for code, cnt in zip(unique_codes, counts)),
            key=lambda x: x[1], reverse=True
        )
    else:
        tech_counts = []

    return num_neg, total, pct_neg, tech_counts



def main():
    parser = argparse.ArgumentParser(description="Check negative time measurements in ROOT file")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(
            # "/eos/project/e/end-to-end-muon-tracking/tracking/data/ml_validation_data_144000/"
            # "P8B_A14_CTEQ6L1_Jpsi1S_mu6mu6_PU200_skip74000_n2000.root"
            # "/eos/project/e/end-to-end-muon-tracking/tracking/data/ml_validation_data_144000/PhPy8EG_A14_ttbar_hdamp258p75_dil_PU200_skip4000_n2000.root"
            "/eos/project/e/end-to-end-muon-tracking/tracking/data/ml_validation_data_144000/PhPy8EG_AZNLO_Zmumu_PU200_skip50000_n2000.root"
        ),
        help="Path to the ROOT file",
    )
    parser.add_argument(
        "--tree",
        type=str,
        default="MuonHitDump",
        help="TTree name containing the branches (default: MuonHitDump)",
    )
    args = parser.parse_args()
    file_path = args.file
    tree_name = args.tree

    num_neg, total, pct_neg, tech_counts = analyze(file_path, tree_name)

    print("=== Negative time measurement check ===")
    print(f"File: {file_path}")
    print(f"Tree: {tree_name}")
    print(f"Total time measurements: {total}")
    print(f"Negative time measurements: {num_neg}")
    print(f"Percentage negative: {pct_neg:.6f}%")

    if tech_counts:
        print("\nTechnologies for negative times (name: count):")
        for name, cnt in tech_counts:
            print(f"- {name}: {cnt}")
    else:
        print("\nNo negative time measurements found.")

if __name__ == "__main__":
    main()
