"""
Diagnostic script to understand why pT fits are giving poor results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from hepattn.experiments.atlas_muon.data import AtlasMuonDataset

# Load data
data_dir = "/scratch/ml_test_156000_regression_hdf5_true_hits_only"

hit_fields = [
    'spacePoint_globEdgeHighX', 'spacePoint_globEdgeHighY', 'spacePoint_globEdgeHighZ',
    'spacePoint_globEdgeLowX', 'spacePoint_globEdgeLowY', 'spacePoint_globEdgeLowZ',
    'spacePoint_globPosX', 'spacePoint_globPosY', 'spacePoint_globPosZ',
    'spacePoint_driftR', 'spacePoint_time', 'spacePoint_channel',
    'spacePoint_layer', 'spacePoint_stationPhi', 'spacePoint_stationEta',
    'spacePoint_stationIndex', 'spacePoint_technology',
    'spacePoint_covXX', 'spacePoint_covXY', 'spacePoint_covYX', 'spacePoint_covYY',
]

inputs = {'hit': hit_fields}
targets = {'particle': ['truthMuon_pt', 'truthMuon_q', 'truthMuon_eta', 'truthMuon_phi']}

dataset = AtlasMuonDataset(
    dirpath=data_dir,
    inputs=inputs,
    targets=targets,
    num_events=-1,
    event_max_num_particles=10,
)

print(f"Dataset has {len(dataset)} events")

# Collect statistics
all_chord_lengths = []  # in meters
all_sagittas = []  # in meters
all_pt_true = []
all_eta_true = []
all_radii_fit = []  # from circle fit
all_pt_fit = []

n_tracks = 0
max_tracks = 1000

for event_idx in range(min(len(dataset), 10000)):
    if n_tracks >= max_tracks:
        break
        
    hits, particles, num_hits, num_tracks_ev = dataset.load_event(event_idx)
    
    if num_tracks_ev == 0 or num_hits == 0:
        continue
    
    truth_links = hits['spacePoint_truthLink'][:num_hits]
    particle_ids = particles['particle_id'][:num_tracks_ev]
    
    for particle_idx, pid in enumerate(particle_ids):
        if n_tracks >= max_tracks:
            break
            
        hit_mask = truth_links == pid
        n_track_hits = np.sum(hit_mask)
        
        if n_track_hits < 3:
            continue
        
        # Get positions (in meters)
        x = hits['spacePoint_globPosX'][:num_hits][hit_mask]
        y = hits['spacePoint_globPosY'][:num_hits][hit_mask]
        z = hits['spacePoint_globPosZ'][:num_hits][hit_mask]
        r = hits['r'][:num_hits][hit_mask]
        
        # Truth values
        pt_true = particles['truthMuon_pt'][particle_idx]
        eta_true = particles['truthMuon_eta'][particle_idx]
        
        # Sort by r
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        z_sorted = z[sort_idx]
        
        # Compute chord length and sagitta
        r0, z0 = r_sorted[0], z_sorted[0]
        r1, z1 = r_sorted[-1], z_sorted[-1]
        chord_length = np.sqrt((r1 - r0)**2 + (z1 - z0)**2)
        
        if chord_length < 0.1:  # Less than 10 cm
            continue
        
        # Direction
        dr = (r1 - r0) / chord_length
        dz = (z1 - z0) / chord_length
        
        # Find max perpendicular distance (sagitta)
        perp_distances = np.abs((r_sorted - r0) * dz - (z_sorted - z0) * dr)
        sagitta = np.max(perp_distances)
        
        # Compute radius from sagitta
        if sagitta > 1e-6:
            radius_fit = chord_length**2 / (8 * sagitta) + sagitta / 2
        else:
            radius_fit = 1000.0  # Very large
        
        # Estimate pT
        B_eff = 0.4 + 0.12 * np.abs(eta_true)
        pt_fit = 0.3 * B_eff * radius_fit
        pt_fit = np.clip(pt_fit, 1.0, 1000.0)
        
        all_chord_lengths.append(chord_length)
        all_sagittas.append(sagitta)
        all_pt_true.append(pt_true)
        all_eta_true.append(eta_true)
        all_radii_fit.append(radius_fit)
        all_pt_fit.append(pt_fit)
        
        n_tracks += 1

# Convert to arrays
chord_lengths = np.array(all_chord_lengths)
sagittas = np.array(all_sagittas)
pt_true = np.array(all_pt_true)
eta_true = np.array(all_eta_true)
radii_fit = np.array(all_radii_fit)
pt_fit = np.array(all_pt_fit)

print(f"\nAnalyzed {n_tracks} tracks")
print(f"\nChord length [m]: min={chord_lengths.min():.3f}, max={chord_lengths.max():.3f}, mean={chord_lengths.mean():.3f}")
print(f"Sagitta [m]: min={sagittas.min():.6f}, max={sagittas.max():.6f}, mean={sagittas.mean():.6f}")
print(f"Sagitta [mm]: mean={sagittas.mean()*1000:.3f}")
print(f"True pT [GeV]: min={pt_true.min():.1f}, max={pt_true.max():.1f}, mean={pt_true.mean():.1f}")
print(f"Fit radius [m]: min={radii_fit.min():.1f}, max={radii_fit.max():.1f}, mean={radii_fit.mean():.1f}")
print(f"Fit pT [GeV]: min={pt_fit.min():.1f}, max={pt_fit.max():.1f}, mean={pt_fit.mean():.1f}")

# The key physics insight: 
# Expected sagitta for a given pT and track length
# For pT = 50 GeV, B = 0.5 T, the radius should be R = pT / (0.3 * B) = 333 m
# For a chord length L ~ 10 m (typical in MS), sagitta = L^2 / (8*R) = 100 / 2664 = 0.038 m = 38 mm

# But our measured sagitta is much smaller, suggesting the B-field integral is weaker
# or we need a different approach

# Let's compute the "implied" B field from the data
implied_B = 0.3 * pt_true / radii_fit
print(f"\nImplied B-field [T]: min={implied_B.min():.3f}, max={implied_B.max():.3f}, mean={implied_B.mean():.3f}")

# Create diagnostic plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Sagitta vs chord length
ax = axes[0, 0]
ax.scatter(chord_lengths, sagittas * 1000, c=pt_true, s=2, alpha=0.5, cmap='viridis')
ax.set_xlabel('Chord Length [m]')
ax.set_ylabel('Sagitta [mm]')
ax.set_title('Sagitta vs Chord Length (colored by true pT)')
plt.colorbar(ax.collections[0], ax=ax, label='True pT [GeV]')

# 2. Sagitta vs true pT
ax = axes[0, 1]
ax.scatter(pt_true, sagittas * 1000, s=2, alpha=0.5)
ax.set_xlabel('True pT [GeV]')
ax.set_ylabel('Sagitta [mm]')
ax.set_title('Sagitta vs True pT')
# Theoretical line for L=5m, B=0.5T
pt_range = np.linspace(5, 200, 100)
L_avg = chord_lengths.mean()
theoretical_sagitta_mm = (L_avg**2 / (8 * pt_range / (0.3 * 0.5))) * 1000
ax.plot(pt_range, theoretical_sagitta_mm, 'r-', label=f'Theory (L={L_avg:.1f}m, B=0.5T)')
ax.legend()

# 3. Fit pT vs true pT
ax = axes[0, 2]
ax.scatter(pt_true, pt_fit, s=2, alpha=0.5)
ax.plot([0, 200], [0, 200], 'r--', label='y=x')
ax.set_xlabel('True pT [GeV]')
ax.set_ylabel('Fit pT [GeV]')
ax.set_title('Fit pT vs True pT')
ax.legend()
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)

# 4. pT resolution vs true pT
ax = axes[1, 0]
resolution = (pt_true - pt_fit) / pt_true
ax.scatter(pt_true, resolution * 100, s=2, alpha=0.5)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel('True pT [GeV]')
ax.set_ylabel('(pT_true - pT_fit) / pT_true [%]')
ax.set_title('pT Resolution vs True pT')
ax.set_ylim(-100, 100)

# 5. Implied B-field vs eta
ax = axes[1, 1]
ax.scatter(np.abs(eta_true), implied_B, s=2, alpha=0.5)
ax.set_xlabel('|eta|')
ax.set_ylabel('Implied B-field [T]')
ax.set_title('Implied B-field vs |eta|')
ax.set_ylim(0, 3)

# 6. Chord length histogram
ax = axes[1, 2]
ax.hist(chord_lengths, bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Chord Length [m]')
ax.set_ylabel('Count')
ax.set_title('Chord Length Distribution')

plt.tight_layout()
plt.savefig('/shared/tracking/hepattn_muon/src/proxy_evaluation_output/pt_fit_diagnostics.png', dpi=150)
plt.savefig('/shared/tracking/hepattn_muon/src/proxy_evaluation_output/pt_fit_diagnostics.pdf')
print("\nSaved pt_fit_diagnostics.png/pdf")

# Key insight: we need to calibrate the B-field from the data
print("\n" + "="*70)
print("KEY INSIGHT:")
print("="*70)
print("To properly calibrate the pT fit, we need to find the effective B-field")
print("that minimizes the pT resolution. Let's do a simple calibration:")

# Calibrate B-field
best_B = None
best_std = float('inf')
for B_test in np.linspace(0.01, 2.0, 200):
    pt_test = 0.3 * B_test * radii_fit
    pt_test = np.clip(pt_test, 1.0, 1000.0)
    res = (pt_true - pt_test) / pt_true
    valid = np.abs(res) < 2.0
    if np.sum(valid) > 100:
        std = np.std(res[valid])
        if std < best_std:
            best_std = std
            best_B = B_test

print(f"\nOptimal constant B-field: {best_B:.3f} T")
print(f"Resulting pT resolution: {best_std*100:.1f}%")

# With calibrated B
pt_calibrated = 0.3 * best_B * radii_fit
pt_calibrated = np.clip(pt_calibrated, 1.0, 1000.0)
res_calibrated = (pt_true - pt_calibrated) / pt_true
valid = np.abs(res_calibrated) < 2.0
print(f"Mean bias: {np.mean(res_calibrated[valid])*100:.1f}%")
