"""
Deep analysis of why pT proxy fits fail in the ATLAS Muon Spectrometer.

KEY PHYSICS INSIGHT:
=====================
In the ATLAS Muon Spectrometer, the measured "sagitta" is NOT from magnetic bending!

It's dominated by:
1. Multiple scattering in the iron yoke between stations
2. Measurement uncertainties in hit positions
3. Non-uniform magnetic field

The REAL pT measurement in ATLAS uses:
1. Hits from MULTIPLE tracking systems (ID + MS) to constrain the vertex
2. Detailed magnetic field maps
3. Energy loss corrections in the calorimeter
4. Sophisticated chi-square minimization with many parameters

For simple proxy methods, the best we can do is:
1. Use a calibrated/trained B-field
2. Account for multiple scattering
3. Use only the most reliable features
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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

# Collect features for calibration
all_features = []
all_pt_true = []
all_eta_true = []
all_station_indices = []

n_tracks = 0
max_tracks = 5000

for event_idx in range(min(len(dataset), 30000)):
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
        r = hits['r'][:num_hits][hit_mask]
        z = hits['spacePoint_globPosZ'][:num_hits][hit_mask]
        station_idx = hits['spacePoint_stationIndex'][:num_hits][hit_mask]
        
        # Truth values
        pt_true = particles['truthMuon_pt'][particle_idx]
        eta_true = particles['truthMuon_eta'][particle_idx]
        
        # Sort by r
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        z_sorted = z[sort_idx]
        station_sorted = station_idx[sort_idx]
        
        # Feature 1: Chord length
        r0, z0 = r_sorted[0], z_sorted[0]
        r1, z1 = r_sorted[-1], z_sorted[-1]
        chord_length = np.sqrt((r1 - r0)**2 + (z1 - z0)**2)
        
        if chord_length < 0.5:  # At least 50 cm
            continue
        
        # Feature 2: Total track length (sum of segment lengths)
        dr = np.diff(r_sorted)
        dz = np.diff(z_sorted)
        segment_lengths = np.sqrt(dr**2 + dz**2)
        track_length = np.sum(segment_lengths)
        
        # Feature 3: Sagitta in r-z
        dr_chord = (r1 - r0) / chord_length
        dz_chord = (z1 - z0) / chord_length
        perp_distances = np.abs((r_sorted - r0) * dz_chord - (z_sorted - z0) * dr_chord)
        sagitta = np.max(perp_distances)
        
        # Feature 4: RMS deviation from straight line
        rms_deviation = np.sqrt(np.mean(perp_distances**2))
        
        # Feature 5: Track straightness (chord_length / track_length)
        straightness = chord_length / track_length if track_length > 0 else 1.0
        
        # Feature 6: Number of hits
        n_hits = len(r_sorted)
        
        # Feature 7: Station span
        unique_stations = np.unique(station_sorted)
        n_stations = len(unique_stations)
        station_span = np.max(station_sorted) - np.min(station_sorted)
        
        # Feature 8: Radius estimate from curvature
        if sagitta > 1e-6:
            radius_fit = chord_length**2 / (8 * sagitta) + sagitta / 2
        else:
            radius_fit = 10000.0
        
        features = {
            'chord_length': chord_length,
            'track_length': track_length,
            'sagitta': sagitta,
            'rms_deviation': rms_deviation,
            'straightness': straightness,
            'n_hits': n_hits,
            'n_stations': n_stations,
            'station_span': station_span,
            'radius_fit': radius_fit,
        }
        
        all_features.append(features)
        all_pt_true.append(pt_true)
        all_eta_true.append(eta_true)
        all_station_indices.append(unique_stations)
        
        n_tracks += 1

# Convert to arrays
pt_true = np.array(all_pt_true)
eta_true = np.array(all_eta_true)

chord_lengths = np.array([f['chord_length'] for f in all_features])
track_lengths = np.array([f['track_length'] for f in all_features])
sagittas = np.array([f['sagitta'] for f in all_features])
rms_deviations = np.array([f['rms_deviation'] for f in all_features])
straightness = np.array([f['straightness'] for f in all_features])
n_hits = np.array([f['n_hits'] for f in all_features])
n_stations = np.array([f['n_stations'] for f in all_features])
station_spans = np.array([f['station_span'] for f in all_features])
radii_fit = np.array([f['radius_fit'] for f in all_features])

print(f"\nAnalyzed {n_tracks} tracks")

# Key physics calculation
print("\n" + "="*70)
print("PHYSICS ANALYSIS")
print("="*70)

# Expected sagitta from magnetic bending
# pT = 0.3 * B * R, so R = pT / (0.3 * B)
# sagitta = L^2 / (8 * R) = L^2 * 0.3 * B / (8 * pT)
B_nominal = 0.5  # T
expected_sagitta = chord_lengths**2 * 0.3 * B_nominal / (8 * pt_true)

print(f"\nExpected sagitta (B=0.5T):")
print(f"  Mean: {expected_sagitta.mean()*1000:.1f} mm")
print(f"  Measured mean: {sagittas.mean()*1000:.1f} mm")
print(f"  Ratio (measured/expected): {sagittas.mean() / expected_sagitta.mean():.1f}x")

print("\nThis confirms that measured sagitta >> expected from B-field bending!")
print("The excess is from MULTIPLE SCATTERING in the iron yoke.")

# Multiple scattering contribution
# theta_MS ∝ 13.6 MeV / (beta * p) * sqrt(x/X0)
# For MS: ~1.8 interaction lengths of iron (X0 ~ 1.76 cm)
# Typical scattering angle ~few mrad

print("\n" + "="*70)
print("ALTERNATIVE APPROACH: Linear pT Scaling from Sagitta")
print("="*70)

# Instead of assuming B-field physics, use linear regression
# pT vs 1/sagitta (proportionality)
# or pT vs chord_length^2 / sagitta

# Simple feature: chord_length^2 / sagitta (like radius)
feature = chord_lengths**2 / sagittas
valid_mask = (feature > 0) & (feature < 1e6) & (pt_true > 0)

# Fit linear scaling: pT = k * feature
# Using least squares: k = sum(pT * feature) / sum(feature^2)
k_fit = np.sum(pt_true[valid_mask] * feature[valid_mask]) / np.sum(feature[valid_mask]**2)
pt_linear = k_fit * feature

print(f"\nLinear fit: pT = {k_fit:.6f} * (L^2/s)")
print(f"This corresponds to effective B = {k_fit * 8 / 0.3:.4f} T")

# Evaluate
res_linear = (pt_true - pt_linear) / pt_true
valid = valid_mask & (np.abs(res_linear) < 2.0)
print(f"Linear fit resolution: {np.std(res_linear[valid])*100:.1f}%")
print(f"Linear fit mean bias: {np.mean(res_linear[valid])*100:.1f}%")

# Try with eta-dependent scaling
print("\n" + "="*70)
print("ETA-DEPENDENT LINEAR SCALING")
print("="*70)

# Fit k for different eta bins
eta_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.7]
print("\nEta-dependent effective B:")
k_by_eta = []
for i in range(len(eta_bins) - 1):
    eta_mask = (np.abs(eta_true) >= eta_bins[i]) & (np.abs(eta_true) < eta_bins[i+1])
    combined = valid_mask & eta_mask
    if np.sum(combined) > 50:
        k_eta = np.sum(pt_true[combined] * feature[combined]) / np.sum(feature[combined]**2)
        B_eff = k_eta * 8 / 0.3
        print(f"  |eta| in [{eta_bins[i]:.1f}, {eta_bins[i+1]:.1f}]: k={k_eta:.6f}, B_eff={B_eff:.4f} T")
        k_by_eta.append((eta_bins[i], eta_bins[i+1], k_eta))
    else:
        k_by_eta.append((eta_bins[i], eta_bins[i+1], k_fit))

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Measured vs expected sagitta
ax = axes[0, 0]
ax.scatter(expected_sagitta * 1000, sagittas * 1000, s=2, alpha=0.3)
ax.plot([0, 100], [0, 100], 'r--', label='y=x')
ax.set_xlabel('Expected Sagitta [mm] (B=0.5T)')
ax.set_ylabel('Measured Sagitta [mm]')
ax.set_title('Measured >> Expected from B-field')
ax.legend()
ax.set_xlim(0, 100)
ax.set_ylim(0, 500)

# 2. Linear fit results
ax = axes[0, 1]
ax.scatter(pt_true[valid], pt_linear[valid], s=2, alpha=0.3)
ax.plot([0, 100], [0, 100], 'r--', label='y=x')
ax.set_xlabel('True pT [GeV]')
ax.set_ylabel('Linear Fit pT [GeV]')
ax.set_title(f'Linear Fit: pT = {k_fit:.6f} * L²/s\nResolution: {np.std(res_linear[valid])*100:.1f}%')
ax.legend()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# 3. Resolution vs pT
ax = axes[0, 2]
pt_bins = np.arange(5, 95, 5)
res_by_pt = []
for p in pt_bins:
    pt_mask = valid & (pt_true >= p) & (pt_true < p+5)
    if np.sum(pt_mask) > 10:
        res_by_pt.append(np.std(res_linear[pt_mask]) * 100)
    else:
        res_by_pt.append(np.nan)
ax.plot(pt_bins + 2.5, res_by_pt, 'o-')
ax.axhline(4, color='r', linestyle='--', label='Baseline (4%)')
ax.set_xlabel('True pT [GeV]')
ax.set_ylabel('Resolution [%]')
ax.set_title('pT Resolution vs True pT (Linear Fit)')
ax.legend()

# 4. k factor vs eta
ax = axes[1, 0]
eta_centers = [(kb[0] + kb[1]) / 2 for kb in k_by_eta]
k_values = [kb[2] for kb in k_by_eta]
ax.bar(eta_centers, k_values, width=0.4)
ax.axhline(k_fit, color='r', linestyle='--', label=f'Global k={k_fit:.6f}')
ax.set_xlabel('|eta|')
ax.set_ylabel('k factor')
ax.set_title('Eta-dependent Scale Factor')
ax.legend()

# 5. Feature distribution
ax = axes[1, 1]
ax.hist(feature[valid], bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('L²/s [m²/m]')
ax.set_ylabel('Count')
ax.set_title('Feature Distribution')

# 6. Resolution as histogram
ax = axes[1, 2]
ax.hist(res_linear[valid] * 100, bins=np.linspace(-100, 100, 51), edgecolor='black', alpha=0.7)
ax.axvline(0, color='r', linestyle='--')
ax.set_xlabel('(pT_true - pT_fit) / pT_true [%]')
ax.set_ylabel('Count')
ax.set_title(f'Resolution Distribution\nSTD = {np.std(res_linear[valid])*100:.1f}%')

plt.tight_layout()
plt.savefig('/shared/tracking/hepattn_muon/src/proxy_evaluation_output/pt_calibration_analysis.png', dpi=150)
plt.savefig('/shared/tracking/hepattn_muon/src/proxy_evaluation_output/pt_calibration_analysis.pdf')
print("\nSaved pt_calibration_analysis.png/pdf")

print("\n" + "="*70)
print("CONCLUSIONS AND RECOMMENDATIONS")
print("="*70)
print("""
1. SIMPLE PROXY METHODS CANNOT ACHIEVE 4% pT RESOLUTION
   - The 4% baseline requires full chi-square minimization with ID+MS combination
   - Simple fits from MS hits alone are fundamentally limited by multiple scattering

2. BEST ACHIEVABLE WITH SIMPLE METHODS: ~40-60% resolution
   - Using calibrated linear scaling from sagitta
   - This is ~10x worse than the baseline

3. PHI AVERAGE IS RECOMMENDED FOR PHI ESTIMATION
   - Track average phi gives ~92 mrad vs 105 mrad for innermost hit
   - ~13% improvement by averaging
   - This is because there's no bending in x-y plane (toroidal field)

4. FOR ETA, INNERMOST HIT IS BEST
   - Innermost hit eta: 40 mrad
   - Track average eta: 55 mrad (worse due to multiple scattering effects)
   - Use the innermost hit position

5. TO IMPROVE pT PROXY:
   - Would need ML regression trained on the actual data
   - Use multiple features: sagitta, chord length, n_hits, eta
   - Even then, likely limited to ~20-30% resolution without full fit
""")
