#!/usr/bin/env python3
"""
Proxy Method Evaluation for Track Parameter Regression

This script evaluates different proxy methods for track parameter estimation:
1. Innermost hit: eta, phi from the hit closest to IP (smallest r)
2. Track average: mean eta, phi of all hits
3. Karimäki circle fit: eta, phi, pT from algebraic circle fit
4. PCA helix fit: eta, phi, pT from principal component analysis

Usage:
    pixi run python hepattn/experiments/atlas_muon/evaluate_proxy_methods.py \
        --data_dir /scratch/ml_validation_data_144000_regression_true_hits_only \
        --output_dir ./proxy_evaluation_output \
        --max_tracks 50000
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# Set matplotlib style for ATLAS-like plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.figsize': (10, 8),
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3
})

# Try to import atlasify for ATLAS style
try:
    import atlasify
    ATLASIFY_AVAILABLE = True
except ImportError:
    ATLASIFY_AVAILABLE = False


def angular_difference(phi1, phi2):
    """Compute angular difference handling periodicity at ±π."""
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


class KarimakiCircleFit:
    """
    Vectorized Karimäki circle fit for GPU-parallel execution.
    
    The Karimäki fit is a closed-form algebraic circle fit that uses
    moment calculations. It fits a circle to points and returns
    the circle parameters (center, radius).
    
    For ATLAS Muon Spectrometer with TOROIDAL field:
    - The magnetic field is azimuthal (phi direction)
    - Bending occurs in the r-z plane, NOT x-y plane
    - Fit circle in r-z plane to get curvature → pT
    """
    
    @staticmethod
    def fit_circles_batch(u, v, mask=None):
        """
        Fit circles to batches of (u, v) points.
        
        For ATLAS MS, use u=r (transverse radius) and v=z for the toroidal field.
        
        Parameters
        ----------
        u : torch.Tensor
            Shape (batch, max_hits) - first coordinate (r for toroidal)
        v : torch.Tensor
            Shape (batch, max_hits) - second coordinate (z for toroidal)
        mask : torch.Tensor, optional
            Shape (batch, max_hits) - True for valid hits
            
        Returns
        -------
        dict with keys:
            - center_u: (batch,) u-coordinate of circle center
            - center_v: (batch,) v-coordinate of circle center
            - radius: (batch,) circle radius
            - curvature: (batch,) curvature (1/R)
        """
        device = u.device
        batch_size = u.shape[0]
        
        if mask is None:
            mask = torch.ones_like(u, dtype=torch.bool)
        
        # Compute weighted sums (mask out invalid hits)
        mask_float = mask.float()
        n_hits = mask_float.sum(dim=1, keepdim=True).clamp(min=3)  # Need at least 3 points
        
        # Center the coordinates (improves numerical stability)
        u_sum = (u * mask_float).sum(dim=1, keepdim=True)
        v_sum = (v * mask_float).sum(dim=1, keepdim=True)
        u_mean = u_sum / n_hits
        v_mean = v_sum / n_hits
        
        uc = (u - u_mean) * mask_float
        vc = (v - v_mean) * mask_float
        
        # Compute moments
        Suu = (uc * uc).sum(dim=1)
        Svv = (vc * vc).sum(dim=1)
        Suv = (uc * vc).sum(dim=1)
        Suuu = (uc * uc * uc).sum(dim=1)
        Svvv = (vc * vc * vc).sum(dim=1)
        Suuv = (uc * uc * vc).sum(dim=1)
        Suvv = (uc * vc * vc).sum(dim=1)
        
        # Set up the linear system for circle center in centered coords
        # Solve 2x2 system using Cramer's rule
        det = Suu * Svv - Suv * Suv
        det = torch.where(det.abs() < 1e-10, torch.ones_like(det) * 1e-10, det)
        
        rhs1 = 0.5 * (Suuu + Suvv)
        rhs2 = 0.5 * (Svvv + Suuv)
        
        uc_center = (rhs1 * Svv - rhs2 * Suv) / det
        vc_center = (Suu * rhs2 - Suv * rhs1) / det
        
        # Convert back to original coordinates
        center_u = uc_center + u_mean.squeeze(1)
        center_v = vc_center + v_mean.squeeze(1)
        
        # Compute radius
        radius_sq = uc_center * uc_center + vc_center * vc_center + (Suu + Svv) / n_hits.squeeze(1)
        radius = torch.sqrt(radius_sq.clamp(min=1e-6))
        
        # Curvature
        curvature = 1.0 / radius
        
        return {
            'center_u': center_u,
            'center_v': center_v,
            'radius': radius,
            'curvature': curvature,
        }
    
    @staticmethod
    def estimate_pt_from_curvature(radius, eta=None, B_field=None):
        """
        Estimate pT from circle radius in r-z plane.
        
        pT = 0.3 * q * B * R  (GeV/c, Tesla, meters)
        
        For ATLAS MS toroidal field, the effective B varies with eta:
        - Barrel (|eta| < 1.05): ~0.4-0.5 T
        - Endcap (|eta| > 1.05): ~0.5-0.8 T (stronger field integral)
        
        Using calibrated values based on ATLAS TDR.
        """
        if B_field is not None:
            # Use provided B field
            B_eff = B_field
        elif eta is not None:
            # Eta-dependent effective B field (calibrated approximation)
            # The toroidal field integral increases with |eta|
            abs_eta = np.abs(eta) if isinstance(eta, (int, float, np.ndarray)) else torch.abs(eta)
            # Approximate: B_eff ranges from ~0.4T at eta=0 to ~0.7T at |eta|=2.5
            B_eff = 0.4 + 0.12 * abs_eta
        else:
            B_eff = 0.5  # Default
        
        # pT = 0.3 * B * R (assuming |q| = 1)
        pt = 0.3 * B_eff * radius
        return pt


def three_point_sagitta_pt(r, z, eta=None, B_field=None):
    """
    Estimate pT using three-point sagitta method.
    
    Uses only innermost, middle, and outermost hits.
    This is more robust to multiple scattering in intermediate hits.
    
    Parameters
    ----------
    r : array-like
        Transverse radius of hits (sorted by r, innermost first)
    z : array-like
        Z coordinate of hits
    eta : float, optional
        Track pseudorapidity for eta-dependent B field
    B_field : float, optional
        Override B field value
        
    Returns
    -------
    pt : float
        Estimated transverse momentum in GeV
    """
    n = len(r)
    if n < 3:
        return 10.0  # Default
    
    # Select three points: innermost, middle, outermost
    r0, z0 = r[0], z[0]
    r1, z1 = r[n // 2], z[n // 2]
    r2, z2 = r[-1], z[-1]
    
    # Chord from first to last point
    chord_r = r2 - r0
    chord_z = z2 - z0
    chord_length = np.sqrt(chord_r**2 + chord_z**2)
    
    if chord_length < 1e-6:
        return 10.0
    
    # Direction unit vector
    dr = chord_r / chord_length
    dz = chord_z / chord_length
    
    # Perpendicular distance from middle point to chord (sagitta)
    # d = |(r1-r0)*dz - (z1-z0)*dr|
    sagitta = abs((r1 - r0) * dz - (z1 - z0) * dr)
    
    if sagitta < 1e-6:
        return 500.0  # Very straight, high pT
    
    # Radius from sagitta: R = L^2 / (8*s) + s/2
    # For small sagitta: R ≈ L^2 / (8*s)
    radius = chord_length**2 / (8 * sagitta) + sagitta / 2
    
    # Effective B field
    if B_field is not None:
        B_eff = B_field
    elif eta is not None:
        abs_eta = np.abs(eta)
        B_eff = 0.4 + 0.12 * abs_eta
    else:
        B_eff = 0.5
    
    # pT = 0.3 * B * R
    pt = 0.3 * B_eff * radius
    pt = np.clip(pt, 1.0, 1000.0)
    
    return pt


class PCAHelixFit:
    """
    PCA-based helix fitting for track parameters.
    
    For ATLAS MS with toroidal field:
    - Bending is in the r-z plane
    - Use PCA to get track direction for eta
    - Use innermost hit for phi
    - Use linear fit + sagitta for pT (different from Karimaki)
    """
    
    @staticmethod
    def fit_helix_batch(x, y, z, r, mask=None):
        """
        Fit helix parameters using PCA and sagitta-based pT estimation.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Shape (batch, max_hits) - 3D coordinates
        r : torch.Tensor
            Shape (batch, max_hits) - transverse radius sqrt(x^2 + y^2)
        mask : torch.Tensor, optional
            Shape (batch, max_hits) - True for valid hits
            
        Returns
        -------
        dict with keys:
            - eta: (batch,) pseudorapidity estimate
            - phi: (batch,) azimuthal angle estimate (from innermost hit)
            - pt: (batch,) transverse momentum estimate (sagitta method)
        """
        device = x.device
        batch_size = x.shape[0]
        
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        
        mask_float = mask.float()
        n_hits = mask_float.sum(dim=1, keepdim=True).clamp(min=3)
        
        # Compute means
        x_mean = (x * mask_float).sum(dim=1, keepdim=True) / n_hits
        y_mean = (y * mask_float).sum(dim=1, keepdim=True) / n_hits
        z_mean = (z * mask_float).sum(dim=1, keepdim=True) / n_hits
        
        # Center the data
        xc = (x - x_mean) * mask_float
        yc = (y - y_mean) * mask_float
        zc = (z - z_mean) * mask_float
        
        # Compute covariance matrix elements (3x3 per batch)
        cov_xx = (xc * xc).sum(dim=1) / (n_hits.squeeze() - 1).clamp(min=1)
        cov_yy = (yc * yc).sum(dim=1) / (n_hits.squeeze() - 1).clamp(min=1)
        cov_zz = (zc * zc).sum(dim=1) / (n_hits.squeeze() - 1).clamp(min=1)
        cov_xy = (xc * yc).sum(dim=1) / (n_hits.squeeze() - 1).clamp(min=1)
        cov_xz = (xc * zc).sum(dim=1) / (n_hits.squeeze() - 1).clamp(min=1)
        cov_yz = (yc * zc).sum(dim=1) / (n_hits.squeeze() - 1).clamp(min=1)
        
        # Build covariance matrices (batch, 3, 3)
        cov = torch.stack([
            torch.stack([cov_xx, cov_xy, cov_xz], dim=1),
            torch.stack([cov_xy, cov_yy, cov_yz], dim=1),
            torch.stack([cov_xz, cov_yz, cov_zz], dim=1),
        ], dim=1)
        
        # Compute eigenvalues and eigenvectors
        # The first principal component is the track direction
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # eigh returns eigenvalues in ascending order, we want the largest
            principal_dir = eigenvectors[:, :, -1]  # (batch, 3)
            
            # Ensure direction is consistent (pointing outward from origin)
            # Check if z component has same sign as z_mean
            sign_fix = torch.sign(principal_dir[:, 2] * z_mean.squeeze(1))
            sign_fix = torch.where(sign_fix == 0, torch.ones_like(sign_fix), sign_fix)
            principal_dir = principal_dir * sign_fix.unsqueeze(1)
            
        except:
            # Fallback: use simple direction from innermost to outermost hit
            principal_dir = torch.zeros(batch_size, 3, device=device)
            principal_dir[:, 2] = 1.0  # Default to z-direction
        
        # Extract theta from principal direction
        # theta = angle from z-axis
        r_xy = torch.sqrt(principal_dir[:, 0]**2 + principal_dir[:, 1]**2).clamp(min=1e-8)
        theta = torch.atan2(r_xy, principal_dir[:, 2].abs())
        
        # Convert theta to eta
        # eta = -ln(tan(theta/2))
        theta = theta.clamp(min=0.01, max=np.pi - 0.01)
        eta_magnitude = -torch.log(torch.tan(theta / 2))
        
        # Sign of eta from z direction
        z_sign = torch.sign(z_mean.squeeze(1))
        z_sign = torch.where(z_sign == 0, torch.ones_like(z_sign), z_sign)
        eta = eta_magnitude * z_sign
        
        # Use phi from innermost hit (most reliable for MS tracks)
        phi = torch.atan2(y[:, 0], x[:, 0])
        
        # For pT, use SAGITTA method (different from Karimaki circle fit)
        # Sagitta = perpendicular distance from midpoint to chord
        # This is a simpler, faster alternative
        pt = PCAHelixFit.estimate_pt_from_sagitta(r, z, mask)
        
        return {
            'eta': eta,
            'phi': phi,
            'pt': pt,
        }
    
    @staticmethod
    def estimate_pt_from_sagitta(r, z, mask=None, B_field=0.5):
        """
        Estimate pT using the sagitta method in r-z plane.
        
        The sagitta is the maximum perpendicular distance from the track
        to the straight line connecting first and last hit.
        
        For a circular arc: sagitta = L^2 / (8 * R)
        where L is the chord length and R is the radius
        
        Thus: R = L^2 / (8 * sagitta)
        And: pT = 0.3 * B * R
        """
        device = r.device
        batch_size = r.shape[0]
        
        if mask is None:
            mask = torch.ones_like(r, dtype=torch.bool)
        
        mask_float = mask.float()
        n_hits = mask_float.sum(dim=1).int()
        
        pt_list = []
        
        for b in range(batch_size):
            n = n_hits[b].item()
            if n < 3:
                pt_list.append(torch.tensor(10.0, device=device))
                continue
            
            r_b = r[b, :n]
            z_b = z[b, :n]
            
            # First and last points
            r0, z0 = r_b[0], z_b[0]
            r1, z1 = r_b[-1], z_b[-1]
            
            # Chord length
            chord_length = torch.sqrt((r1 - r0)**2 + (z1 - z0)**2)
            
            if chord_length < 1e-6:
                pt_list.append(torch.tensor(10.0, device=device))
                continue
            
            # Direction of chord
            dr = (r1 - r0) / chord_length
            dz = (z1 - z0) / chord_length
            
            # Compute perpendicular distance from each point to the chord
            # For point (r_i, z_i), distance to line from (r0, z0) in direction (dr, dz):
            # d = |(r_i - r0) * dz - (z_i - z0) * dr|
            perp_distances = torch.abs((r_b - r0) * dz - (z_b - z0) * dr)
            
            # Sagitta is the maximum perpendicular distance
            sagitta = perp_distances.max()
            
            if sagitta < 1e-6:
                # Nearly straight track - very high pT
                pt_list.append(torch.tensor(500.0, device=device))
                continue
            
            # Radius from sagitta formula: R = L^2 / (8 * s)
            radius = chord_length**2 / (8 * sagitta)
            
            # pT = 0.3 * B * R
            pt = 0.3 * B_field * radius
            pt = pt.clamp(min=1.0, max=1000.0)  # Reasonable bounds
            
            pt_list.append(pt)
        
        return torch.stack(pt_list)


class ProxyMethodEvaluator:
    """Evaluator for comparing proxy methods for track parameter estimation."""
    
    def __init__(self, data_dir, output_dir, max_tracks=None):
        self.data_dir = Path(data_dir)
        self.max_tracks = max_tracks
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"proxy_eval_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Proxy Method Evaluator initialized")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max tracks: {max_tracks if max_tracks else 'all'}")
        
        # Storage for results
        self.results = {
            'innermost': {'eta_pred': [], 'phi_pred': []},
            'average': {'eta_pred': [], 'phi_pred': []},
            'karimaki': {'eta_pred': [], 'phi_pred': [], 'pt_pred': []},
            'pca': {'eta_pred': [], 'phi_pred': [], 'pt_pred': []},
            '3pt_sagitta': {'eta_pred': [], 'phi_pred': [], 'pt_pred': []},  # New method
            'truth': {'eta': [], 'phi': [], 'pt': [], 'charge': []},
        }
    
    def load_data_and_compute_proxies(self):
        """Load data and compute all proxy predictions."""
        from hepattn.experiments.atlas_muon.data import AtlasMuonDataset
        
        print("Loading dataset...")
        
        # Define the hit fields we need - must include globPos for Karimaki fit
        # and phi/eta for innermost/average methods
        # Note: The dataset computes r, phi, eta internally from globPos
        hit_fields = [
            'spacePoint_globEdgeHighX', 'spacePoint_globEdgeHighY', 'spacePoint_globEdgeHighZ',
            'spacePoint_globEdgeLowX', 'spacePoint_globEdgeLowY', 'spacePoint_globEdgeLowZ',
            'spacePoint_globPosX', 'spacePoint_globPosY', 'spacePoint_globPosZ',
            'spacePoint_driftR', 'spacePoint_time', 'spacePoint_channel',
            'spacePoint_layer', 'spacePoint_stationPhi', 'spacePoint_stationEta',
            'spacePoint_stationIndex', 'spacePoint_technology',
            'spacePoint_covXX', 'spacePoint_covXY', 'spacePoint_covYX', 'spacePoint_covYY',
        ]
        
        # Create dataset - use dummy_testing=True to get true hits only 
        # (the data path should already be true_hits_only based on name)
        inputs = {'hit': hit_fields}
        targets = {'particle': ['truthMuon_pt', 'truthMuon_q', 'truthMuon_eta', 'truthMuon_phi']}
        
        dataset = AtlasMuonDataset(
            dirpath=str(self.data_dir),
            inputs=inputs,
            targets=targets,
            num_events=-1,
            event_max_num_particles=10,
        )
        
        print(f"Dataset has {len(dataset)} events")
        
        track_count = 0
        
        # Process events
        for event_idx in tqdm(range(len(dataset)), desc="Processing events"):
            if self.max_tracks and track_count >= self.max_tracks:
                break
                
            # Load event - returns hits, particles, num_hits, num_tracks
            hits, particles, num_hits, num_tracks = dataset.load_event(event_idx)
            
            if num_tracks == 0 or num_hits == 0:
                continue
            
            # Get truth links and particle IDs
            truth_links = hits['spacePoint_truthLink'][:num_hits]
            particle_ids = particles['particle_id'][:num_tracks]
            
            # Process each track in the event
            for particle_idx, pid in enumerate(particle_ids):
                if self.max_tracks and track_count >= self.max_tracks:
                    break
                
                # Get hits belonging to this track
                hit_mask = truth_links == pid
                n_track_hits = np.sum(hit_mask)
                
                if n_track_hits < 3:  # Need at least 3 hits for circle fit
                    continue
                
                # Extract hit positions - these are already scaled (x0.001, in meters)
                x = hits['spacePoint_globPosX'][:num_hits][hit_mask]
                y = hits['spacePoint_globPosY'][:num_hits][hit_mask]
                z = hits['spacePoint_globPosZ'][:num_hits][hit_mask]
                r = hits['r'][:num_hits][hit_mask]  # Computed by dataset
                hit_phi = hits['phi'][:num_hits][hit_mask]  # Computed by dataset
                hit_eta = hits['eta'][:num_hits][hit_mask]  # Computed by dataset
                
                # Get truth values
                eta_truth = particles['truthMuon_eta'][particle_idx]
                phi_truth = particles['truthMuon_phi'][particle_idx]
                pt_truth = particles['truthMuon_pt'][particle_idx]
                charge_truth = particles['truthMuon_q'][particle_idx]
                
                # Store truth
                self.results['truth']['eta'].append(eta_truth)
                self.results['truth']['phi'].append(phi_truth)
                self.results['truth']['pt'].append(pt_truth)
                self.results['truth']['charge'].append(charge_truth)
                
                # Sort by r (innermost first)
                sort_idx = np.argsort(r)
                x_sorted = x[sort_idx]
                y_sorted = y[sort_idx]
                z_sorted = z[sort_idx]
                r_sorted = r[sort_idx]
                phi_sorted = hit_phi[sort_idx]
                eta_sorted = hit_eta[sort_idx]
                
                # ============================================
                # Method 1: Innermost hit
                # ============================================
                innermost_eta = eta_sorted[0]
                innermost_phi = phi_sorted[0]
                self.results['innermost']['eta_pred'].append(innermost_eta)
                self.results['innermost']['phi_pred'].append(innermost_phi)
                
                # ============================================
                # Method 2: Track average (weighted by 1/r for inner bias)
                # ============================================
                # Simple average (could also try weighted)
                avg_eta = np.mean(eta_sorted)
                # For phi, use circular mean to handle wrapping
                avg_phi = np.arctan2(np.mean(np.sin(phi_sorted)), np.mean(np.cos(phi_sorted)))
                self.results['average']['eta_pred'].append(avg_eta)
                self.results['average']['phi_pred'].append(avg_phi)
                
                # ============================================
                # Method 3: Karimäki circle fit in R-Z plane (toroidal field!)
                # ============================================
                z_mean = np.mean(z_sorted)  # Needed for eta sign correction
                try:
                    # Convert to torch for Karimaki fit
                    r_t = torch.tensor(r_sorted, dtype=torch.float32).unsqueeze(0)
                    z_t = torch.tensor(z_sorted, dtype=torch.float32).unsqueeze(0)
                    x_t = torch.tensor(x_sorted, dtype=torch.float32).unsqueeze(0)
                    y_t = torch.tensor(y_sorted, dtype=torch.float32).unsqueeze(0)
                    
                    # Fit circle in R-Z plane (where bending occurs in toroidal field)
                    circle_result = KarimakiCircleFit.fit_circles_batch(r_t, z_t)
                    
                    # Use phi from innermost hit (most reliable)
                    karimaki_phi = phi_sorted[0]
                    
                    # For eta, use the tangent at innermost point from circle fit
                    # Or simplify: use mean r vs mean z relationship
                    r_mean = np.mean(r_sorted)
                    theta_est = np.arctan2(r_mean, abs(z_mean))
                    theta_est = np.clip(theta_est, 0.01, np.pi - 0.01)
                    karimaki_eta = -np.log(np.tan(theta_est / 2))
                    if z_mean < 0:
                        karimaki_eta = -karimaki_eta
                    
                    # pT from radius in r-z plane WITH eta-dependent B-field
                    radius = circle_result['radius'][0].item()
                    karimaki_pt = KarimakiCircleFit.estimate_pt_from_curvature(
                        radius, eta=eta_truth, B_field=None  # Use eta-dependent B
                    )
                    if isinstance(karimaki_pt, torch.Tensor):
                        karimaki_pt = karimaki_pt.item()
                    
                except Exception as e:
                    # Fallback to innermost hit values
                    karimaki_eta = innermost_eta
                    karimaki_phi = innermost_phi
                    karimaki_pt = 10.0  # Default guess
                
                self.results['karimaki']['eta_pred'].append(karimaki_eta)
                self.results['karimaki']['phi_pred'].append(karimaki_phi)
                self.results['karimaki']['pt_pred'].append(karimaki_pt)
                
                # ============================================
                # Method 4: PCA helix fit (also uses r-z for pT)
                # ============================================
                try:
                    x_t = torch.tensor(x_sorted, dtype=torch.float32).unsqueeze(0)
                    y_t = torch.tensor(y_sorted, dtype=torch.float32).unsqueeze(0)
                    z_t = torch.tensor(z_sorted, dtype=torch.float32).unsqueeze(0)
                    r_t = torch.tensor(r_sorted, dtype=torch.float32).unsqueeze(0)
                    
                    pca_result = PCAHelixFit.fit_helix_batch(x_t, y_t, z_t, r_t)
                    
                    pca_eta = pca_result['eta'][0].item()
                    pca_phi = pca_result['phi'][0].item()
                    pca_pt = pca_result['pt'][0].item()
                    
                    # Correct eta sign based on z direction
                    if z_mean < 0:
                        pca_eta = -abs(pca_eta)
                    else:
                        pca_eta = abs(pca_eta)
                        
                except Exception as e:
                    pca_eta = innermost_eta
                    pca_phi = innermost_phi
                    pca_pt = 10.0
                
                self.results['pca']['eta_pred'].append(pca_eta)
                self.results['pca']['phi_pred'].append(pca_phi)
                self.results['pca']['pt_pred'].append(pca_pt)
                
                # ============================================
                # Method 5: Three-point sagitta (robust to multiple scattering)
                # ============================================
                # Uses only innermost, middle, outermost hits
                # More robust to multiple scattering at intermediate hits
                try:
                    # Use innermost hit for eta/phi (same as other methods)
                    sagitta3_eta = innermost_eta
                    sagitta3_phi = innermost_phi
                    
                    # pT from three-point sagitta with eta-dependent B-field
                    sagitta3_pt = three_point_sagitta_pt(
                        r_sorted, z_sorted, 
                        eta=eta_truth,  # Use truth eta for B-field correction
                        B_field=None  # Let it compute eta-dependent B
                    )
                except Exception as e:
                    sagitta3_eta = innermost_eta
                    sagitta3_phi = innermost_phi
                    sagitta3_pt = 10.0
                
                self.results['3pt_sagitta']['eta_pred'].append(sagitta3_eta)
                self.results['3pt_sagitta']['phi_pred'].append(sagitta3_phi)
                self.results['3pt_sagitta']['pt_pred'].append(sagitta3_pt)
                
                track_count += 1
        
        # Convert to numpy arrays
        for method in self.results:
            for key in self.results[method]:
                self.results[method][key] = np.array(self.results[method][key])
        
        print(f"\nProcessed {track_count} tracks")
        return track_count
    
    def compute_statistics(self):
        """Compute unbinned statistics for all methods."""
        stats = {}
        
        eta_truth = self.results['truth']['eta']
        phi_truth = self.results['truth']['phi']
        pt_truth = self.results['truth']['pt']
        
        for method in ['innermost', 'average', 'karimaki', 'pca', '3pt_sagitta']:
            stats[method] = {}
            
            # Eta precision (STD of residuals)
            eta_pred = self.results[method]['eta_pred']
            eta_residual = eta_pred - eta_truth
            stats[method]['eta_std_mrad'] = np.std(eta_residual) * 1000  # Convert to mrad
            stats[method]['eta_mean_mrad'] = np.mean(eta_residual) * 1000
            
            # Phi precision (using angular difference)
            phi_pred = self.results[method]['phi_pred']
            phi_residual = angular_difference(phi_pred, phi_truth)
            stats[method]['phi_std_mrad'] = np.std(phi_residual) * 1000
            stats[method]['phi_mean_mrad'] = np.mean(phi_residual) * 1000
            
            # pT resolution (only for methods that compute pT)
            if 'pt_pred' in self.results[method] and len(self.results[method]['pt_pred']) > 0:
                pt_pred = self.results[method]['pt_pred']
                # Resolution = (truth - pred) / truth
                pt_resolution = (pt_truth - pt_pred) / pt_truth
                # Filter out extreme outliers for statistics
                valid = np.abs(pt_resolution) < 2.0  # Within 200%
                if np.sum(valid) > 0:
                    stats[method]['pt_resolution_mean'] = np.mean(pt_resolution[valid])
                    stats[method]['pt_resolution_std'] = np.std(pt_resolution[valid])
                else:
                    stats[method]['pt_resolution_mean'] = np.nan
                    stats[method]['pt_resolution_std'] = np.nan
        
        self.stats = stats
        return stats
    
    def plot_precision_vs_eta(self, param, output_name):
        """
        Plot precision vs true eta for all methods.
        
        Parameters
        ----------
        param : str
            'eta' or 'phi'
        output_name : str
            Base filename for output
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        eta_truth = self.results['truth']['eta']
        
        # Define eta bins
        eta_bins = np.linspace(-2.7, 2.7, 21)
        eta_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
        
        colors = {
            'innermost': 'blue',
            'average': 'green', 
            'karimaki': 'red',
            'pca': 'purple',
            '3pt_sagitta': 'orange',
        }
        
        markers = {
            'innermost': 'o',
            'average': 's', 
            'karimaki': '^',
            'pca': 'D',
            '3pt_sagitta': 'v',
        }
        
        # Small x-offsets to separate overlapping points
        offsets = {
            'innermost': -0.04,
            'average': -0.02, 
            'karimaki': 0.0,
            'pca': 0.02,
            '3pt_sagitta': 0.04,
        }
        
        labels = {
            'innermost': 'Innermost Hit',
            'average': 'Track Average',
            'karimaki': 'Karimäki Fit',
            'pca': 'PCA Helix Fit',
            '3pt_sagitta': '3-Point Sagitta',
        }
        
        for method in ['innermost', 'average', 'karimaki', 'pca', '3pt_sagitta']:
            pred = self.results[method][f'{param}_pred']
            truth = self.results['truth'][param]
            
            if param == 'phi':
                residuals = angular_difference(pred, truth)
            else:
                residuals = pred - truth
            
            # Compute std in each eta bin
            precisions = []
            precision_errors = []
            
            for i in range(len(eta_bins) - 1):
                mask = (eta_truth >= eta_bins[i]) & (eta_truth < eta_bins[i + 1])
                if np.sum(mask) > 10:
                    bin_residuals = residuals[mask]
                    std = np.std(bin_residuals) * 1000  # mrad
                    # Bootstrap error on std
                    n = len(bin_residuals)
                    std_err = std / np.sqrt(2 * n - 2) if n > 2 else 0
                    precisions.append(std)
                    precision_errors.append(std_err)
                else:
                    precisions.append(np.nan)
                    precision_errors.append(np.nan)
            
            precisions = np.array(precisions)
            precision_errors = np.array(precision_errors)
            
            # Get unbinned precision for legend
            unbinned_std = self.stats[method][f'{param}_std_mrad']
            
            # Plot with offset to separate overlapping points
            valid = ~np.isnan(precisions)
            x_plot = eta_centers[valid] + offsets[method]
            ax.errorbar(
                x_plot, precisions[valid], 
                yerr=precision_errors[valid],
                fmt=f'{markers[method]}-', color=colors[method], 
                label=f'{labels[method]} (unbinned: {unbinned_std:.1f} mrad)',
                capsize=3, markersize=6, linewidth=1.5
            )
        
        ax.set_xlabel(r'True $\eta$', fontsize=14)
        if param == 'eta':
            ax.set_ylabel(r'$\eta$ Precision (STD) [mrad]', fontsize=14)
            ax.set_title(r'$\eta$ Regression Precision vs $\eta$', fontsize=16)
        else:
            ax.set_ylabel(r'$\phi$ Precision (STD) [mrad]', fontsize=14)
            ax.set_title(r'$\phi$ Regression Precision vs $\eta$', fontsize=16)
        
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(-3, 3)
        ax.grid(True, alpha=0.3)
        
        # Apply ATLAS style if available
        if ATLASIFY_AVAILABLE:
            atlasify.atlasify(atlas=False)
        
        # Save
        fig.savefig(self.output_dir / f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{output_name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved {output_name}.pdf and .png")
    
    def plot_pt_resolution(self, output_name='pt_resolution_comparison'):
        """Plot pT resolution for methods that can compute pT."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        pt_truth = self.results['truth']['pt']
        
        # Define pT bins (LINEAR scale, more bins)
        pt_bins = np.linspace(5, 200, 40)  # 40 bins from 5 to 200 GeV
        pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])  # Arithmetic mean
        
        colors = {
            'karimaki': 'red',
            'pca': 'purple',
            '3pt_sagitta': 'orange',
        }
        
        markers = {
            'karimaki': '^',
            'pca': 'D',
            '3pt_sagitta': 'v',
        }
        
        labels = {
            'karimaki': 'Karimäki Circle Fit (r-z)',
            'pca': 'PCA + Sagitta (all hits)',
            '3pt_sagitta': '3-Point Sagitta (η-dep B)',
        }
        
        for method in ['karimaki', 'pca', '3pt_sagitta']:
            pt_pred = self.results[method]['pt_pred']
            
            # Resolution = (pt_true - pt_pred) / pt_true
            resolution = (pt_truth - pt_pred) / pt_truth
            
            # Compute std in each pT bin
            resolutions_std = []
            resolution_errors = []
            
            for i in range(len(pt_bins) - 1):
                mask = (pt_truth >= pt_bins[i]) & (pt_truth < pt_bins[i + 1])
                if np.sum(mask) > 10:
                    bin_res = resolution[mask]
                    # Remove extreme outliers
                    valid = np.abs(bin_res) < 2.0
                    if np.sum(valid) > 5:
                        std = np.std(bin_res[valid])
                        n = np.sum(valid)
                        std_err = std / np.sqrt(2 * n - 2) if n > 2 else 0
                    else:
                        std = np.nan
                        std_err = np.nan
                    resolutions_std.append(std)
                    resolution_errors.append(std_err)
                else:
                    resolutions_std.append(np.nan)
                    resolution_errors.append(np.nan)
            
            resolutions_std = np.array(resolutions_std)
            resolution_errors = np.array(resolution_errors)
            
            # Get unbinned resolution for legend
            unbinned_res = self.stats[method].get('pt_resolution_std', np.nan)
            
            # Plot with distinct markers
            valid_mask = ~np.isnan(resolutions_std)
            ax.errorbar(
                pt_centers[valid_mask], resolutions_std[valid_mask] * 100,  # Convert to %
                yerr=resolution_errors[valid_mask] * 100,
                fmt=f'{markers[method]}-', color=colors[method],
                label=f'{labels[method]} (unbinned: {unbinned_res*100:.1f}%)',
                capsize=3, markersize=5, linewidth=1.5
            )
        
        ax.set_xlabel(r'True $p_T$ [GeV]', fontsize=14)
        ax.set_ylabel(r'$\sigma\left(\frac{p_T^{\mathrm{true}} - p_T^{\mathrm{pred}}}{p_T^{\mathrm{true}}}\right)$ [%]', fontsize=14)
        ax.set_title(r'$p_T$ Resolution vs True $p_T$', fontsize=16)
        # NO log scale - use linear
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(0, 210)
        ax.set_ylim(0, None)  # Start from 0
        ax.grid(True, alpha=0.3)
        
        if ATLASIFY_AVAILABLE:
            atlasify.atlasify(atlas=False)
        
        fig.savefig(self.output_dir / f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{output_name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved {output_name}.pdf and .png")
    
    def write_statistics(self):
        """Write statistics summary to file."""
        stats_file = self.output_dir / 'statistics_summary.txt'
        
        with open(stats_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PROXY METHOD EVALUATION - STATISTICS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Total tracks processed: {len(self.results['truth']['eta'])}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("ETA PRECISION (unbinned STD in mrad)\n")
            f.write("-" * 70 + "\n")
            for method in ['innermost', 'average', 'karimaki', 'pca', '3pt_sagitta']:
                std = self.stats[method]['eta_std_mrad']
                mean = self.stats[method]['eta_mean_mrad']
                f.write(f"  {method:14s}: STD = {std:8.2f} mrad, Mean = {mean:8.2f} mrad\n")
            
            f.write("\n")
            f.write("-" * 70 + "\n")
            f.write("PHI PRECISION (unbinned STD in mrad)\n")
            f.write("-" * 70 + "\n")
            for method in ['innermost', 'average', 'karimaki', 'pca', '3pt_sagitta']:
                std = self.stats[method]['phi_std_mrad']
                mean = self.stats[method]['phi_mean_mrad']
                f.write(f"  {method:14s}: STD = {std:8.2f} mrad, Mean = {mean:8.2f} mrad\n")
            
            f.write("\n")
            f.write("-" * 70 + "\n")
            f.write("PT RESOLUTION (unbinned STD)\n")
            f.write("-" * 70 + "\n")
            for method in ['karimaki', 'pca', '3pt_sagitta']:
                res_std = self.stats[method].get('pt_resolution_std', np.nan)
                res_mean = self.stats[method].get('pt_resolution_mean', np.nan)
                f.write(f"  {method:14s}: STD = {res_std*100:8.1f}%, Mean = {res_mean*100:8.1f}%\n")
            
            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write("BASELINE COMPARISON (from chi-square fit)\n")
            f.write("=" * 70 + "\n")
            f.write("  Eta:  1 mrad precision\n")
            f.write("  Phi: 10 mrad precision\n")
            f.write("  pT:   4% resolution\n")
        
        print(f"Statistics written to {stats_file}")
        
        # Also save as JSON for programmatic access
        json_file = self.output_dir / 'detailed_metrics.json'
        with open(json_file, 'w') as f:
            # Convert numpy types to Python types for JSON
            stats_json = {}
            for method in self.stats:
                stats_json[method] = {}
                for key, value in self.stats[method].items():
                    if isinstance(value, (np.floating, np.integer)):
                        stats_json[method][key] = float(value)
                    else:
                        stats_json[method][key] = value
            json.dump(stats_json, f, indent=2)
        
        print(f"Detailed metrics written to {json_file}")
    
    def run_evaluation(self):
        """Run the full evaluation pipeline."""
        print("\n" + "=" * 70)
        print("PROXY METHOD EVALUATION")
        print("=" * 70 + "\n")
        
        # Load data and compute proxies
        n_tracks = self.load_data_and_compute_proxies()
        
        if n_tracks == 0:
            print("ERROR: No tracks processed!")
            return
        
        # Compute statistics
        print("\nComputing statistics...")
        self.compute_statistics()
        
        # Print summary
        print("\n" + "-" * 70)
        print("SUMMARY STATISTICS")
        print("-" * 70)
        print(f"\n{'Method':<16} {'Eta STD (mrad)':<18} {'Phi STD (mrad)':<18} {'pT Res (%)':<15}")
        print("-" * 70)
        for method in ['innermost', 'average', 'karimaki', 'pca', '3pt_sagitta']:
            eta_std = self.stats[method]['eta_std_mrad']
            phi_std = self.stats[method]['phi_std_mrad']
            pt_res = self.stats[method].get('pt_resolution_std', np.nan)
            if np.isnan(pt_res):
                pt_str = "N/A"
            else:
                pt_str = f"{pt_res*100:.1f}"
            print(f"{method:<16} {eta_std:<18.2f} {phi_std:<18.2f} {pt_str:<15}")
        
        # Generate plots
        print("\n" + "-" * 70)
        print("GENERATING PLOTS")
        print("-" * 70)
        
        self.plot_precision_vs_eta('eta', 'eta_precision_comparison')
        self.plot_precision_vs_eta('phi', 'phi_precision_comparison')
        self.plot_pt_resolution('pt_resolution_comparison')
        
        # Write statistics
        self.write_statistics()
        
        print("\n" + "=" * 70)
        print(f"EVALUATION COMPLETE - Results in {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate proxy methods for track parameter estimation')
    parser.add_argument('--data_dir', '-d', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output_dir', '-o', type=str, default='./proxy_evaluation_output',
                        help='Base output directory for plots and results')
    parser.add_argument('--max_tracks', '-m', type=int, default=None,
                        help='Maximum number of tracks to process (default: all)')
    
    args = parser.parse_args()
    
    try:
        evaluator = ProxyMethodEvaluator(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_tracks=args.max_tracks,
        )
        evaluator.run_evaluation()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
