"""
================================================================================
PRECISION HEATMAP GENERATOR: SPATIAL STABILITY DIAGNOSTIC TOOL
================================================================================
Description:
    This module serves as a post-processing diagnostic and visualization tool 
    designed to evaluate the spatial precision of balancing controllers (both 
    Reinforcement Learning policies and classical PID) within a 2D juggling 
    platform environment.

    The software transforms raw simulation data (.npy format) into academic-grade 
    visual heatmaps by analyzing the Steady-State Integrated Absolute Error 
    (SS-IAE) across the platform's workspace.

Key Features:
    * Dynamic Normalization: Scales the color gradient between the minimum and 
      maximum observed errors to highlight precision variations.
    * Stability Failure Detection: Automatically identifies and masks "stability 
      failures" (ball drops or safety limit resets) as black segments.
    * Global Performance Scoring (E_total): Calculates the total accumulated 
      error across the entire target sweep for quantitative benchmarking.
    * LaTeX Integration: Utilizes Computer Modern fonts and mathematical 
      notation for professional PDF output.

Author: Martín Sotelo Aguirre
Version: 1.0
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

plt.rcParams.update({
    "text.usetex": False,            
    "mathtext.fontset": "cm",        
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "svg.fonttype": "none", 
})

def create_pdf_heatmap(exp_name="Experimento003", pid=False):
    """
    Generates a high-fidelity PDF heatmap of the controller's precision profile.

    Arguments:
        exp_name (str): Identifier for the RL experiment or PID baseline.
        pid (bool): Toggle between RL logs (False) and PID metrics path (True).

    Metric Definition:
        SS-IAE = (1 / (T - t_start)) * Integral(|x_target - x_real|) dt
        evaluated from t_start = 2.0s to T = 8.0s.
    """
    if pid:
        log_dir = f"/home/admin/workspace/src/rl_policies/rl_policies/juggling_platform_2d/utils/npy/"
        file_path = os.path.join(log_dir, f"metric_PID_{exp_name}.npy")
    else:
        log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
        file_path = os.path.join(log_dir, f"metrica_{exp_name}.npy")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load data: [Target_X, Error, Time]
    data = np.load(file_path)
    total_points = len(data)
    
    x_targets = data[:, 0]
    errors = data[:, 1]
    times = data[:, 2]

    # --- DINAMIC NORMALIZATION LOGIC ---
    valid_mask = ~np.isinf(times)
    valid_errors = errors[valid_mask]

    if len(valid_errors) == 0:
        print("Error: No valid data points (all falls).")
        return

    e_min = np.min(valid_errors)
    e_max = np.max(valid_errors)
    e_mean = np.mean(valid_errors)
    e_total = np.sum(valid_errors) 

    print(f"\n" + "="*30)
    print(f" STATS: {exp_name} ")
    print(f"="*30)
    print(f"Min Error:  {e_min:.6f}")
    print(f"Max Error:  {e_max:.6f}")
    print(f"Mean Error: {e_mean:.6f}")
    print(f"TOTAL ACCUMULATED ERROR (E_total): {e_total:.6f}")
    print(f"Stability:  {len(valid_errors)}/{total_points} points OK")
    print(f"="*30 + "\n")

    heatmap_row = errors.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(10, 3.5)) 

    cmap = plt.cm.RdYlGn_r 
    masked_errors = np.ma.masked_where(np.isinf(times).reshape(1, -1), heatmap_row)
    ax.set_facecolor('black') 

    norm = mcolors.Normalize(vmin=e_min, vmax=e_max)

    im = ax.imshow(masked_errors, cmap=cmap, aspect='auto', norm=norm,
                   extent=[x_targets.min(), x_targets.max(), -0.01, 0.01])

    clean_exp = exp_name.replace('_', ' ')
    ax.set_xlabel(r'$\mathrm{Platform\ Position}\ x\ \mathrm{[m]}$')
    ax.set_yticks([]) 
    cbar = fig.colorbar(im, orientation='horizontal', pad=0.45)
    cbar.set_label(r'Error [m]', fontsize=10)
    output_path = os.path.join(log_dir, f"heatmap_{exp_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    
    print(f"Normalized Heatmap saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    create_pdf_heatmap("001", True)