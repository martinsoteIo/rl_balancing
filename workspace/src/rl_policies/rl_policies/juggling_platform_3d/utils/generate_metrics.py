"""
================================================================================
3D PRECISION HEATMAP GENERATOR: SPATIAL STABILITY DIAGNOSTIC TOOL
================================================================================
Description:
    Adapted from the 2D version to visualize X-Y spatial precision.
    Uses the exact same font configuration and style as the working 2D script.
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

def create_pdf_heatmap_3d(exp_name="001_3d_grid", pid=False):
    """
    Generates a high-fidelity PDF heatmap of the 3D controller's precision profile.
    """
    if pid:
        base_dir = f"/home/admin/workspace/src/rl_policies/rl_policies/juggling_platform_3d/utils/npy/"
        file_path = os.path.join(base_dir, f"metric_PID_GRID_{exp_name}.npy")
    else:
        base_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
        file_path = os.path.join(base_dir, f"metrica_{exp_name}_3d.npy")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    data = np.load(file_path)
    total_points = len(data)
    N = int(np.sqrt(total_points))

    x_col = data[:, 0]
    y_col = data[:, 1]
    errors_col = data[:, 2]
    times_col = data[:, 3]

    valid_mask = ~np.isinf(times_col)
    valid_errors = errors_col[valid_mask]

    if len(valid_errors) == 0:
        print("Error: No valid data points (all falls).")
        return

    e_min = np.min(valid_errors)
    e_max = np.max(valid_errors)
    e_mean = np.mean(valid_errors)
    e_total = np.sum(valid_errors) 

    print(f"\n" + "="*30)
    print(f" 3D STATS: {exp_name} ")
    print(f"="*30)
    print(f"Grid Size:  {N}x{N}")
    print(f"Min Error:  {e_min:.6f}")
    print(f"Max Error:  {e_max:.6f}")
    print(f"Mean Error: {e_mean:.6f}")
    print(f"Stability:  {len(valid_errors)}/{total_points} points OK")
    print(f"="*30 + "\n")

    error_grid = errors_col.reshape(N, N).T
    time_grid = times_col.reshape(N, N).T
    
    extent = [x_col.min(), x_col.max(), y_col.min(), y_col.max()]
    fig, ax = plt.subplots(figsize=(6, 5)) 
    cmap = plt.cm.RdYlGn_r 
    

    masked_errors = np.ma.masked_where(np.isinf(time_grid), error_grid)
    ax.set_facecolor('black') 
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max)
    im = ax.imshow(masked_errors, cmap=cmap, aspect='equal', norm=norm,
                   extent=extent, origin='lower', interpolation='nearest')
    
    ax.set_xlabel(r'$\mathrm{Target}\ X\ \mathrm{[m]}$')
    ax.set_ylabel(r'$\mathrm{Target}\ Y\ \mathrm{[m]}$')

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(r'Error [m]', fontsize=10)


    output_path = os.path.join(base_dir, f"heatmap_3D_{exp_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Normalized 3D Heatmap saved to: {output_path}")

if __name__ == "__main__":
    create_pdf_heatmap_3d("Experimento008", pid=False)