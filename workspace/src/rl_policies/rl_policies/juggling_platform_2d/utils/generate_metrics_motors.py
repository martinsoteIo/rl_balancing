"""
================================================================================
TFG PLOTTING UTILITY: MOTOR CONTROL COMPARISON (PID vs RL)
================================================================================
Description:
    This script generates high-quality, academic-grade PDF plots comparing the 
    motor control signals of the PID baseline and the RL Policy.

    REQUIREMENTS: Before running this script, you MUST execute the following data 
    collection scripts to generate the required .npy files:
    1. evaluate_motors_pid.py  -> Generates 'motor_data_pid_001.npy'
    2. evaluate_motors_rl.py   -> Generates 'motor_data_rl_001.npy'

Visual Features:
    * LaTeX-style fonts (Computer Modern) for academic consistency.
    * Pastel color scheme: Soft Red (PID) and Soft Blue (RL).
    * Vectorial output (PDF) for maximum resolution in the final document.
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,            
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,            
    "font.size": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,               
    "grid.alpha": 0.5,
    "grid.linestyle": ":",
    "lines.linewidth": 1.8,          
})

COLOR_PID = '#E57373'  # Soft Red for baseline
COLOR_RL = '#64B5F6'   # Soft Blue for proposed policy

def generate_comparative_plots():
    # 1. Path Definitions
    base_dir = "/home/admin/workspace/src/rl_policies/rl_policies/juggling_platform_2d/utils/"
    path_pid = os.path.join(base_dir, "motors/motor_data_pid_001.npy")
    path_rl = os.path.join(base_dir, "motors/motor_data_rl_001.npy")
    output_dir = os.path.join(base_dir, "motors/comparative_plots_final/")

    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Check for Pre-requisite Data Files
    if not os.path.exists(path_pid) or not os.path.exists(path_rl):
        print("\n" + "!"*60)
        print(" ERROR: DATA FILES NOT FOUND")
        print(" Please run the following scripts first:")
        print(" 1. python3 evaluate_motors_pid.py")
        print(" 2. python3 evaluate_motors_rl.py")
        print("!"*60 + "\n")
        return

    # 3. Load Collected Data
    data_pid = np.load(path_pid, allow_pickle=True).item()
    data_rl = np.load(path_rl, allow_pickle=True).item()
    print("--- Data loaded successfully. Starting plotting process... ---")

    dt = 0.02
    sorted_targets = sorted(data_pid.keys())

    # 4. PDF Generation Loop
    for i, x_target in enumerate(sorted_targets):
        if x_target not in data_rl:
            continue

        print(f"[{i+1}/{len(sorted_targets)}] Plotting Target x = {x_target:+.3f} m")

        h_pid = data_pid[x_target]
        h_rl = data_rl[x_target]

        t_pid = np.arange(len(h_pid)) * dt
        t_rl = np.arange(len(h_rl)) * dt

        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True, dpi=150)
        motor_names = ["Left Motor", "Right Motor"]
        
        for m_idx, ax in enumerate(axs):
            # PID Signal: Dashed line
            ax.plot(t_pid, h_pid[:, m_idx], color=COLOR_PID, linestyle='--', label='PID Baseline')
            # RL Signal: Solid line
            ax.plot(t_rl, h_rl[:, m_idx], color=COLOR_RL, linestyle='-', label='RL Policy')
            
            ax.set_ylabel(rf'$\theta_{{{motor_names[m_idx].split()[0].lower()}}}\ [\mathrm{{rad}}]$')
            ax.set_ylim(-1.6, 0.1) # Motor physical control range
            if m_idx == 0:
                ax.legend(loc='upper right', frameon=True)

        fig.suptitle(rf'Target x = {x_target:+.3f}', y=0.95)
        axs[1].set_xlabel(r'$\mathrm{Time}\ t\ [\mathrm{s}]$')
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.90)

        filename = f"comp_motors_x_{x_target:+.3f}.pdf"
        fig.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
        plt.close(fig)

    print(f"\n✅ Process complete. PDFs saved in: {output_dir}")

if __name__ == "__main__":
    generate_comparative_plots()