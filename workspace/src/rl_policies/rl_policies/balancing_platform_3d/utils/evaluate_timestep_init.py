import os
import pickle
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from rl_policies.balancing_platform_3d.envs.xy_pos_env_mass_friction import XPos3DEnv

plt.rcParams.update({
    "text.usetex": False,            
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 11,
    "svg.fonttype": "none", 
})

def run_full_sweep(exp_name="Experimento006", ckpt=9999):
    gs.init(logging_level="warning")
    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env = XPos3DEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    
    runner.alg.actor_critic.eval()
    policy = runner.get_inference_policy(device=gs.device)
    
    targets_val = np.arange(-0.07, 0.07, 0.02)
    T_max, dt = 8.0, 0.02
    max_steps = int(T_max / dt)
    t_eval_start = 2.0
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(targets_val)))

    print(f"--- Starting Full XY Sweep: {len(targets_val)} trajectories ---")

    for idx, val in enumerate(targets_val):
        target_x, target_y = val, val
        time_series, err_sq_series = [], []
        
        obs, _ = env.reset(is_train=False)
        target_tensor = torch.tensor([[target_x, target_y]], device=gs.device, dtype=torch.float)
        print(f"Running trajectory for target position: x={target_x:+.3f}, y={target_y:+.3f}")
        for step in range(max_steps):
            with torch.no_grad():
                env.commands = target_tensor
                actions = policy(obs.float())
                obs, _, resets, _ = env.step(actions, is_train=False)
                
                t = step * dt
                bx = env.ball_pos[0, 0].item()
                by = env.ball_pos[0, 1].item()

                err_sq = (bx - target_x)**2 + (by - target_y)**2

                time_series.append(t)
                err_sq_series.append(err_sq)

                
                if resets.any(): break

        ax.plot(time_series, err_sq_series, color=colors[idx], lw=1.5, label=f'T:{val:+.2f}')

    ax.axvline(x=t_eval_start, color='gray', linestyle=':', lw=2, alpha=0.5)
    
    leg1 = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=r'$x_{target}, y_{target}$', fontsize=9)
    ax.add_artist(leg1)

    ax.set_ylabel(r'$(x-x^*)^2 + (y-y^*)^2\ \mathrm{[m^2]}$')
    ax.set_ylim([0.0, 0.01])
    ax.set_xlim([0, T_max])
    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(log_dir, f"full_sweep_combined_{exp_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"END. PDF saved in: {output_path}")

if __name__ == "__main__":
    run_full_sweep(exp_name="Experimento007", ckpt=9500)