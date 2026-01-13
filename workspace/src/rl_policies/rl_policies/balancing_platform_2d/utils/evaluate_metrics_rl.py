import os
import pickle
import torch
import numpy as np
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from rl_policies.balancing_platform_2d.envs.x_pos_env_genesis_mass_friction import XPos2DEnv

def evaluate_metrics_precision(exp_name="Experimento003", ckpt=9999):
    """
    Evaluates the policy precision using a normalized relative IAE metric.
    The error is integrated over time and normalized by both episode duration 
    and the magnitude of the target position.
    """
    # Initialize Genesis with reduced logging
    gs.init(logging_level="warning")

    # 1. Path setup and configuration loading
    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    # Grid search and Algorithm parameters
    d = 0.20          # Total platform width (meters)
    N = 100            # Number of sampling points across the width
    T_max = 8.0      # Total evaluation time per point (seconds) - t_start
    t_start = 2.0    # Time before starting error accumulation (seconds)
    dt = 0.02         # Simulation timestep
    step_start = int(t_start / dt)  # Corresponding step index
    max_steps = int(T_max / dt)
    epsilon = 1e-6    # Small constant to avoid division by zero

    # Instantiate the environment (single env for metric fidelity)
    env = XPos2DEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # Load the trained policy runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # Buffer for results: [x_target, relative_accumulated_error, flight_time]
    results_matrix = []

    print(f"--- Starting precision sweep for: {exp_name} ---")
    print(f"Parameters: d={d}, N={N}, T={T_max}s, Total Steps={max_steps}")

    for i in range(N):
        # Calculation of target X based on equidistant subdivision
        x_target = -d/2 + (d/N) * (i + 0.5)
        
        obs, _ = env.reset(is_train=False)
        target_tensor = torch.tensor([[x_target]], device=gs.device)
        
        accumulated_error = 0.0
        current_time = 0.0
        has_fallen = False

        for step in range(max_steps):
            with torch.no_grad():
                env.commands = target_tensor
                actions = policy(obs)
                obs, _, resets, _ = env.step(actions)

                # Real ball position at current timestep
                ball_x_real = env.ball_pos[0, 0].item()
                
                # Relative Error Calculation:
                # We divide by (|x_target| + epsilon) to normalize by command magnitude
                error_instant = abs(ball_x_real - x_target)
                normalization_factor = T_max + epsilon
                if step >= step_start:
                    accumulated_error += (error_instant * dt) / normalization_factor
                
                current_time += dt

                # Break and flag if ball falls (stability failure)
                if resets.any():
                    has_fallen = True
                    current_time = float('inf') 
                    break

        results_matrix.append([x_target, accumulated_error, current_time])
        
        status = "OK" if not has_fallen else "FAIL"
        print(f"i={i:02d} | Target: {x_target:+.4f} | Rel. Error: {accumulated_error:.6f} | Time: {current_time:.2f} | {status}")

    # 2. Saving the data in .npy format
    file_name = f"metrica_{exp_name}.npy"
    output_path = os.path.join(log_dir, file_name)
    
    final_data = np.array(results_matrix)
    np.save(output_path, final_data)
    
    print(f"\nMetrics successfully saved to: {output_path}")

if __name__ == "__main__":
    # You can specify your experiment and checkpoint here
    evaluate_metrics_precision(exp_name="Experimento002", ckpt=5700)