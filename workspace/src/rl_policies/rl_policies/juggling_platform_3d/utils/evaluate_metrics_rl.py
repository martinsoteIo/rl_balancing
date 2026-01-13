import os
import pickle
import torch
import numpy as np
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from rl_policies.juggling_platform_3d.envs.xy_pos_env_mass_friction import XPos3DEnv


def evaluate_metrics_precision_3d(exp_name="Experimento3D", ckpt=9999):
    """
    Evaluates the policy precision on a (x,y) grid using a steady-state IAE-like metric.
    Error is integrated over time starting from t_start and normalized by evaluation window length.
    """

    gs.init(logging_level="warning")

    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    d = 0.20          # Total platform width (meters) (assumed square workspace in x,y)
    Nx = 50           # Number of sampling points in x
    Ny = 50           # Number of sampling points in y
    T_max = 8.0       # Total evaluation time per point (seconds) - includes the transient, but we start accumulating at t_start
    t_start = 2.0     # Time before starting error accumulation (seconds)
    dt = 0.02         # Evaluation step time (should match env control dt; see note below)
    step_start = int(t_start / dt)
    max_steps = int(T_max / dt)
    epsilon = 1e-6

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
    policy = runner.get_inference_policy(device=gs.device)

    results_matrix = []

    print(f"--- Starting 3D precision grid sweep for: {exp_name} ---")
    print(f"Parameters: d={d}, Nx={Nx}, Ny={Ny}, T={T_max}s, Steps={max_steps}, t_start={t_start}s")

    xs = [-d/2 + (d/Nx) * (i + 0.5) for i in range(Nx)]
    ys = [-d/2 + (d/Ny) * (j + 0.5) for j in range(Ny)]

    for ix, x_target in enumerate(xs):
        for iy, y_target in enumerate(ys):

            obs, _ = env.reset(is_train=False)

            target_tensor = torch.tensor([[x_target, y_target]], device=gs.device)

            accumulated_error = 0.0
            current_time = 0.0
            has_fallen = False

            for step in range(max_steps):
                with torch.no_grad():
                    env.commands = target_tensor
                    actions = policy(obs)
                    obs, _, resets, _ = env.step(actions)

                ball_x = float(env.ball_pos[0, 0].item())
                ball_y = float(env.ball_pos[0, 1].item())

                ex = ball_x - x_target
                ey = ball_y - y_target
                error_instant = (ex * ex + ey * ey) ** 0.5

                normalization_factor = T_max + epsilon

                if step >= step_start:
                    accumulated_error += (error_instant * dt) / normalization_factor

                current_time += dt

                if resets.any():
                    has_fallen = True
                    current_time = float("inf")
                    break

            results_matrix.append([x_target, y_target, accumulated_error, current_time])

            status = "OK" if not has_fallen else "FAIL"
            print(
                f"ix={ix:02d}, iy={iy:02d} | "
                f"Target: ({x_target:+.4f}, {y_target:+.4f}) | "
                f"Err: {accumulated_error:.6f} | Time: {current_time:.2f} | {status}"
            )

    # Save
    file_name = f"metrica_{exp_name}_3d.npy"
    output_path = os.path.join(log_dir, file_name)

    final_data = np.array(results_matrix, dtype=np.float64)
    np.save(output_path, final_data)

    print(f"\n 3D metrics successfully saved to: {output_path}")


if __name__ == "__main__":
    evaluate_metrics_precision_3d(exp_name="Experimento008", ckpt=5700)
