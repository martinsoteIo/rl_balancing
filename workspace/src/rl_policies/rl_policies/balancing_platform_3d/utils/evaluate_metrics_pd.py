import os
import math
import pickle
import torch
import numpy as np
import genesis as gs
from rl_policies.balancing_platform_3d.envs.xy_pos_env_mujoco_pd import XPos3DEnv
from rl_policies.balancing_platform_3d.utils.ik import inverse_kinematics

def pd_controller_3d(x_target, y_target, x_meas, y_meas, vx_meas, vy_meas, Kp, Kd, ang_lim):
    ex = x_target - x_meas
    ey = y_target - y_meas
    evx = 0.0 - vx_meas
    evy = 0.0 - vy_meas
    
    k = math.sqrt(2) / 2
    
    ex_rot = -k * ex - k * ey
    ey_rot =  k * ex - k * ey
    evx_rot = -k * evx - k * evy
    evy_rot =  k * evx - k * evy

    pitch_cmd = Kp * ex_rot - Kd * evx_rot
    roll_cmd  = -Kp * ey_rot + Kd * evy_rot

    pitch_cmd = float(np.clip(pitch_cmd, -ang_lim, ang_lim))
    roll_cmd  = float(np.clip(roll_cmd, -ang_lim, ang_lim))
    
    return pitch_cmd, roll_cmd

def evaluate_metrics_pid_3d_grid(exp_name="001_3d_grid"):
    gs.init(logging_level="warning")

    base_path = "/home/admin/workspace/src/rl_policies"
    log_dir_output = f"{base_path}/rl_policies/balancing_platform_3d/utils/npy/"
    log_dir_cfgs = f"{base_path}/logs/xy_pos_3d_juggling" 
    
    if not os.path.exists(log_dir_output):
        os.makedirs(log_dir_output, exist_ok=True)

    try:
        with open(f"{log_dir_cfgs}/cfgs.pkl", "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró cfgs.pkl en {log_dir_cfgs}")
        return

    env_cfg["termination_if_pitch_greater_than"] = 10.0
    env_cfg["termination_if_roll_greater_than"] = 10.0
    env_cfg["termination_if_ball_x_greater_than"] = 10.0
    env_cfg["termination_if_ball_y_greater_than"] = 10.0
    env_cfg["termination_if_ball_falls"] = 0.05 
    env_cfg["ball_init_pos"] = [0.0, 0.0, 0.18]

    d = 0.20
    N = 50
    
    T_max, t_start = 8.0, 2.0 
    dt = 0.02                 
    step_start = int(t_start / dt)
    max_steps = int(T_max / dt)
    
    Kp, Kd = 0.8, 0.12 
    ang_lim = 0.35
    platform_height = 0.12 

    env = XPos3DEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False, 
        add_camera=False,
    )

    results_matrix = []
    total_points = N * N
    point_counter = 0

    print(f"--- Starting 3D PID GRID Sweep (Total points: {total_points}) ---")

    for i in range(N):
        x_target = -d/2 + (d/N) * (i + 0.5)
        
        for j in range(N):
            y_target = -d/2 + (d/N) * (j + 0.5)
            
            obs, _ = env.reset()
            accumulated_error = 0.0
            current_time = 0.0
            has_fallen = False

            for step in range(max_steps):
                ball_x = env.ball_pos[0, 0].item()
                ball_y = env.ball_pos[0, 1].item()
                ball_vx = env.ball_vel[0, 0].item()
                ball_vy = env.ball_vel[0, 1].item()

                pitch_cmd, roll_cmd = pd_controller_3d(
                    x_target, y_target, 
                    ball_x, ball_y, ball_vx, ball_vy, 
                    Kp, Kd, ang_lim
                )

                actions = inverse_kinematics(roll_cmd, pitch_cmd, platform_height)

                env.commands = torch.tensor([[x_target, y_target]], device=gs.device)
                
                obs, _, resets, _ = env.step(actions, is_train=False)

                error_instant = math.sqrt((ball_x - x_target)**2 + (ball_y - y_target)**2)
                
                if step >= step_start:
                    accumulated_error += (error_instant * dt) / (T_max + 1e-6)
                
                current_time += dt

                if resets.any():
                    has_fallen = True
                    current_time = float('inf')
                    break
            
            results_matrix.append([x_target, y_target, accumulated_error, current_time])
            point_counter += 1

            status = "OK" if not has_fallen else "FAIL"
            print(f"[{point_counter}/{total_points}] Tgt: ({x_target:+.2f}, {y_target:+.2f}) | Err: {accumulated_error:.4f} | {status}")

    output_path = os.path.join(log_dir_output, f"metric_PID_GRID_{exp_name}.npy")
    np.save(output_path, np.array(results_matrix))
    print(f"\n 3D Grid Metrics saved: {output_path}")

if __name__ == "__main__":
    evaluate_metrics_pid_3d_grid(exp_name="001_grid")