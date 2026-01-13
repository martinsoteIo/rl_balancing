import os
import math
import pickle
import torch
import numpy as np
import genesis as gs
from rl_policies.balancing_platform_2d.envs.pd.x_pos_env_mujoco_pd import XPos2DEnv

def angle_to_ctrl(theta, theta_min, theta_max, ctrl_min, ctrl_max, delta=0.0):
    t = (np.array(theta) - theta_min) / (theta_max - theta_min)
    u = ctrl_max - t * (ctrl_max - ctrl_min)
    return float(np.clip(delta + u, ctrl_min, ctrl_max))

def inverse_kinematics(angle, z):
    """ Calcula las acciones de los motores basadas en el ángulo deseado y altura z """
    l1, l2, d = 0.08, 0.28, 0.1979899
    LEFT_TH_MIN, LEFT_TH_MAX = np.deg2rad(90.0), np.deg2rad(180.0)
    RIGHT_TH_MIN, RIGHT_TH_MAX = -np.deg2rad(90.0), np.deg2rad(90.0)
    
    c, s = math.cos(angle), math.sin(angle)
    xl, yl = d/2 - (l2/2) * c, z - (l2/2) * s
    xr, yr = (l2/2) * c - d/2, z + (l2/2) * s
    
    rl, rr = math.sqrt(xl**2 + yl**2), math.sqrt(xr**2 + yr**2)
    
    # Alfa para IK
    alfa1 = math.acos(np.clip((rl**2 - 2*l1**2) / (2*l1**2), -1.0, 1.0))
    alfa2 = math.acos(np.clip((rr**2 - 2*l1**2) / (2*l1**2), -1.0, 1.0))
    
    theta1 = math.atan2(yl, xl) + math.atan2(l1 * math.sin(alfa1), l1 - l1 * math.cos(alfa1))
    theta2 = math.atan2(yr, xr) - math.atan2(l1 * math.sin(alfa2), l1 - l1 * math.cos(alfa2))
    
    ctrl_l = angle_to_ctrl(theta1, LEFT_TH_MIN, LEFT_TH_MAX, -1.5, 0.0)
    ctrl_r = angle_to_ctrl(theta2, RIGHT_TH_MIN, RIGHT_TH_MAX, -1.5, 0.0)
    
    return torch.tensor([[ctrl_l, ctrl_r]], device=gs.device, dtype=torch.float32)

def evaluate_metrics_pid(exp_name="001"):
    gs.init(logging_level="warning")

    # 1. Configuración de rutas
    # Carpeta donde se guardará el resultado
    log_dir = f"/home/admin/workspace/src/rl_policies/rl_policies/balancing_platform_2d/utils/npy/"
    # Carpeta de donde se cargan los hiperparámetros del entorno original
    log_dir_cfgs = f"/home/admin/workspace/src/rl_policies/logs/x_pos_2d_juggling_v3"
    
    # --- CORRECCIÓN: Crear la carpeta de salida si no existe ---
    if not os.path.exists(log_dir):
        print(f"📁 Creating directory: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir_cfgs}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(f)

    env_cfg["termination_if_pitch_greater_than"] = 0.5236
    env_cfg["termination_if_ball_x_greater_than"] = 0.12
    env_cfg["termination_if_ball_falls"] = 0.025
    env_cfg["ball_init_pos"] = [0.0, 0.0, 0.12]
    # Parámetros del Barrido
    d, N = 0.20, 100
    T_max, t_start = 8.0, 2.0
    dt = 0.02
    step_start = int(t_start / dt)
    max_steps = int(T_max / dt)
    epsilon = 1e-6

    # Parámetros Sintonizados
    Kp, Kd = 24.63, 0.815
    ang_lim = 0.40
    target_z = 0.03

    env = XPos2DEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    results_matrix = []
    print(f"--- Starting PID precision sweep for target {exp_name} ---")

    for i in range(N):
        x_target = -d/2 + (d/N) * (i + 0.5)
        obs, _ = env.reset() # Nota: Asegúrate que reset() no requiere is_train aquí si usas Mujoco Env
        
        accumulated_error = 0.0
        current_time = 0.0
        has_fallen = False

        for step in range(max_steps):
            ball_x = env.ball_pos[0, 0].item()
            ball_vx = env.ball_vel[0, 0].item()

            ex = x_target - ball_x
            ev = 0.0 - ball_vx
            angle_cmd = np.clip(Kp * ex + Kd * ev, -ang_lim, ang_lim)

            actions = inverse_kinematics(angle_cmd, target_z)

            env.commands = torch.tensor([[x_target]], device=gs.device)
            obs, _, resets, _ = env.step(actions, is_train=False)

            error_instant = abs(ball_x - x_target)
            if step >= step_start:
                accumulated_error += (error_instant * dt) / (T_max + epsilon)
            
            current_time += dt

            if resets.any():
                has_fallen = True
                current_time = float('inf') 
                break

        results_matrix.append([x_target, accumulated_error, current_time])
        
        if i % 10 == 0:
            status = "OK" if not has_fallen else "FAIL"
            print(f"i={i:02d} | Target: {x_target:+.3f} | Error: {accumulated_error:.6f} | {status}")

    # Guardado de datos
    output_path = os.path.join(log_dir, f"metric_PID_{exp_name}.npy")
    np.save(output_path, np.array(results_matrix))
    print(f"\nPID Metrics saved successfully to: {output_path}")

if __name__ == "__main__":
    evaluate_metrics_pid(exp_name="001")