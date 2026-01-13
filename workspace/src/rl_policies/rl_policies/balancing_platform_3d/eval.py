""" Balancing platform evaluation script """

import os
import argparse
import pickle
import torch
import numpy as np
import cv2
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from rl_policies.balancing_platform_3d.envs.xy_pos_env import XPos3DEnv


def evaluate(env_class,
             exp_name: str = "x_pos_3d_juggling",
             ckpt: int = 100,
             cmds: list = None):
    """ 
    Generic and environment-agnostic policy evaluation method.
    """
    if cmds is None:
        cmds = [0.1, 0.1]

    gs.init()
    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    reward_cfg["reward_scales"] = {}
    env_cfg["termination_if_ball_x_greater_than"] = 0.15
    env_cfg["termination_if_ball_falls"] = 0.0
    env_cfg["ball_init_pos"] = [0.0, 0.0, 0.18]

    env = env_class(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        add_camera=True,
    )

    assert env.cam_0 is not None, "cam_0 no se ha creado: revisa add_camera en XPos3DEnv"
    cam = env.cam_0
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset(is_train = False)
    
    video_path = os.path.join(log_dir, "eval.mp4")
    cam.start_recording()
    step = 0

    try:
        with torch.no_grad():
            while True:
                env.commands = torch.tensor(cmds, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                actions = policy(obs)
                obs, _, _, _ = env.step(actions, is_train = False)
                cam.render(rgb=False, depth=False, segmentation=False, normal=True)
                step += 1
    except KeyboardInterrupt:
        print("Ctrl+C detectado, parando evaluación...")
    cam.stop_recording(save_to_filename=video_path, fps=15)
    print(f"\n✓ Video guardado exitosamente: {video_path}")
        
def main():
    """
    Main evaluation script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="Experimento007")
    parser.add_argument("--ckpt", type=int, default=9999)
    args = parser.parse_args()

    cmds = [0.00, 0.00]

    evaluate(
        env_class=XPos3DEnv,
        exp_name=args.exp_name,
        ckpt=args.ckpt,
        cmds=cmds
    )

if __name__ == "__main__":
    main()
