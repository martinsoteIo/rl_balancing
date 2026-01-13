""" Balancing platform teleoperated evaluation script """

import os
import argparse
import pickle
import torch
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import genesis as gs
from pynput import keyboard
from rl_policies.balancing_platform_3d.envs.xy_pos_env import XPos3DEnv

class TeleopState:
    """ Stores the current teleoperation command state. """
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        # self.peak_height = 0.1
        self.stop = False

    def clip_values(self):
        """ Clips the command values to stay within defined safe limits """
        self.x = np.clip(self.x, -0.2, 0.2)
        self.y = np.clip(self.y, -0.2, 0.2)
        # self.peak_height = np.clip(self.peak_height, 0.0, 0.4)

    def print_status(self):
        """ Prints the current teleoperation state to the console """
        os.system('clear')
        print(f"POS_X: {self.x:.2f}, POS_Y: {self.y:.2f}")

def make_on_press(state: TeleopState):
    """
    Creates a key press handler that modifies the teleoperation state
    """
    def on_press(key):
        """ Handles key press events """
        try:
            if key == keyboard.Key.up:
                state.y += 0.01
            elif key == keyboard.Key.down:
                state.y -= 0.01
            elif key == keyboard.Key.left:
                state.x += 0.01
            elif key == keyboard.Key.right:
                state.x -= 0.01
            elif key.char == '8':
                state.stop = True

            state.clip_values()
            state.print_status()

        except AttributeError:
            pass

    return on_press

def on_release(key):
    """ Handles key release events. Stops the keyboard listener if ESC is pressed """
    return key != keyboard.Key.esc

def evaluate_teleop(env_class, exp_name: str = "xy_pos_3d_02112025_added_similar_reward",
                    ckpt: int = 5000, num_envs: int = 1, save_data: bool = False):
    """ Generic and environment-agnostic policy teleoperated evaluation method """

    gs.init(
        logger_verbose_time = False,
        logging_level="warning",
    )

    state = TeleopState()

    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env_cfg["termination_if_pitch_greater_than"] =  10.0
    env_cfg["termination_if_roll_greater_than"] =  10.0
    env_cfg["termination_if_ball_x_greater_than"] = 10.0
    env_cfg["termination_if_ball_y_greater_than"] = 10.0
    env_cfg["termination_if_ball_falls"] = 0.025
    env_cfg["ball_init_pos"] = [0.0, 0.0, 0.18]
    env_cfg["episode_length_s"] = 60.0
    reward_cfg["reward_scales"] = {}

    env = env_class(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False, # Set to True to visualize
        add_camera=True,
    )
    assert env.cam_0 is not None, "cam_0 no se ha creado: revisa add_camera en XPos3DEnv"
    cam = env.cam_0
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset(is_train = False)

    # start keyboard listener
    listener = keyboard.Listener(on_press=make_on_press(state), on_release=on_release)
    listener.start()

    images_buffer, commands_buffer = [], []
    with torch.no_grad():
        while not state.stop:
            env.commands = torch.tensor(
                [[state.x, state.y]],
                dtype=torch.float
            ).to(gs.device).repeat(num_envs, 1)

            actions = policy(obs)
            obs, *_ = env.step(actions, is_train=False)
            cam.render(rgb=False, depth=False, segmentation=False, normal=True)
            # optionally store camera images and command history
            # if env.cam_0 is not None and save_data:
            #     rgb, *_ = env.cam_0.render(
            #         rgb=True,
            #         depth=False,
            #         segmentation=False,
            #     )
            #     images_buffer.append(rgb)
            #     commands_buffer.append([state.lin_x, state.lin_y, state.ang_z, state.base_height])

    if save_data:
        # save captured image frames and commands to disk
        pickle.dump(np.array(images_buffer), open(os.path.join(log_dir, "/imgs/image_buffer.pkl"), "wb"))
        pickle.dump(np.array(commands_buffer), open(os.path.join(log_dir, "/cmds/commands_buffer.pkl"), "wb"))

def main():
    """
    Main evaluation script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="Experimento007")
    parser.add_argument("--ckpt", type=int, default=9999)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--save-data", type=bool, default=False)
    args = parser.parse_args()

    evaluate_teleop(
        env_class=XPos3DEnv,
        exp_name=args.exp_name,
        num_envs=args.num_envs,
        ckpt=args.ckpt,
        save_data=args.save_data,
    )

if __name__ == "__main__":
    main()
