""" Balancing platform teleoperated evaluation script """

import os
import argparse
import pickle
import torch
import numpy as np
import genesis as gs
from pynput import keyboard
from rl_policies.juggling_platform_3d.envs.xy_pos_env_mujoco_pd import XPos3DEnv
from rl_policies.juggling_platform_3d.utils.ik import inverse_kinematics


class TeleopState:
    """ Stores the current teleoperation command state. """
    def __init__(self):
        self.pitch = 0.0
        self.roll = 0.0
        self.height = 0.12
        self.stop = False

    def clip_values(self):
        """ Clips the command values to stay within defined safe limits """
        self.pitch = np.clip(self.pitch, -0.35, 0.35)
        self.roll = np.clip(self.roll, -0.35, 0.35)

    def print_status(self):
        """ Prints the current teleoperation state to the console """
        os.system('clear')
        print(f"ROLL: {self.roll:.2f}, PITCH: {self.pitch:.2f}, HEIGHT: {self.height:.2f}")

def make_on_press(state: TeleopState):
    """
    Creates a key press handler that modifies the teleoperation state
    """
    def on_press(key):
        """ Handles key press events """
        try:
            if key.char == 'q':
                state.pitch += 0.01
            elif key.char == 'a':
                state.pitch -= 0.01
            elif key.char == 'w':
                state.roll += 0.01
            elif key.char == 's':
                state.roll -= 0.01
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

def evaluate_teleop(env_class, exp_name: str = "x_pos_3d_juggling_v3",
                    ckpt: int = 100, num_envs: int = 1, save_data: bool = False):
    """ Generic and environment-agnostic policy teleoperated evaluation method """

    gs.init(
        logger_verbose_time = False,
        logging_level="warning",
    )

    state = TeleopState()

    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(f)
    
    env_cfg["default_joint_angles"] = {
        "first_motor_joint": 0.0,
        "second_motor_joint": 0.0,
        "third_motor_joint": 0.0,
        "first_arm_joint": 0.0,
        "second_arm_joint": 0.0,
        "third_arm_joint": 0.0,
    }
    env_cfg["default_motor_angles"] = {
        "first_motor_joint": 0.0,
        "second_motor_joint": 0.0,
        "third_motor_joint": 0.0,
    }
    env_cfg["termination_if_pitch_greater_than"] =  10.0
    env_cfg["termination_if_roll_greater_than"] =  10.0
    env_cfg["termination_if_ball_x_greater_than"] = 10.0
    env_cfg["termination_if_ball_y_greater_than"] = 10.0
    env_cfg["termination_if_ball_falls"] = 0.025
    env_cfg["ball_init_pos"] = [0.0, 0.0, 0.4]
    env_cfg["episode_length_s"] = 60.0
    reward_cfg["reward_scales"] = {}

    env = env_class(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        add_camera=True,
    )

    # start keyboard listener
    listener = keyboard.Listener(on_press=make_on_press(state), on_release=on_release, suppress=True,)
    listener.start()

    with torch.no_grad():
        while not state.stop:
            actions = inverse_kinematics(state.roll, state.pitch, state.height)
            env.step(actions, is_train=False)

def main():
    """
    Main evaluation script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="xy_pos_3d_juggling")
    parser.add_argument("--ckpt", type=int, default=100)
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
