""" Balancing Platform teleoperated evaluation script """

import os
import math
import argparse
import pickle
import torch
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import genesis as gs
from pynput import keyboard
from rl_policies.balancing_platform_2d.envs.pd.x_pos_env_mujoco import XPos2DEnv

class TeleopState:
    """ Stores the current teleoperation command state. """
    def __init__(self):
        self.x = 0.0
        self.angle = 0.0
        self.height = 0.03
        self.stop = False

    def clip_values(self):
        """ Clips the command values to stay within defined safe limits """
        self.x = np.clip(self.x, -0.10, 0.10)
        self.angle = np.clip(self.angle, -0.4, 0.4)
        self.height = np.clip(self.height, 0.02, 0.05)

    def print_status(self):
        """ Prints the current teleoperation state to the console """
        os.system('clear')
        print(f"ANGLE: {self.angle:.2f}, HEIGHT: {self.height:.2f}")

def make_on_press(state: TeleopState):
    """
    Creates a key press handler that modifies the teleoperation state
    """
    def on_press(key):
        """ Handles key press events """
        try:
            if key == keyboard.Key.up:
                state.height += 0.01
            elif key == keyboard.Key.down:
                state.height -= 0.01
            elif key == keyboard.Key.left:
                state.angle -= 0.01
            elif key == keyboard.Key.right:
                state.angle += 0.01
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
import time

def a2ctrl_left(alpha_rad, delta=0.0):
    return float(np.clip(delta - 1.5*np.array(alpha_rad)/np.pi, -1.5, 0.0))

def a2ctrl_right(alpha_rad, delta=0.0):
    return float(np.clip(delta - 1.5*(np.pi - np.array(alpha_rad))/np.pi, -1.5, 0.0))

def inverse_kinematics(angle, z):
    """ Simple 2D inverse kinematics """
    d = 0.1979899
    l1 = 0.08
    l2 = 0.28
    
    def wrap_to_pi(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    def unwrap_near(a, prev):
        a = wrap_to_pi(a)
        if prev is None:
            return a
        delta = a - wrap_to_pi(prev)
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        return prev + delta
    
    c, s = math.cos(angle), math.sin(angle)
    xl =  d/2 - (l2/2) * c
    yl =  z - (l2/2) * s
    xr =  (l2/2) * c - d/2
    yr =  z + (l2/2) * s
    
    rl = math.sqrt(xl**2 + yl**2)
    rr = math.sqrt(xr**2 + yr**2)
    # print(f"IK: angle: {angle:.2f}, height: {z:.2f} => rl: {rl:.2f}, rr: {rr:.2f}")
    alfa1 = math.acos((rl**2 - 2*l1**2) / (2*l1**2))
    alfa2 = math.acos((rr**2 - 2*l1**2) / (2*l1**2))
    
    beta1 = math.atan2(yl, xl)
    beta2 = math.atan2(yr, xr)
    
    gamma1 = math.atan2(l1 * math.sin(alfa1), l1 - l1 * math.cos(alfa1))
    gamma2 = math.atan2(l1 * math.sin(alfa2), l1 - l1 * math.cos(alfa2))
    
    theta1 = beta1 + gamma1
    theta2 = beta2 - gamma2
    print(f"IK: angle: {angle:.2f}, height: {z:.2f} => theta1: {theta1:.2f}, theta2: {theta2:.2f}")
    print(f"IK 2: angle: {angle:.2f}, height: {z:.2f} => theta1: {a2ctrl_left(theta1):.2f}, theta2: {a2ctrl_right(theta2):.2f}")
    actions = torch.tensor([[theta1,theta2]], device=gs.device, dtype=torch.float32)
    return actions

def evaluate_teleop(env_class, exp_name: str = "x_pos_2d_juggling_v3",
                    ckpt: int = 100, num_envs: int = 1, save_data: bool = False):
    """ Generic and environment-agnostic policy teleoperated evaluation method """

    gs.init(
        logger_verbose_time = False,
        logging_level="warning",
    )

    state = TeleopState()

    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env_cfg["termination_if_pitch_greater_than"] =  0.5236
    env_cfg["termination_if_ball_x_greater_than"] = 0.12
    env_cfg["termination_if_ball_falls"] = 0.025
    env_cfg["ball_init_pos"] = [0.0, 0.0, 0.14]
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

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    # policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    # start keyboard listener
    listener = keyboard.Listener(on_press=make_on_press(state), on_release=on_release, suppress=True,)
    listener.start()

    images_buffer, commands_buffer = [], []
    with torch.no_grad():
        while not state.stop:
            env.commands = torch.tensor(
                [[state.x]],
                dtype=torch.float
            ).to(gs.device).repeat(num_envs, 1)

            actions = inverse_kinematics(state.angle, state.height)
            # actions = policy(obs)
            obs, *_ = env.step(actions, is_train=False)

    if save_data:
        # save captured image frames and commands to disk
        pickle.dump(np.array(images_buffer), open(os.path.join(log_dir, "/imgs/image_buffer.pkl"), "wb"))
        pickle.dump(np.array(commands_buffer), open(os.path.join(log_dir, "/cmds/commands_buffer.pkl"), "wb"))

def main():
    """
    Main evaluation script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="x_pos_2d_juggling_v3")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--save-data", type=bool, default=False)
    args = parser.parse_args()

    evaluate_teleop(
        env_class=XPos2DEnv,
        exp_name=args.exp_name,
        num_envs=args.num_envs,
        ckpt=args.ckpt,
        save_data=args.save_data,
    )

if __name__ == "__main__":
    main()
