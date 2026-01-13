""" Balancing platform teleoperated evaluation script """

import os
import math
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
        self.x = 0.0
        self.y = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.height = 0.12
        self.ang_lim = 0.35
        self.stop = False
        
        self.Kp_x, self.Kd_x = 0.7, 0.1
        self.Kp_y, self.Kd_y = 0.7, 0.1
        
        self.x_meas = 0.0
        self.y_meas = 0.0
        self.x_meas_rot = 0.0
        self.y_meas_rot = 0.0
        self.e_x = 0.0
        self.e_y = 0.0
        self.e_vx = 0.0
        self.e_vy = 0.0
        self.pitch_cmd = 0.0
        self.roll_cmd = 0.0

    def clip_values(self):
        """ Clips the command values to stay within defined safe limits """
        self.x = np.clip(self.x, -0.09, 0.09)
        self.y = np.clip(self.y, -0.09, 0.09)
        self.pitch = np.clip(self.pitch, -self.ang_lim, self.ang_lim)
        self.roll = np.clip(self.roll, -self.ang_lim, self.ang_lim)
        self.Kp_x = np.clip(self.Kp_x, 0.0, 20.0)
        self.Kd_x = np.clip(self.Kd_x, 0.0, 5.0)
        
    def set_meas(self, x_meas, y_meas, x_meas_rot, y_meas_rot):
        self.x_meas = x_meas
        self.y_meas = y_meas
        self.x_meas_rot = x_meas_rot
        self.y_meas_rot = y_meas_rot
    
    def set_errors(self, ex, ey, evx, evy):
        self.e_x = ex
        self.e_y = ey
        self.e_vx = evx
        self.e_vy = evy
    
    def set_cmds(self, pitch_cmd, roll_cmd):
        self.pitch_cmd = pitch_cmd
        self.roll_cmd = roll_cmd

    def print_status(self, env):
        """ Prints the current teleoperation state to the console """
        os.system('clear')
        print(f"TARGET x: {self.x:+.3f} y: {self.y:+.3f} HEIGHT: {self.height:.3f}")
        print(f"PDx Kp:{self.Kp_x:.2f} Kd:{self.Kd_x:.2f} | PDy Kp:{self.Kp_y:.2f} Kd:{self.Kd_y:.2f}")
        print(f"MEAS x: {self.x_meas:+.3f} y: {self.y_meas:+.3f}")
        print(f"MEAS ROT x: {self.x_meas_rot:+.3f} y: {self.y_meas_rot:+.3f}")
        print(f"ERRORS: {self.e_x:.3f}, {self.e_y:.3f}, {self.e_vx:.3f}, {self.e_vy:.3f}")
        print(f"CMDs: PITCH: {self.pitch_cmd:.3f}, ROLL: {self.roll_cmd:.3f}")

def make_on_press(state: TeleopState):
    """
    Creates a key press handler that modifies the teleoperation state
    """
    def on_press(key):
        """ Handles key press events """
        try:
            if key == keyboard.Key.up:
                state.x += 0.01
            elif key == keyboard.Key.down:
                state.x -= 0.01
            elif key == keyboard.Key.left:
                state.y -= 0.01
            elif key == keyboard.Key.right:
                state.y += 0.01
            elif hasattr(key, "char") and key.char == 'q':
                state.Kp_x += 0.1
                state.Kp_y += 0.1
            elif hasattr(key, "char") and key.char == 'a':
                state.Kp_x -= 0.1
                state.Kp_y -= 0.1
            elif hasattr(key, "char") and key.char == 'w':
                state.Kd_x += 0.01
                state.Kd_y += 0.01
            elif hasattr(key, "char") and key.char == 's':
                state.Kd_x -= 0.01
                state.Kd_y -= 0.01
            elif hasattr(key, "char") and key.char == '8':
                state.stop = True

            state.clip_values()
            
        except AttributeError:
            pass

    return on_press

def on_release(key):
    """ Handles key release events. Stops the keyboard listener if ESC is pressed """
    return key != keyboard.Key.esc

def pd_tilts_from_xy(state: TeleopState, x_meas, y_meas, vx_meas, vy_meas):
    ex  = state.x - x_meas
    ey  = state.y - y_meas
    evx = 0.0 - vx_meas
    evy = 0.0 - vy_meas
    
    k = math.sqrt(2) / 2
    x_rot, y_rot = -k*x_meas - k*y_meas, k*x_meas - k*y_meas
    state.set_meas(x_meas, y_meas, x_rot, y_rot)
    
    ex, ey = -k*ex - k*ey, k*ex - k*ey
    evx, evy = -k*evx - k*evy, k*evx - k*evy
    state.set_errors(ex, ey, evx, evy)

    pitch_cmd = state.Kp_x * ex - state.Kd_x * evx
    roll_cmd  = - state.Kp_y * ey + state.Kd_y * evy

    pitch_cmd = float(np.clip(pitch_cmd, -state.ang_lim, state.ang_lim))
    roll_cmd  = float(np.clip(roll_cmd, -state.ang_lim, state.ang_lim))
    state.set_cmds(pitch_cmd, roll_cmd)

    return pitch_cmd, roll_cmd

def get_ball_state(env):
    try:
        x = float(env.ball_pos[0, 0].item())
        y = float(env.ball_pos[0, 1].item())
        vx = float(env.ball_vel[0, 0].item())
        vy = float(env.ball_vel[0, 1].item())
        return x, y, vx, vy
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

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
    env_cfg["ball_init_pos"] = [0.0, 0.0, 0.18]
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

    listener = keyboard.Listener(on_press=make_on_press(state), on_release=on_release, suppress=True)
    listener.start()

    with torch.no_grad():
        while not state.stop:
            state.print_status(env)
            x_meas, y_meas, vx_meas, vy_meas = get_ball_state(env)
            pitch_cmd, roll_cmd = pd_tilts_from_xy(state, x_meas, y_meas, vx_meas, vy_meas)
            actions = inverse_kinematics(roll_cmd, pitch_cmd, state.height)
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
