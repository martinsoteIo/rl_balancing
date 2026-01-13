""" Balancing Platform RL Environment """

import math
import torch
import genesis as gs
import mujoco
from mujoco import viewer
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
import numpy as np
import threading
import time

def gs_rand_float(lower, upper, shape, device):
    """ 
    Generate a tensor of random float uniformly sampled 
    """
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def gs_rand_gaussian(mean, lower, upper, n_std, shape, device):
    """
    Generate a tensor of random float following a Gaussian distribution
    """
    mean_tensor = mean.expand(shape).to(device)
    std_tensor = torch.full(shape, (upper - lower)/ 4.0 * n_std, device=device)
    return torch.clamp(torch.normal(mean_tensor, std_tensor), lower, upper)

def launch_viewer(model, data, data_lock):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with data_lock:
                viewer.sync()         

class XPos3DEnv:
    """
    Simulated environment for the Balancing platform using the Genesis simulator.
    """
        
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, add_camera=False):
        """ 
        Initialize the XPos3DEnv simulation environment.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.viewer_data_lock = threading.Lock()
        
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_dof = env_cfg["num_dof"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
    
        self.simulate_action_latency = False  # there is a 1 step latency on real robot
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        
        model_path = "/home/admin/workspace/src/descriptions/balancing_platform_3d_description/urdf/robot_bigger_platform.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetData(self.model, self.data)

        self.platform_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        # names to indices
        self.motor_names = ["first_motor","second_motor","third_motor"]
        self.motors_dof_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.motor_names]
        self.joint_names = [
            "first_motor_joint", "second_motor_joint", "third_motor_joint",
            "first_arm_joint", "second_arm_joint", "third_arm_joint"
        ]

        self.joints_dof_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.joint_ids = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                  for n in self.joint_names}
        self.qpos_adr = {n: self.model.jnt_qposadr[self.joint_ids[n]] for n in self.joint_names}
        self.qvel_adr = {n: self.model.jnt_dofadr[self.joint_ids[n]] for n in self.joint_names}

        self.jnt_qpos_idx = np.array([self.model.jnt_qposadr[j] for j in self.joints_dof_idx], dtype=np.int32)
        self.jnt_qvel_idx = np.array([self.model.jnt_dofadr[j] for j in self.joints_dof_idx], dtype=np.int32)
        
        self.ball_joint_names = ["ball_joint"]
        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qpos_adr = self.model.jnt_qposadr[self.ball_joint_id]
        self.ball_qvel_adr = self.model.jnt_dofadr[self.ball_joint_id]

        # # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)

        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device)
        self.commands_scale = torch.tensor([self.obs_scales["ball_pos_x"], self.obs_scales["ball_pos_y"]], device=self.device)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        self.platform_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.platform_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.ball_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ball_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_ball_z = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_vel = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.ball_init_pos = torch.tensor(self.env_cfg["ball_init_pos"], device=self.device)
        self.ball_init_qpos = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        self.ball_init_qvel = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float)
        self.ball_init_qpos[:, 0] = self.ball_init_pos[0]
        self.ball_init_qpos[:, 1] = self.ball_init_pos[1]
        self.ball_init_qpos[:, 2] = self.ball_init_pos[2]
        self.ball_init_qpos[:, 3] = 1.0
        
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=torch.float,
        )
        
        self.default_motors_pos = torch.tensor(
            [self.env_cfg["default_motor_angles"][name] for name in self.env_cfg["motor_names"]],
            device=self.device,
            dtype=torch.float,
        )
        
        if show_viewer:
            self.viewer_thread = threading.Thread(target=launch_viewer, args=(self.model, self.data, self.viewer_data_lock), daemon=True)
            self.viewer_thread.start()
            
        self.extras = {}
        self.extras["observations"] = {}
        
    def _sample_commands(self, envs_idx):
        """
        Resample velocity command targets for the given environment indices.
        """
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["target_x"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["target_y"], (len(envs_idx),), self.device)

    def step(self, actions, is_train=True):
        """
        Apply the given actions, advance the simulation by one step, and return updated observations and rewards.
        """

        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_motors_pos
        clipped_dof_pos = torch.clip(target_dof_pos, -1.5, 0.0)
        clipped_dof_pos_np = clipped_dof_pos.cpu().numpy()
        
        for i in range(self.num_envs):
            for j, dof_id in enumerate(self.motors_dof_idx):
                self.data.ctrl[dof_id] = clipped_dof_pos_np[i, j]
                
        with self.viewer_data_lock:
            mujoco.mj_step(self.model, self.data)

        # update observation buffers
        self.episode_length_buf += 1
        
        # Platform pose (world)
        quat_p = self.data.xquat[self.platform_body_id].astype(np.float32)        # WXYZ
        R_plat  = self.data.xmat[self.platform_body_id].reshape(3, 3).astype(np.float32)
        self.platform_quat.copy_(torch.from_numpy(quat_p).to(self.device))
        self.platform_euler = quat_to_xyz(self.platform_quat)

        omega_body_p = self.data.cvel[self.platform_body_id, 0:3].astype(np.float32)
        omega_world_p = R_plat @ omega_body_p
        self.platform_ang_vel.copy_(torch.from_numpy(omega_world_p).to(self.device))

        # DOFs (si aplica)
        qpos_np = self.data.qpos[self.jnt_qpos_idx].astype(np.float32)
        qvel_np = self.data.qvel[self.jnt_qvel_idx].astype(np.float32)
        self.dof_pos.copy_(torch.from_numpy(qpos_np).to(self.device))
        self.dof_vel.copy_(torch.from_numpy(qvel_np).to(self.device))
        # Ball pose (world)
        pos_b   = self.data.xpos[self.ball_body_id].astype(np.float32)
        quat_b  = self.data.xquat[self.ball_body_id].astype(np.float32)           # WXYZ
        R_ball  = self.data.xmat[self.ball_body_id].reshape(3, 3).astype(np.float32)
        self.ball_pos.copy_(torch.from_numpy(pos_b).to(self.device))
        self.ball_quat.copy_(torch.from_numpy(quat_b).to(self.device))

        self.ball_euler_world = quat_to_xyz(self.ball_quat)

        # Ball linear velocity in world
        v_body_b  = self.data.cvel[self.ball_body_id, 3:6].astype(np.float32)
        v_world_b = R_ball @ v_body_b
        self.ball_vel.copy_(torch.from_numpy(v_world_b).to(self.device))

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.platform_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.abs(self.platform_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.ball_pos[:, 0]) > self.env_cfg["termination_if_ball_x_greater_than"]
        self.reset_buf |= torch.abs(self.ball_pos[:, 1]) > self.env_cfg["termination_if_ball_y_greater_than"]
        self.reset_buf |= self.ball_pos[:, 2] < self.env_cfg["termination_if_ball_falls"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf.copy_(torch.cat(
            [
                (self.platform_euler[:, 1] * self.obs_scales["platform_pitch"]).unsqueeze(-1), # 1
                (self.platform_euler[:, 0] * self.obs_scales["platform_roll"]).unsqueeze(-1), # 1
                (self.platform_ang_vel[:, 1] * self.obs_scales["platform_pitch_vel"]).unsqueeze(-1), # 1
                (self.platform_ang_vel[:, 0] * self.obs_scales["platform_roll_vel"]).unsqueeze(-1), # 1
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 6
                self.dof_vel * self.obs_scales["dof_vel"],  # 6
                (self.ball_pos[:, 0] * self.obs_scales["ball_pos_x"]).unsqueeze(-1), # 1
                (self.ball_pos[:, 1] * self.obs_scales["ball_pos_y"]).unsqueeze(-1), # 1
                (self.ball_pos[:, 2] * self.obs_scales["ball_pos_z"]).unsqueeze(-1), # 1
                (self.ball_vel[:, 0] * self.obs_scales["ball_vel_x"]).unsqueeze(-1), # 1
                (self.ball_vel[:, 1] * self.obs_scales["ball_vel_y"]).unsqueeze(-1), # 1
                (self.ball_vel[:, 2] * self.obs_scales["ball_vel_z"]).unsqueeze(-1), # 1
                self.commands * self.commands_scale, # 2
                self.actions, # 3
            ],
            axis=-1,
        ))

        self.last_actions.copy_(self.actions)

        bad_envs = torch.isnan(self.obs_buf).any(dim=1).nonzero(as_tuple=False).flatten()
        if len(bad_envs) > 0:
            print(f"\033[93mResetting {len(bad_envs)} environments due to NaN in obs_buf.\033[0m")
            self.reset_idx(bad_envs)
            self.obs_buf[bad_envs] = 0.0

        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        """
        Return the current observation buffer and auxiliary info.
        """
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        """
        Return privileged observations (if any). This environment does not use them.
        """
        return None

    def reset_idx(self, envs_idx):
        """
        Reset the simulation state for the specified environment indices.
        """
        if len(envs_idx) == 0:
            return
        
        # self.dof_pos[envs_idx] = self.default_dof_pos
        # self.dof_vel[envs_idx] = 0.0

        for i, name in enumerate(self.joint_names):
            qadr = self.qpos_adr[name]
            dadr = self.qvel_adr[name]
            self.data.qpos[qadr] = self.dof_pos[envs_idx][0, i].item()
            self.data.qvel[dadr] = 0.0

        for j, act_id in enumerate(self.motors_dof_idx):
            self.data.ctrl[act_id] = self.default_motors_pos[j].item()

        self.ball_pos[envs_idx] = self.ball_init_pos
        self.ball_vel[envs_idx] = 0.0
        
        ball_init_qpos_np = self.ball_init_qpos[0].cpu().numpy()
        ball_init_qvel_np = self.ball_init_qvel[0].cpu().numpy()
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 7] = ball_init_qpos_np
        self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6] = ball_init_qvel_np
    
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # self._sample_commands(envs_idx)
        with self.viewer_data_lock:
            mujoco.mj_forward(self.model, self.data)
        
    def reset(self):
        """
        Reset all environments.
        """
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None
    
    def _alive_mask(self):
        x_half = self.reward_cfg.get("platform_x_half", 0.11)
        y_half = self.reward_cfg.get("platform_y_half", 0.11)
        z_min  = self.reward_cfg.get("z_min_alive",     0.026)

        on_x = torch.abs(self.ball_pos[:, 0]) <= x_half
        on_y = torch.abs(self.ball_pos[:, 1]) <= y_half
        on_z = self.ball_pos[:, 2] >= z_min
        return (on_x & on_y & on_z).float()

    def _reward_tracking_xy(self):
        sx = self.reward_cfg.get("tracking_sigma_x", self.reward_cfg.get("tracking_sigma", 0.06))
        sy = self.reward_cfg.get("tracking_sigma_y", self.reward_cfg.get("tracking_sigma", 0.06))

        ex = self.commands[:, 0] - self.ball_pos[:, 0]
        ey = self.commands[:, 1] - self.ball_pos[:, 1]

        r = torch.exp(- (ex * ex) / (sx * sx) - (ey * ey) / (sy * sy))
        return r * self._alive_mask()

    def _reward_stabilize_z_velocity(self):
        s  = self.reward_cfg["stability_sigma"]      
        v0 = 0.04                                 
        vz = torch.abs(self.ball_vel[:, 2])
        eff = torch.clamp((vz - v0) / s, min=0.0, max=1.0)
        return - (eff * eff) 

    def _success_mask(self):
        x_tol  = self.reward_cfg.get("success_x_tol",  0.003)
        y_tol  = self.reward_cfg.get("success_y_tol",  0.003)
        vz_tol = self.reward_cfg.get("success_vz_tol", 0.003)

        close_x = torch.abs(self.ball_pos[:, 0] - self.commands[:, 0]) < x_tol
        close_y = torch.abs(self.ball_pos[:, 1] - self.commands[:, 1]) < y_tol
        slow_vz = torch.abs(self.ball_vel[:, 2]) < vz_tol

        return close_x & close_y & slow_vz

    def _reward_success_event(self):
        success = self._success_mask()
        remaining_frac = 1.0 - (self.episode_length_buf.float() / float(self.max_episode_length))
        return success.float() * remaining_frac

    def _reward_fall_event(self):
        x_half = self.reward_cfg.get("platform_x_half", 0.11)
        y_half = self.reward_cfg.get("platform_y_half", 0.11)
        z_min  = self.reward_cfg.get("z_min_alive",     0.026)

        fell_x = torch.abs(self.ball_pos[:, 0]) > x_half
        fell_y = torch.abs(self.ball_pos[:, 1]) > y_half
        fell_z = self.ball_pos[:, 2] < z_min
        fell = fell_x | fell_y | fell_z
        return fell.float()
