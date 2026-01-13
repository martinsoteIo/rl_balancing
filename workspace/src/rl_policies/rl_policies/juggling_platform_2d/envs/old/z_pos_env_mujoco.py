""" Go2 RL Environment """

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

class ZPos2DEnv:
    """
    Simulated environment for the Go2 quadruped robot using the Genesis simulator.
    """
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, add_camera=False):
        """ 
        Initialize the Go2Env simulation environment.
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

        # # create scene
        # self.scene = gs.Scene(
        #     sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
        #     viewer_options=gs.options.ViewerOptions(
        #         max_FPS=int(0.5 / self.dt),
        #         camera_pos=(0.0, 1.25, 0.4),
        #         camera_lookat=(0.0, 0.0, 0.3),
        #         camera_fov=40,
        #     ),
        #     vis_options=gs.options.VisOptions(n_rendered_envs=num_envs, show_world_frame=False),
        #     rigid_options=gs.options.RigidOptions(
        #         dt=self.dt,
        #         constraint_solver=gs.constraint_solver.Newton,
        #         enable_collision=True,
        #         enable_joint_limit=True,
        #         constraint_timeconst=0.05,
        #         integrator=gs.integrator.implicitfast,
        #     ),
        #     show_viewer=show_viewer,
        # )
        
        model_path = "/home/admin/workspace/src/descriptions/juggling_platform_2d_description/urdf/robot_mujoco_z.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # ball_z = 7  # índice confirmado de qpos
        # print("Antes de step:")
        # print("qpos[z]:", self.data.qpos[ball_z])
        # print("qvel[z]:", self.data.qvel[ball_z])
        # print("qfrc_bias[z]:", self.data.qfrc_bias[ball_z])
        # print("qfrc_applied[z]:", self.data.qfrc_applied[ball_z])
        # print("qfrc_constraint[z]:", self.data.qfrc_constraint[ball_z])
        # mujoco.mj_step(self.model, self.data)
        # print("\nDespués de un paso:")
        # print("qpos[z]:", self.data.qpos[ball_z])
        # print("qvel[z]:", self.data.qvel[ball_z])
        # print("qfrc_bias[z]:", self.data.qfrc_bias[ball_z])
        # print("qfrc_applied[z]:", self.data.qfrc_applied[ball_z])
        # print("qfrc_constraint[z]:", self.data.qfrc_constraint[ball_z])

        mujoco.mj_resetData(self.model, self.data)
        # Prueba mínima: comprobar si la bola cae con velocidad inicial
        # ball_joint_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint_z")
        # print("== PRUEBA DE CAÍDA DE LA BOLA ==")
        # for _ in range(100):
        #     mujoco.mj_step(self.model, self.data)
        #     print("ball z pos:", self.data.qpos[ball_joint_z_id])
        # print("== FIN DE PRUEBA ==")


        # if show_viewer:
        #     self.viewer_data_lock = threading.Lock()
        #     self.viewer_thread = threading.Thread(target=launch_viewer, args=(self.model, self.data, self.viewer_data_lock), daemon=True)
        #     self.viewer_thread.start()
        # adapt to have same camera pose as genesis

        # self.platform: RigidLink = self.robot.get_link("platform")
        # self.ball: RigidLink = self.robot.get_link("ball")
        self.platform_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        # names to indices
        self.motor_names = ["left_motor", "right_motor"]
        self.motors_dof_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.motor_names]
        
        self.joint_names = [
            "left_motor_joint", "right_motor_joint", "left_arm_joint",
            "right_arm_joint", "left_platform_joint", "right_platform_joint",
        ]
        self.joints_dof_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.ball_joint_names = ["ball_joint_x", "ball_joint_z"]
        self.ball_dof_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.ball_joint_names]
        
        # ball_joint_x = self.robot.get_joint("ball_joint_x").idx_local
        # ball_joint_z = self.robot.get_joint("ball_joint_z").idx_local
        # self.ball_dof_idx = [ball_joint_x, ball_joint_z]

        # # PD control parameters
        # self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        # self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            if name not in ["bounce", "drop"]:
                self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # # initialize buffers
        # # general buffers
        # self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        # self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        # self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        # self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        # # command buffers
        # self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        # self.commands_scale = torch.tensor(
        #     [self.obs_scales["ball_pos_x"]],
        #     device=self.device,
        #     dtype=gs.tc_float,
        # )
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device)
        self.commands_scale = torch.tensor([self.obs_scales["ball_pos_x"]], device=self.device)

        # # action buffers
        # self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        # self.last_actions = torch.zeros_like(self.actions)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        # # obs buffers
        # self.platform_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        # self.platform_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.platform_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.platform_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        # self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        # self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        # self.ball_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # self.ball_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        # self.last_ball_z = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)
        # self.ball_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.ball_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ball_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_ball_z = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_vel = torch.zeros((self.num_envs, 3), device=self.device)

        # # default pos buffers
        # self.ball_init_pos = torch.tensor(self.env_cfg["ball_init_pos"], device=self.device)
        # self.ball_init_qpos = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        # self.ball_init_qpos[:, 0] = self.ball_init_pos[0]
        # self.ball_init_qpos[:, 1] = self.ball_init_pos[2]
        self.ball_init_pos = torch.tensor(self.env_cfg["ball_init_pos"], device=self.device)
        self.ball_init_qpos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.ball_init_qpos[:, 0] = self.ball_init_pos[0]
        # self.ball_init_qpos[:, 1] = self.ball_init_pos[2]
        self.ball_init_qpos[:, 1] = self.ball_init_pos[2]
        
        # self.default_dof_pos = torch.tensor(
        #     [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
        #     device=self.device,
        #     dtype=gs.tc_float,
        # )
        # self.default_motors_pos = torch.tensor(
        #     [self.env_cfg["default_motor_angles"][name] for name in self.env_cfg["motor_names"]],
        #     device=self.device,
        #     dtype=gs.tc_float,
        # )
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
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_adr = self.model.jnt_qposadr[i]
            # print(f"{i}: {name} → qpos index: {qpos_adr}")
        with self.viewer_data_lock:
            # print(f"\n--- DEBUG STATE init ---")
            for j, dof_id in enumerate(self.joints_dof_idx):
                self.data.qpos[dof_id] = self.default_dof_pos[j].item()
            for j, dof_id in enumerate(self.ball_dof_idx):
                self.data.qpos[dof_id] = self.ball_init_qpos[0, j].item()
            mujoco.mj_forward(self.model, self.data)
            
        if show_viewer:
            self.viewer_thread = threading.Thread(target=launch_viewer, args=(self.model, self.data, self.viewer_data_lock), daemon=True)
            self.viewer_thread.start()
        # print("\n--- DEBUG STATE init ---")
        # for i in range(self.model.njnt):
        #     name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        #     print(f"Joint {i} ({name}): qpos = {self.data.qpos[i]:.4f}, qvel = {self.data.qvel[i]:.4f}")
        # for i in range(self.model.nbody):
        #     name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     pos = self.data.xpos[i]
        #     quat = self.data.xquat[i]
        #     print(f"Body {i} ({name}): xpos = {np.round(pos, 3)}, xquat = {np.round(quat, 3)}")
        # print("--------------------------\n")
        self.extras = {}  # extra information for logging
        self.extras["observations"] = {}
        
        # print(f"\n--- DEBUG ---")
        # print(f"Step: {self.episode_length_buf[0].item()}")
        # print(f"dof_pos (joint angles): {self.default_dof_pos}")
        # print(f"dof_pos_motor (motors angles): {self.default_motors_pos}")
        # print(f"ball_pos (x, y, z): {self.ball_init_pos}")
        # print(f"ball_quat (x, y, z, w): {self.ball_quat}")
        # print(f"--------------------------\n")
    def _sample_commands(self, envs_idx):
        """
        Resample velocity command targets for the given environment indices.
        """
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["target_x"], (len(envs_idx),), self.device)

    def step(self, actions, is_train=True):
        """
        Apply the given actions, advance the simulation by one step, and return updated observations and rewards.
        """
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_motors_pos
        clipped_dof_pos = torch.clip(target_dof_pos, -1.5, 0.0)
        clipped_dof_pos_np = clipped_dof_pos.cpu().numpy()
        # self.robot.control_dofs_position(clipped_dof_pos, self.motors_dof_idx)
        for i in range(self.num_envs):
            for j, dof_id in enumerate(self.motors_dof_idx):
                # self.data.ctrl[dof_id] = clipped_dof_pos[i, j].item()
                self.data.ctrl[dof_id] = clipped_dof_pos_np[i, j]
                # print("self.data.ctrl: ", self.data.ctrl[dof_id])
        with self.viewer_data_lock:
            mujoco.mj_step(self.model, self.data)
        # print("qvel:", self.data.qvel)
        # update observation buffers
        self.episode_length_buf += 1
        # self.platform_quat[:] = self.platform.get_quat()
        # self.platform_ang_vel[:] = self.platform.get_ang()
        # self.inv_platform_quat = inv_quat(self.platform_quat)
        # self.platform_euler = quat_to_xyz(
        #     transform_quat_by_quat(torch.ones_like(self.platform_quat) * self.inv_platform_quat, self.platform_quat)
        # )
        quat_np = self.data.xquat[self.platform_body_id]
        self.platform_quat.copy_(torch.from_numpy(quat_np).to(self.device))
        angvel_np = self.data.cvel[self.platform_body_id, 3:6]
        self.platform_ang_vel.copy_(torch.from_numpy(angvel_np).to(self.device))
        self.inv_platform_quat = inv_quat(self.platform_quat)
        rel_quat = transform_quat_by_quat(
            self.inv_platform_quat.expand_as(self.platform_quat),
            self.platform_quat
        )
        self.platform_euler = quat_to_xyz(rel_quat)

        # self.dof_pos[:] = self.robot.get_dofs_position(self.joints_dof_idx)
        # self.dof_vel[:] = self.robot.get_dofs_velocity(self.joints_dof_idx)
        # self.ball_pos[:] = self.ball.get_pos()
        # self.ball_quat[:] = self.ball.get_quat()
        # self.inv_ball_quat = inv_quat(self.ball_quat)
        # self.ball_euler = quat_to_xyz(
        #     transform_quat_by_quat(torch.ones_like(self.ball_quat) * self.inv_ball_quat, self.ball_quat)
        # )
        # self.ball_vel[:] = self.ball.get_vel()
        qpos_np = self.data.qpos[self.joints_dof_idx]               # array (num_dof,)
        qvel_np = self.data.qvel[self.joints_dof_idx]
        self.dof_pos.copy_(torch.from_numpy(qpos_np).to(self.device))
        self.dof_vel.copy_(torch.from_numpy(qvel_np).to(self.device))

        pos_np = self.data.xpos[self.ball_body_id]
        self.ball_pos.copy_(torch.from_numpy(pos_np).to(self.device))
        quat_b_np = self.data.xquat[self.ball_body_id]
        self.ball_quat.copy_(torch.from_numpy(quat_b_np).to(self.device))
        self.inv_ball_quat = inv_quat(self.ball_quat)

        rel_quat_b = transform_quat_by_quat(
            self.inv_ball_quat.expand_as(self.ball_quat),
            self.ball_quat
        )
        self.ball_euler = quat_to_xyz(rel_quat_b)

        linvel_np = self.data.cvel[self.ball_body_id, :3]
        self.ball_vel.copy_(torch.from_numpy(linvel_np).to(self.device))        

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        if is_train:
            self._sample_commands(envs_idx)
            random_idxs = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
            self._sample_commands(random_idxs)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.platform_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.ball_pos[:, 0]) > self.env_cfg["termination_if_ball_x_greater_than"]
        self.reset_buf |= self.ball_pos[:, 2] < self.env_cfg["termination_if_ball_falls"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute observations
        self.obs_buf.copy_(torch.cat(
            [
                (self.platform_euler[:, 1] * self.obs_scales["platform_pitch"]).unsqueeze(-1), # 1
                (self.platform_ang_vel[:, 1] * self.obs_scales["platform_pitch_vel"]).unsqueeze(-1), # 1
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 6
                self.dof_vel * self.obs_scales["dof_vel"],  # 6
                (self.ball_pos[:, 0] * self.obs_scales["ball_pos_x"]).unsqueeze(-1), # 1
                (self.ball_pos[:, 2] * self.obs_scales["ball_pos_z"]).unsqueeze(-1), # 1
                (self.ball_vel[:, 0] * self.obs_scales["ball_vel_x"]).unsqueeze(-1), # 1
                (self.ball_vel[:, 2] * self.obs_scales["ball_vel_z"]).unsqueeze(-1), # 1
                self.commands * self.commands_scale, # 1
                self.actions, # 2
            ],
            axis=-1,
        ))

        self.last_ball_z.copy_(self.ball_pos[:, 2].unsqueeze(-1))
        self.last_actions.copy_(self.actions)

        # Detect and reset corrupt environments with NaN
        bad_envs = torch.isnan(self.obs_buf).any(dim=1).nonzero(as_tuple=False).flatten()
        if len(bad_envs) > 0:
            print(f"\033[93mResetting {len(bad_envs)} environments due to NaN in obs_buf.\033[0m")
            self.reset_idx(bad_envs)
            self.obs_buf[bad_envs] = 0.0

        # self.obs_buf[bad_envs].zero_()
        self.extras["observations"]["critic"] = self.obs_buf
        # print("ball z vel:", self.data.qvel[self.ball_dof_idx[1]])
        # print("ball z pos:", self.data.qpos[self.ball_dof_idx[1]])
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
        # print("self.data.qpos: ", self.data.qpos)
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        # self.robot.set_dofs_position(
        #     position=self.dof_pos[envs_idx],
        #     dofs_idx_local=self.joints_dof_idx,
        #     zero_velocity=True,
        #     envs_idx=envs_idx,
        # )
        for i, dof_id in enumerate(self.joints_dof_idx):
            self.data.qpos[dof_id] = self.dof_pos[envs_idx][0, i].item()
            self.data.qvel[dof_id] = 0.0
            
        # # reset ball
        # x_low, x_high = self.env_cfg["ball_spawn_range"]["x"]
        # z_low, z_high = self.env_cfg["ball_spawn_range"]["z"]
        # random_x = gs_rand_float(x_low, x_high, (len(envs_idx),), self.device)
        # random_z = gs_rand_float(z_low, z_high, (len(envs_idx),), self.device)
        # self.ball_pos[envs_idx, 0] = random_x
        # self.ball_pos[envs_idx, 1] = 0.0  # plane XZ
        # self.ball_pos[envs_idx, 2] = random_z
                
        self.ball_pos[envs_idx] = self.ball_init_pos
        self.ball_vel[envs_idx] = 0.0
        
        # self.robot.set_dofs_position(
        #     position=self.ball_init_qpos[envs_idx],
        #     dofs_idx_local=self.ball_dof_idx,
        #     zero_velocity=True,
        #     envs_idx=envs_idx,
        # )
        # print("gravity:", self.model.opt.gravity)
        for j, dof_id in enumerate(self.ball_dof_idx):
            self.data.qpos[dof_id] = self.ball_init_qpos[envs_idx][0, j].item()
            self.data.qvel[dof_id] = 0.0
        
        # self.robot.zero_all_dofs_velocity(envs_idx)
        for j, dof_id in enumerate(self.joints_dof_idx):
            self.data.qvel[dof_id] = 0.0
    
        # reset buffers
        self.last_actions[envs_idx] = 0.0
        # self.actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._sample_commands(envs_idx)
        with self.viewer_data_lock:
            mujoco.mj_forward(self.model, self.data)
        
    def reset(self):
        """
        Reset all environments.
        """
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # Reward functions
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_similar_to_default(self):
    #     # Penalize joint poses far away from default pose
    #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_tracking_x(self):
        # Penalize ball is far from objetive x
        ball_pos_error = torch.square(self.commands[:, 0] - self.ball_pos[:, 0])
        return torch.exp(-ball_pos_error / self.reward_cfg["tracking_sigma"])

    def _reward_ball_lin_vel_z(self):
        # Penalize ball z velocity far from zero
        return torch.square(self.ball_vel[:, 2])

    def _reward_bounce(self):
        # Bonus SOLO cuando hay cruce hacia arriba cerca de la plataforma y en x objetivo,
        # además con velocidad de despegue cercana a la requerida para alcanzar H.
        g = 9.81
        platform_z = self.platform.get_pos()[:, 2]
        vz = self.ball_vel[:, 2]
        last_vz = self.last_ball_vel_z.squeeze(-1)

        cross_up = (last_vz <= 0.0) & (vz > 0.0)
        near_plate = torch.abs(self.ball_pos[:, 2] - platform_z) < self.reward_cfg["eps_plate"]
        near_x = torch.abs(self.ball_pos[:, 0] - self.commands[:, 0]) < self.reward_cfg["eps_x"]
        mask = (cross_up & near_plate & near_x)

        # vH = sqrt(2 g H)
        vH = torch.sqrt(torch.clamp(2.0 * g * self.reward_cfg["H_apex"], min=0.0))
        vel_term = torch.exp(- ((vz - vH) ** 2) / (2.0 * (self.reward_cfg.get("v_sigma", 0.40) ** 2)))

        # Afinidad adicional a x objetivo en el instante del bote (más exigente que el shaping continuo)
        ex = self.ball_pos[:, 0] - self.commands[:, 0]
        x_term = torch.exp(- (ex * ex) / (2.0 * (self.reward_cfg["eps_x"] ** 2)))

        r = torch.zeros_like(vz)
        r[mask] = vel_term[mask] * x_term[mask]
        return r

    def _reward_apex_height(self):
        # Premio altura del ápex ~ H por encima de la plataforma SOLO cerca del ápice
        platform_z = self.platform.get_pos()[:, 2]
        vz = self.ball_vel[:, 2]
        last_vz = self.last_ball_vel_z.squeeze(-1)

        cross_down = (last_vz > 0.0) & (vz <= 0.0)

        z_target = platform_z + self.reward_cfg["H_apex"]
        dz = self.ball_pos[:, 2] - z_target
        ex = self.ball_pos[:, 0] - self.commands[:, 0]

        height_term = torch.exp(- (dz * dz) / (2.0 * self.reward_cfg["sigma_h"] ** 2))
        x_term = torch.exp(- (ex * ex) / (2.0 * (self.reward_cfg["eps_x"] ** 2)))

        r = torch.zeros_like(vz)
        r[cross_down] = height_term[cross_down] * x_term[cross_down]
        return r

    def _reward_above_platform(self):
        platform_z = self.platform.get_pos()[:, 2]
        r = (self.ball_pos[:, 2] > (platform_z + self.reward_cfg["z_margin"])).float()
        return r

    def _reward_level_platform(self):
        # Penaliza pitch^2 (tu pitch está en self.platform_euler[:,1])
        pitch = self.platform_euler[:, 1]
        return - pitch * pitch

    def _reward_action_rate(self):
        return - torch.sum((self.last_actions - self.actions)**2, dim=1)

    def _reward_idle(self):
        # Penaliza "no actuar" si no hay altura suficiente
        a_norm = torch.norm(self.actions, dim=1)
        platform_z = self.platform.get_pos()[:, 2]
        low_height = (self.ball_pos[:, 2] < platform_z + self.reward_cfg["H_min"]).float()
        idle = (a_norm < self.reward_cfg["a_idle_eps"]).float()
        return - idle * low_height

    def _reward_drop(self):
        dropped = (self.ball_pos[:, 2] < self.env_cfg["termination_if_ball_falls"]).float()
        return - dropped * 1.0  # pon gran escala negativa en reward_scales["drop"]
