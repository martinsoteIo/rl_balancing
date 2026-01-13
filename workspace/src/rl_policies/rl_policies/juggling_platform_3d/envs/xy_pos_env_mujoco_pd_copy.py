""" Balancing Platform RL Environment """

import math
import threading
import torch
import numpy as np
import mujoco
import mujoco.viewer
from genesis.utils.geom import quat_to_xyz


class XPos3DEnv:
    """
    Simulated environment for the 3D Juggling Platform PD using the Genesis simulator.
    """

    def __init__(self, env_cfg: dict, show_viewer: bool = False):
        """ 
        Initialize the 3D Juggling Platform PD simulation environment.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.viewer_data_lock = threading.Lock()
    
        self.num_dof = env_cfg["num_dof"]
        self.num_actions = env_cfg["num_actions"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        
        model_path = "/home/admin/workspace/src/descriptions/juggling_platform_3d_description/urdf/robot.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetData(self.model, self.data)

        self.platform_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        # names to indices
        self.motor_names = env_cfg["motor_names"]
        self.motors_dof_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.motor_names]

        self.joint_names = env_cfg["joint_names"]
        self.joints_dof_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]

        self.joint_ids = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names}
        self.qpos_adr = {n: self.model.jnt_qposadr[self.joint_ids[n]] for n in self.joint_names}
        self.qvel_adr = {n: self.model.jnt_dofadr[self.joint_ids[n]] for n in self.joint_names}
        self.jnt_qpos_idx = np.array([self.model.jnt_qposadr[j] for j in self.joints_dof_idx], dtype=np.int32)
        self.jnt_qvel_idx = np.array([self.model.jnt_dofadr[j] for j in self.joints_dof_idx], dtype=np.int32)
        
        self.ball_joint_names = ["ball_joint"]
        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qpos_adr = self.model.jnt_qposadr[self.ball_joint_id]
        self.ball_qvel_adr = self.model.jnt_dofadr[self.ball_joint_id]

        # initialize buffers
        self.reset_buf = torch.ones((1,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((1,), device=self.device, dtype=torch.int)
        self.actions = torch.zeros((1, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.platform_quat = torch.zeros((1, 4), device=self.device)
        self.platform_ang_vel = torch.zeros((1, 3), device=self.device)

        self.dof_pos = torch.zeros((1, self.num_dof), device=self.device)
        self.dof_vel = torch.zeros((1, self.num_dof), device=self.device)
        self.ball_pos = torch.zeros((1, 3), device=self.device)
        self.ball_quat = torch.zeros((1, 4), device=self.device)
        self.last_ball_z = torch.zeros((1, 1), device=self.device)
        self.ball_vel = torch.zeros((1, 3), device=self.device)
        
        self.ball_init_pos = torch.tensor(self.env_cfg["ball_init_pos"], device=self.device)
        self.ball_init_qpos = torch.zeros((1, 7), device=self.device, dtype=torch.float)
        self.ball_init_qvel = torch.zeros((1, 6), device=self.device, dtype=torch.float)
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
            self.viewer_thread = threading.Thread(target=self._launch_viewer, daemon=True)
            self.viewer_thread.start()

    def _launch_viewer(self) -> None:
        """
        Launch the MuJoCo viewer in a separate thread.
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                with self.viewer_data_lock:
                    viewer.sync()

    def _reset_idx(self, envs_idx: torch.Tensor) -> None:
        """
        Reset the simulation state for the specified environment indices.
        """
        if len(envs_idx) == 0:
            return

        print('Resetting environments...')
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

        with self.viewer_data_lock:
            mujoco.mj_forward(self.model, self.data)
            
    def step(self, actions: torch.Tensor) -> None:
        """
        Apply the given actions, advance the simulation by one step, and return updated observations and rewards.
        """
        self.actions = actions
        target_dof_pos = self.last_actions if self.simulate_action_latency else self.actions
        motor_ranges = self.env_cfg["motor_ranges"]
        
        # Clip actions and read data
        with self.viewer_data_lock:
            for j, dof_id in enumerate(self.motors_dof_idx):
                dof_min, dof_max = motor_ranges[j]
                clipped_pos = float(torch.clip(target_dof_pos[0, j], dof_min, dof_max).item())
                self.data.ctrl[dof_id] = clipped_pos
                
            mujoco.mj_step(self.model, self.data)
            
            # Read data from MuJoCo
            quat_p = self.data.xquat[self.platform_body_id].astype(np.float32)
            R_plat  = self.data.xmat[self.platform_body_id].reshape(3, 3).astype(np.float32)
            qpos_np = self.data.qpos[self.jnt_qpos_idx].astype(np.float32)
            qvel_np = self.data.qvel[self.jnt_qvel_idx].astype(np.float32)
            pos_b = self.data.xpos[self.ball_body_id].astype(np.float32)
            quat_b = self.data.xquat[self.ball_body_id].astype(np.float32)
            v_body_b = self.data.cvel[self.ball_body_id, 3:6].astype(np.float32)
            R_ball = self.data.xmat[self.ball_body_id].reshape(3, 3).astype(np.float32)
            omega_body_p = self.data.cvel[self.platform_body_id, 0:3].astype(np.float32)
        
        self.episode_length_buf += 1
        
        self.platform_quat.copy_(torch.from_numpy(quat_p).to(self.device))
        self.platform_euler = quat_to_xyz(self.platform_quat)
        omega_world_p = R_plat @ omega_body_p
        self.platform_ang_vel.copy_(torch.from_numpy(omega_world_p).to(self.device))

        self.dof_pos.copy_(torch.from_numpy(qpos_np).to(self.device))
        self.dof_vel.copy_(torch.from_numpy(qvel_np).to(self.device))

        self.ball_pos.copy_(torch.from_numpy(pos_b).to(self.device))
        self.ball_quat.copy_(torch.from_numpy(quat_b).to(self.device))
        v_world_b = R_ball @ v_body_b
        self.ball_vel.copy_(torch.from_numpy(v_world_b).to(self.device))

        # Check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.platform_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.abs(self.platform_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.ball_pos[:, 0]) > self.env_cfg["termination_if_ball_x_greater_than"]
        self.reset_buf |= torch.abs(self.ball_pos[:, 1]) > self.env_cfg["termination_if_ball_y_greater_than"]
        self.reset_buf |= self.ball_pos[:, 2] < self.env_cfg["termination_if_ball_falls"]

        self._reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        self.last_actions.copy_(self.actions)
