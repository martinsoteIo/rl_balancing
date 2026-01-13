""" Balancing Platform RL Environment """

import csv
import math
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink


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

class XPos3DEnv:
    """
    Simulated environment for the Balancing Platform using the Genesis simulator.
    """
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, add_camera=False):
        """ 
        Initialize the XPos3DEnv simulation environment.
        """
        self.device = gs.device
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

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(0.8, 0.8, 0.6),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=40,
                refresh_rate=15,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=num_envs, show_world_frame=False, ambient_light=(0.4, 0.4, 0.4)),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                constraint_timeconst=0.05,
                integrator=gs.integrator.implicitfast,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=show_viewer,
        )

        # add plane
        plane: RigidEntity = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        floor: RigidLink = plane.get_link("planeLink")

        # add robot
        self.robot_init_pos = torch.tensor(self.env_cfg["robot_init_pos"], device=self.device)
        self.robot_init_quat = torch.tensor(self.env_cfg["robot_init_quat"], device=self.device)

        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/home/admin/workspace/src/descriptions/balancing_platform_3d_description/urdf/robot.xml",
                pos=self.robot_init_pos.cpu().numpy(),
                quat=self.robot_init_quat.cpu().numpy(),
            ),
        )
        
        self.platform: RigidLink = self.robot.get_link("platform")
        self.ball: RigidLink = self.robot.get_link("ball")

        # change collision solver parameters
        bouncy = [0.05, 0.2, 0.9, 0.95, 0.001, 0.5, 2.0]
        self.platform.geoms[0].set_sol_params(bouncy)
        self.ball.geoms[0].set_sol_params(bouncy)
        floor.geoms[0].set_sol_params(bouncy)

        if add_camera:
            self.cam_0 = self.scene.add_camera(
                res=(720, 480),
                pos=(1.5, 0.0, 1.5),
                lookat=(0.0, 0.0, 0.5),
                fov=30,
                GUI=True,
            )

        # build
        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["motor_names"]]
        self.joints_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        ball_joint = self.robot.get_joint("ball_joint")
        self.ball_dof_idx = list(range(ball_joint.dof_start, ball_joint.dof_start + 6))

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        # general buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        # command buffers
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["ball_pos_x"], self.obs_scales["ball_pos_y"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        # action buffers
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # obs buffers
        self.platform_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.platform_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.ball_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.ball_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.last_ball_z = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.ball_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        # default pos buffers
        self.ball_init_pos = torch.tensor(self.env_cfg["ball_init_pos"], device=self.device)
        self.ball_init_qpos = torch.zeros((self.num_envs, 6), device=self.device, dtype=gs.tc_float) # 3(x,y,z)+4(qx,qy,qz,qw)
        self.ball_init_qpos[:, 0] = self.ball_init_pos[0]
        self.ball_init_qpos[:, 1] = self.ball_init_pos[1]
        self.ball_init_qpos[:, 2] = self.ball_init_pos[2]
        self.ball_init_qpos[:, 3] = 0.0
        self.ball_init_qpos[:, 4] = 0.0
        self.ball_init_qpos[:, 5] = 0.0
        
        rand_cfg = self.env_cfg.get("randomization", {})
        self.rand_ball_init_x_range = torch.tensor(
            rand_cfg.get("ball_init_x_range", [float(self.ball_init_pos[0]), float(self.ball_init_pos[0])]),
            device=self.device, dtype=gs.tc_float
        )
        self.rand_ball_init_y_range = torch.tensor(
            rand_cfg.get("ball_init_y_range", [float(self.ball_init_pos[1]), float(self.ball_init_pos[1])]),
            device=self.device, dtype=gs.tc_float
        )
        self.rand_ball_init_z_range = torch.tensor(
            rand_cfg.get("ball_init_z_range", [float(self.ball_init_pos[2]), float(self.ball_init_pos[2])]),
            device=self.device, dtype=gs.tc_float
        )
        self.motor_bias_range = torch.tensor(rand_cfg.get("motor_bias_range", [0.0, 0.0]), device=self.device, dtype=gs.tc_float)
        self.motor_bias = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)

        
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.default_motors_pos = torch.tensor(
            [self.env_cfg["default_motor_angles"][name] for name in self.env_cfg["motor_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.extras = {}  # extra information for logging
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

        self.actions[:, 0] = torch.clip(actions[:, 0], -0.3, 1.0)
        self.actions[:, 1:] = torch.clip(actions[:, 1:], -1.0, 0.3)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_motors_pos
        clipped_dof_pos = torch.clip(target_dof_pos, -2.0, 2.0)

        self.robot.control_dofs_position(clipped_dof_pos, self.motors_dof_idx)
        self.scene.step()
         
        self.episode_length_buf += 1
        self.platform_quat[:] = self.platform.get_quat()
        self.platform_ang_vel[:] = self.platform.get_ang()
        self.platform_euler = quat_to_xyz(self.platform_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.joints_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.joints_dof_idx)
        self.ball_pos[:] = self.ball.get_pos()
        self.ball_quat[:] = self.ball.get_quat()
        self.inv_ball_quat = inv_quat(self.ball_quat)
        self.ball_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.ball_quat) * self.inv_ball_quat, self.ball_quat)
        )
        self.ball_vel[:] = self.ball.get_vel()

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.platform_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.abs(self.platform_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.ball_pos[:, 0]) > self.env_cfg["termination_if_ball_x_greater_than"]
        self.reset_buf |= torch.abs(self.ball_pos[:, 1]) > self.env_cfg["termination_if_ball_y_greater_than"]
        self.reset_buf |= self.ball_pos[:, 2] < self.env_cfg["termination_if_ball_falls"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        
        if is_train:
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        else:
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten(), is_train=False)
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.obs_buf = torch.cat(
            [
                (self.platform_euler[:, 1] * self.obs_scales["platform_pitch"]).unsqueeze(-1), # 1
                (self.platform_euler[:, 0] * self.obs_scales["platform_roll"]).unsqueeze(-1), # 1
                (self.platform_ang_vel[:, 1] * self.obs_scales["platform_pitch_vel"]).unsqueeze(-1), # 1
                (self.platform_ang_vel[:, 0] * self.obs_scales["platform_roll_vel"]).unsqueeze(-1), # 1
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 15
                self.dof_vel * self.obs_scales["dof_vel"],  # 15
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
        )

        self.last_ball_z[:] = self.ball_pos[:, 2].unsqueeze(-1)
        self.last_actions[:] = self.actions[:]
        
        # Detect and reset corrupt environments with NaN
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

    def reset_idx(self, envs_idx, is_train=True):
        """
        Reset the simulation state for the specified environment indices.
        """
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.joints_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # # reset ball
        if is_train:
            z_low, z_high = self.rand_ball_init_z_range[0].item(), self.rand_ball_init_z_range[1].item()
            z0 = gs_rand_float(z_low, z_high, (len(envs_idx),), self.device)
            x_low, x_high = self.rand_ball_init_x_range[0].item(), self.rand_ball_init_x_range[1].item()
            x0 = gs_rand_float(x_low, x_high, (len(envs_idx),), self.device)
            y_low, y_high = self.rand_ball_init_y_range[0].item(), self.rand_ball_init_y_range[1].item()
            y0 = gs_rand_float(y_low, y_high, (len(envs_idx),), self.device)
                
            self.ball_pos[envs_idx] = self.ball_init_pos
            self.ball_init_qpos[envs_idx, 0] = x0
            self.ball_init_qpos[envs_idx, 1] = y0
            self.ball_init_qpos[envs_idx, 2] = z0
        else:
            self.ball_pos[envs_idx] = self.ball_init_pos
            self.ball_init_qpos[envs_idx, 0] = self.ball_init_pos[0]
            self.ball_init_qpos[envs_idx, 1] = self.ball_init_pos[1]
            self.ball_init_qpos[envs_idx, 2] = self.ball_init_pos[2]
        self.ball_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.ball_init_qpos[envs_idx],
            dofs_idx_local=self.ball_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.robot.zero_all_dofs_velocity(envs_idx)

        bmin, bmax = self.motor_bias_range[0].item(), self.motor_bias_range[1].item()
        if bmin != 0.0 or bmax != 0.0 and is_train:
            self.motor_bias[envs_idx] = gs_rand_float(bmin, bmax, (len(envs_idx), self.num_actions), self.device)
        else:
            self.motor_bias[envs_idx] = 0.0
            
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

    def reset(self, is_train=True):
        """
        Reset all environments.
        """
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device), is_train=is_train)
        return self.obs_buf, None

    # Reward functions
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_tracking_xy(self):
        on_x = torch.abs(self.ball_pos[:, 0]) <= 0.20
        on_y = torch.abs(self.ball_pos[:, 1]) <= 0.20
        on_z = self.ball_pos[:, 2] >= 0.050
        alive = (on_x & on_y & on_z).float()
        # Penalize ball is far from objetive xy
        ex = self.commands[:, 0] - self.ball_pos[:, 0]
        ey = self.commands[:, 1] - self.ball_pos[:, 1]

        sigma = float(self.reward_cfg.get("tracking_sigma", 0.06))
        inv2s2 = 1.0 / (sigma * sigma)

        d2 = ex*ex + ey*ey
        r = torch.exp(- d2 * inv2s2)
        return r * alive

    def _reward_stabilize_z_velocity(self):
        s  = self.reward_cfg["stability_sigma"]     
        v0 = 0.0                                     
        vz = torch.abs(self.ball_vel[:, 2])
        reward = torch.exp(-torch.square(vz - v0) / (2 * s**2))
        return reward

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
        x_half = self.reward_cfg.get("platform_x_half")
        y_half = self.reward_cfg.get("platform_y_half")
        z_min  = self.reward_cfg.get("z_min_alive")

        fell_x = torch.abs(self.ball_pos[:, 0]) > x_half
        fell_y = torch.abs(self.ball_pos[:, 1]) > y_half
        fell_z = torch.abs(self.ball_pos[:, 2]) < z_min
        fell = fell_x | fell_y | fell_z
        return fell.float()

