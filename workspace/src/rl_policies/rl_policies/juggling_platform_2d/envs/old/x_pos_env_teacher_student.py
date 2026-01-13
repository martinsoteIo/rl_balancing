""" Go2 RL Environment """

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


LEFT_TH_MIN, LEFT_TH_MAX   = math.radians(90.0),  math.radians(180.0)
RIGHT_TH_MIN, RIGHT_TH_MAX = -math.radians(90.0), math.radians(90.0)

def angle_to_ctrl(theta, theta_min, theta_max, ctrl_min=-1.5, ctrl_max=0.0, invert=False):
    t = (theta - theta_min) / (theta_max - theta_min)
    if invert:
        u = ctrl_min + t*(ctrl_max - ctrl_min)
    else:
        u = ctrl_max - t*(ctrl_max - ctrl_min)
    return torch.clamp(u, min=ctrl_min, max=ctrl_max)

def a2ctrl_left(theta):
    return angle_to_ctrl(theta, LEFT_TH_MIN, LEFT_TH_MAX, invert=False)

def a2ctrl_right(theta):
    return angle_to_ctrl(theta, RIGHT_TH_MIN, RIGHT_TH_MAX, invert=False)

def pd_angle_from_x_torch(x_cmd, x_meas, vx_meas, Kp: float, Kd: float, ang_lim: float, angle_sign: float):
    ex = x_cmd - x_meas
    ev = -vx_meas
    ang = angle_sign * (Kp * ex + Kd * ev)
    return torch.clamp(ang, -ang_lim, ang_lim)

def ik_platform_angle_to_motors(angle, z):
    d, l1, l2 = 0.1979899, 0.08, 0.28
    c, s = torch.cos(angle), torch.sin(angle)
    xl =  d/2 - (l2/2) * c; yl =  z - (l2/2) * s
    xr =  (l2/2) * c - d/2; yr =  z + (l2/2) * s
    rl = torch.sqrt(xl*xl + yl*yl); rr = torch.sqrt(xr*xr + yr*yr)
    clip = lambda v: torch.clamp(v, -1.0, 1.0)
    alfa1 = torch.acos(clip((rl*rl - 2*l1*l1) / (2*l1*l1)))
    alfa2 = torch.acos(clip((rr*rr - 2*l1*l1) / (2*l1*l1)))
    beta1 = torch.atan2(yl, xl); beta2 = torch.atan2(yr, xr)
    gamma1 = torch.atan2(l1 * torch.sin(alfa1), l1 - l1 * torch.cos(alfa1))
    gamma2 = torch.atan2(l1 * torch.sin(alfa2), l1 - l1 * torch.cos(alfa2))
    theta1 = beta1 + gamma1; theta2 = beta2 - gamma2
    u_left  = a2ctrl_left(theta1)
    u_right = a2ctrl_right(theta2)
    return torch.stack([u_left, u_right], dim=-1)

class XPos2DEnv:
    """
    Simulated environment for the Go2 quadruped robot using the Genesis simulator.
    """
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, add_camera=False):
        """ 
        Initialize the Go2Env simulation environment.
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
        
        # teacher-student
        self.teacher_Kp = 24.63
        self.teacher_Kd = 0.815
        self.teacher_ang_lim = 0.40
        self.teacher_angle_sign = 1.0
        self.teacher_z_target = 0.03
        # self.imitation_sigma = 0.12
        
        self.use_residual_teacher = bool(self.env_cfg.get("use_residual_teacher", True))
        self.residual_scale = float(self.env_cfg.get("residual_scale", 0.05))
        self.imitation_sigma = float(self.env_cfg.get("imitation_sigma", 0.1))
        self.imitation_w0   = float(self.reward_cfg["reward_scales"].get("imitate_teacher", 20.0))
        self.imitation_w_min = float(self.env_cfg.get("imitation_w_min", 5.0))  # peso mínimo al final del currículo
        self.curriculum_iters = int(self.env_cfg.get("imitation_curriculum_iters", 2000))
        self.learn_iter = 0
        
        # Buffer de acciones objetivo del teacher
        self.teacher_targets = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
            
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(0.0, 1.25, 0.4),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
                refresh_rate=15,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=num_envs, show_world_frame=False),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                constraint_timeconst=0.05,
                integrator=gs.integrator.implicitfast,
            ),
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
                file="/home/admin/workspace/src/descriptions/juggling_platform_2d_description/urdf/robot_bounce.xml",
                pos=self.robot_init_pos.cpu().numpy(),
                quat=self.robot_init_quat.cpu().numpy(),
            ),
        )
        
        self.platform: RigidLink = self.robot.get_link("platform")
        self.ball: RigidLink = self.robot.get_link("ball")

        # change collision solver parameters
        # bouncy = [0.05, 0.2, 0.9, 0.95, 0.001, 0.5, 2.0]
        # self.platform.geoms[0].set_sol_params(bouncy)
        # self.ball.geoms[0].set_sol_params(bouncy)
        # floor.geoms[0].set_sol_params(bouncy)
        
        if add_camera:
            self.cam_0 = self.scene.add_camera(
                res=(1920, 1080),
                pos=(2.5, 0.5, 3.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=True,
            )

        # build
        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))
        
        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["motor_names"]]
        self.joints_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        ball_joint_x = self.robot.get_joint("ball_joint_x").idx_local
        ball_joint_z = self.robot.get_joint("ball_joint_z").idx_local
        self.ball_dof_idx = [ball_joint_x, ball_joint_z]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            # if name not in ["fall_event", "success_event", "imitate_teacher"]:
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
            [self.obs_scales["ball_pos_x"]],
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
        self.ball_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        
        # self.platform_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # self.platform_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # default pos buffers
        self.ball_init_pos = torch.tensor(self.env_cfg["ball_init_pos"], device=self.device)
        self.ball_init_qpos = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.ball_init_qpos[:, 0] = self.ball_init_pos[0]
        self.ball_init_qpos[:, 1] = self.ball_init_pos[2]
        
        rand_cfg = self.env_cfg.get("randomization", {})
        self.rand_ball_init_z_range = torch.tensor(
            rand_cfg.get("ball_init_z_range", [float(self.ball_init_pos[2]), float(self.ball_init_pos[2])]),
            device=self.device, dtype=gs.tc_float
        )
        self.rand_ball_init_x_range = torch.tensor(
            rand_cfg.get("ball_init_x_range", [float(self.ball_init_pos[0]), float(self.ball_init_pos[0])]),
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

    def _compute_teacher_targets(self):
        x_cmd  = self.commands[:, 0]
        x_meas = self.ball_pos[:, 0]
        vx_meas = self.ball_vel[:, 0]
        angle = pd_angle_from_x_torch(
            x_cmd, x_meas, vx_meas,
            self.teacher_Kp, self.teacher_Kd, self.teacher_ang_lim, self.teacher_angle_sign
        )
        z = torch.full_like(angle, self.teacher_z_target)
        motors_pos = ik_platform_angle_to_motors(angle, z)  # en [-1.5,0.0]
        self.teacher_targets[:] = motors_pos
        self.extras.setdefault("teacher", {})
        self.extras["teacher"]["angle"] = angle
        self.extras["teacher"]["motors_pos"] = motors_pos
        
        a0 = float(angle[0].cpu().item()) if angle.ndim == 1 else float(angle.cpu().item())
        mp0 = motors_pos[0].cpu().numpy().copy() if motors_pos.ndim == 2 else motors_pos.cpu().numpy().copy()
        print(f"[DEBUG _compute_teacher_targets] angle[0]={a0:+.6f} rad ({math.degrees(a0):+.2f}deg), teacher_motors[0]={mp0}")
            
    def _sample_commands(self, envs_idx):
        """
        Resample position command targets for the given environment indices.
        """
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["target_x"], (len(envs_idx),), self.device)

    def step(self, actions, is_train=True):
        """
        Apply the given actions, advance the simulation by one step, and return updated observations and rewards.
        """
        raw_actions = actions.clone() if isinstance(actions, torch.Tensor) else torch.tensor(actions, device=self.device)
        if getattr(self, "force_action_squash", False):
            proc_actions = torch.tanh(raw_actions)
        else:
            proc_actions = raw_actions
        max_raw = float(torch.max(torch.abs(raw_actions)).cpu().item())
        if max_raw > getattr(self, "raw_action_warn_threshold", 10.0):
            print(f"[WARN] raw policy outputs large: max_abs={max_raw:.2f}")
        self.actions = torch.clip(proc_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # Use last_actions if simulating latency
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # Compute residual in motor space from exec_actions
        residual = torch.clamp(exec_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]) \
                                * self.env_cfg["action_scale"] * self.residual_scale
        # self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_motors_pos
        # clipped_dof_pos = torch.clip(target_dof_pos, -1.5, 0.0)
        # # print(f"Actions: {actions}. \n Clipped Target DOF Pos: {target_dof_pos}. \n Clipped Actions: {self.actions}")
        # self.robot.control_dofs_position(clipped_dof_pos, self.motors_dof_idx)
        # self.scene.step()
        residual = torch.clamp(exec_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]) \
           * self.env_cfg["action_scale"] * self.residual_scale

        # 2) Teacher para etiquetas y/o ejecución
        self._compute_teacher_targets()  # [-1.5, 0.0] en motores

        if self.use_residual_teacher:
            # ejecuta teacher + residual (en espacio de motores)
            target_dof_pos = self.teacher_targets + residual
        else:
            # ejecuta sólo policy (como tenías)
            target_dof_pos = residual + self.default_motors_pos

        # límites físicos consistentes con teleop
        target_dof_pos = torch.clamp(target_dof_pos, -1.57, 0.0)
        
        # target_dof_pos_policy = exec_actions * self.env_cfg["action_scale"] + self.default_motors_pos
        # target_dof_pos_policy = torch.clamp(target_dof_pos_policy, -1.57, 0.0)
        self.extras.setdefault("control", {})
        self.extras["control"]["target_dof_pos_exec"] = target_dof_pos
        
        tt0 = self.teacher_targets[0].cpu().numpy().copy()
        te0 = target_dof_pos[0].cpu().numpy().copy()
        res0 = residual[0].cpu().numpy().copy() if residual.ndim == 2 else residual.cpu().numpy().copy()
        print(f"[DEBUG step pre-control] exec_actions[0]={self.actions[0].cpu().numpy()} residual[0]={res0}")
        print(f"[DEBUG step pre-control] teacher_targets[0]={tt0}, target_dof_pos_exec[0]={te0}, diff={te0 - tt0}")

        # 3) Ejecutar SOLO policy
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update observation buffers
        self.episode_length_buf += 1
        self.platform_quat[:] = self.platform.get_quat()
        self.platform_ang_vel[:] = self.platform.get_ang()
        self.inv_platform_quat = inv_quat(self.platform_quat)
        self.platform_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.platform_quat) * self.inv_platform_quat, self.platform_quat)
        )
        
        self.dof_pos[:] = self.robot.get_dofs_position(self.joints_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.joints_dof_idx)
        self.ball_pos[:] = self.ball.get_pos()
        self.ball_quat[:] = self.ball.get_quat()
        self.inv_ball_quat = inv_quat(self.ball_quat)
        self.ball_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.ball_quat) * self.inv_ball_quat, self.ball_quat)
        )
        self.ball_vel[:] = self.ball.get_vel()
        # print('BALL POS: ', self.ball_pos[:])
        # print('BALL VEL: ', self.ball_vel[:])
        bp = self.ball_pos[0].cpu().numpy().copy()
        bv = self.ball_vel[0].cpu().numpy().copy()
        print(f"[DEBUG step post-step] ball_pos[0]={bp}, ball_vel[0]={bv}")

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.platform_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.ball_pos[:, 0]) > self.env_cfg["termination_if_ball_x_greater_than"]
        self.reset_buf |= self.ball_pos[:, 2] < self.env_cfg["termination_if_ball_falls"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            # print(f"REWARDS: {reward_func()} * {self.reward_scales[name]}:{name}")
            # print("velocidad ball en z: ", self.ball_vel[:, 2])
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            
        # self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        self.obs_buf = torch.cat(
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
        )
        
        self.last_actions[:] = self.actions[:]
        if "imitate_teacher" in self.reward_scales:
            alpha = min(1.0, float(self.learn_iter) / float(max(1, self.curriculum_iters)))
            w = self.imitation_w0 * (1.0 - alpha) + self.imitation_w_min * alpha
            self.reward_scales["imitate_teacher"] = w
            self.learn_iter += 1
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

    def reset_idx(self, envs_idx):
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
        # z_low, z_high = self.rand_ball_init_z_range[0].item(), self.rand_ball_init_z_range[1].item()
        # z0 = gs_rand_float(z_low, z_high, (len(envs_idx),), self.device)
        x_low, x_high = self.rand_ball_init_x_range[0].item(), self.rand_ball_init_x_range[1].item()
        x0 = gs_rand_float(x_low, x_high, (len(envs_idx),), self.device)
        
     
        self.ball_pos[envs_idx] = self.ball_init_pos
        
        self.ball_init_qpos[envs_idx, 0] = x0
        # self.ball_init_qpos[envs_idx, 1] = z0
        # print(f'Posición random en X, Z ball: {x0, z0}')
        self.ball_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.ball_init_qpos[envs_idx],
            dofs_idx_local=self.ball_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.robot.zero_all_dofs_velocity(envs_idx)
        
        bmin, bmax = self.motor_bias_range[0].item(), self.motor_bias_range[1].item()
        if bmin != 0.0 or bmax != 0.0:
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

    def reset(self):
        """
        Reset all environments.
        """
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # Reward functions
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_similar_to_default(self):
    #     # Penalize joint poses far away from default pose
    #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_tracking_x(self):
        on_x = torch.abs(self.ball_pos[:, 0]) <= 0.11
        on_z = self.ball_pos[:, 2] >= 0.026
        alive = (on_x & on_z).float()

        err = self.commands[:, 0] - self.ball_pos[:, 0]
        r = torch.exp(- (err * err) / (self.reward_cfg["tracking_sigma"] ** 2))
        return r * alive  # [0,1]

    def _reward_stabilize_x_velocity(self):
        s  = self.reward_cfg["stability_sigma"]
        v0 = 0.04
        vz = torch.abs(self.ball_vel[:, 0])
        eff = torch.clamp((vz - v0) / s, min=0.0, max=1.0)
        return - (eff * eff)  # en [-1,0], así el scale puede ser POSITIVO (0.8)

    # def _success_mask(self):
    #     x_tol = self.reward_cfg.get("success_x_tol", 0.003)
    #     vz_tol = self.reward_cfg.get("success_vz_tol", 0.003)
    #     close_x = torch.abs(self.ball_pos[:, 0] - self.commands[:, 0]) < x_tol
    #     slow_vz = torch.abs(self.ball_vel[:, 2]) < vz_tol
    #     return close_x & slow_vz

    # def _reward_success_event(self):
    #     # BONUS ∝ tiempo restante (0..1)
    #     success = self._success_mask()
    #     remaining_frac = 1.0 - (self.episode_length_buf.float() / float(self.max_episode_length))
    #     return success.float() * remaining_frac  # ¡no escalar por dt en los scales!

    # def _reward_fall_event(self):
    #     # Caída por X fuera de la plataforma o Z por debajo del umbral
    #     fell_x = torch.abs(self.ball_pos[:, 0]) > 0.11   # borde en X
    #     fell_z = self.ball_pos[:, 2] < 0.026             # umbral en Z
    #     fell = fell_x | fell_z
    #     return fell.float()
    
    # def _reward_imitate_teacher(self):
    #     target_exec   = self.extras["control"]["target_dof_pos_exec"]  # lo que ejecutas
    #     target_teacher = self.teacher_targets
    #     diff = (target_exec - target_teacher) / self.imitation_sigma
    #     return torch.exp(- diff * diff).mean(dim=1)
    
    def _reward_imitate_teacher(self):
        # target_exec   = target executed (teacher+residual)  shape [N, nactions]
        # target_teacher = teacher targets                shape [N, nactions]
        target_exec   = self.extras["control"]["target_dof_pos_exec"]
        target_teacher = self.teacher_targets
        diff = target_exec - target_teacher
        mse = torch.mean(diff * diff, dim=1)  # 0..inf (lower better)
        # convert to a bounded reward: r = exp(-mse/(2*sigma^2))
        sigma = max(1e-3, self.imitation_sigma)
        r_exp = torch.exp(- mse / (2.0 * sigma * sigma))
        # Also give a linear-ish reward to keep gradient signal:
        r_lin = torch.clamp(1.0 - (mse / ( (sigma*4.0)**2 )), min=0.0, max=1.0)
        # combine them (weights tunables)
        return 0.6 * r_exp + 0.4 * r_lin   

