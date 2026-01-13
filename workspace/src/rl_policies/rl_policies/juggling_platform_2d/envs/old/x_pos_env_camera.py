""" Balancing Platform RL Environment """

import math
import torch
import cv2 as cv
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from ...camera.ball_detector import BallDetector

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

class XPos2DCameraEnv:
    """
    Simulated environment for the XPos2D platform robot using the Genesis simulator.
    """
    def _grab_camera_frame(self, env_idx: int):
        # Robust camera rendering across Genesis versions / APIs.
        # Returns a BGR uint8 image of shape (height, width, 3).
        # Tries multiple calling conventions and handles tuple/list returns.
        def _black_frame():
            w, h = self.cam_res
            return np.zeros((h, w, 3), dtype=np.uint8)

        # Helper to normalize an output into an RGB ndarray or None
        def _extract_image(out):
            if out is None:
                return None
            if isinstance(out, np.ndarray):
                # Common case: single image (H,W,3/4)
                return out
            if isinstance(out, (tuple, list)):
                # Try to find first valid ndarray with 3/4 channels
                for item in out:
                    if isinstance(item, np.ndarray) and item.ndim == 3 and item.shape[2] in (3, 4):
                        return item
                # Some APIs return (ok, img) -> try second element
                if len(out) >= 2 and isinstance(out[1], np.ndarray):
                    return out[1]
                return None
            # Try to coerce other types (PIL, tensor-like)
            try:
                if hasattr(out, "toarray"):
                    return np.asarray(out.toarray())
                if hasattr(out, "numpy"):
                    return np.asarray(out.numpy())
            except Exception:
                return None
            return None

        # Get the configured rendered env indices if available (may be int or list)
        rendered_list = None
        try:
            # scene may hold vis_options with rendered_envs_idx
            vis = getattr(self.scene, "vis_options", None)
            if vis is not None:
                rendered_list = getattr(vis, "rendered_envs_idx", None)
                # If n_rendered_envs (older) exists and is integer, convert to list of first N envs
                if rendered_list is None:
                    n = getattr(vis, "n_rendered_envs", None)
                    if isinstance(n, int):
                        rendered_list = list(range(n))
        except Exception:
            rendered_list = None

        # Try 1: direct call with env_idx
        out = None
        try:
            out = self.cam_0.render(env_idx)
        except Exception as e:
            if not getattr(self, "_warned_render_attempts", False):
                print(f"\033[93mWarning: cam.render(env_idx) raised: {e}\033[0m")
                self._warned_render_attempts = True
            out = None

        img = _extract_image(out)
        # Try 2: if nothing, call camera.render() (may return array for all rendered envs)
        if img is None:
            try:
                out_all = self.cam_0.render()
                # If out_all is an ndarray of shape (N, H, W, C), pick appropriate entry
                if isinstance(out_all, np.ndarray) and out_all.ndim == 4:
                    # Determine index in rendered_list
                    if rendered_list is not None:
                        try:
                            pos = rendered_list.index(env_idx)
                        except ValueError:
                            # env_idx not in list -> fallback to position env_idx if possible
                            pos = env_idx if env_idx < out_all.shape[0] else None
                    else:
                        pos = env_idx if env_idx < out_all.shape[0] else 0
                    if pos is not None and pos < out_all.shape[0]:
                        img = out_all[pos]
                else:
                    # If render() returned a tuple/list for multiple images, try to extract
                    img = _extract_image(out_all)
                    # if img is a multi-frame ndarray, handle above
                    if img is not None and img.ndim == 4:
                        # same indexing logic
                        if rendered_list is not None:
                            try:
                                pos = rendered_list.index(env_idx)
                            except ValueError:
                                pos = env_idx if env_idx < img.shape[0] else 0
                        else:
                            pos = env_idx if env_idx < img.shape[0] else 0
                        img = img[pos]
            except Exception as e:
                # ignore, we'll try other strategies or fallback
                pass

        # Try 3: if still nothing, try mapping env_idx to position in rendered_list and call with that
        if img is None and rendered_list is not None:
            try:
                pos = rendered_list.index(env_idx)
                try:
                    out = self.cam_0.render(pos)
                    img = _extract_image(out)
                except Exception:
                    img = None
            except ValueError:
                img = None

        # Final fallback: if still no valid image, return black frame and warn once
        if img is None:
            if not getattr(self, "_warned_no_image_data", False):
                print(f"\033[93mWarning: camera.render returned no image data for env {env_idx}. Returning black frame.\033[0m")
                self._warned_no_image_data = True
            return _black_frame()

        # Convert floats to uint8 if necessary
        try:
            if img.dtype != np.uint8 and not np.issubdtype(img.dtype, np.integer):
                img = (img * 255).clip(0, 255).astype(np.uint8)
        except Exception:
            try:
                img = img.astype(np.uint8)
            except Exception:
                return _black_frame()

        # If RGBA, take first 3 channels (RGB)
        if img.ndim == 3 and img.shape[2] == 4:
            rgb = img[..., :3]
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = img
        else:
            if not getattr(self, "_warned_unexpected_image_shape", False):
                print(f"\033[93mWarning: unexpected image shape {getattr(img, 'shape', None)} from camera.render; using black frame.\033[0m")
                self._warned_unexpected_image_shape = True
            return _black_frame()

        # Genesis gives RGB, OpenCV expects BGR
        return rgb[..., ::-1].copy()


    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, add_camera=True, vision_debug=False):
        print(f"--> INICIANDO XPos2DCameraEnv | Vision: {add_camera} | Num Envs: {num_envs}")
        
        self.device = gs.device
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_dof = env_cfg["num_dof"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = False
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # --- CONFIGURACIÓN DE ESCENA CRÍTICA ---
        # 1. Creamos el renderizador explícitamente antes
        rasterizer = gs.renderers.Rasterizer()

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            renderer=rasterizer,
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(0.0, 1.25, 0.4),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
                refresh_rate=15,
            ),
            # usar rendered_envs_idx (lista de índices) — n_rendered_envs espera un entero
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(num_envs)),
                show_world_frame=False
            ),
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
        
        # --- EL RESTO IGUAL QUE ANTES ---
        plane: RigidEntity = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        
        self.robot_init_pos = torch.tensor(self.env_cfg["robot_init_pos"], device=self.device)
        self.robot_init_quat = torch.tensor(self.env_cfg["robot_init_quat"], device=self.device)

        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/home/admin/workspace/src/descriptions/juggling_platform_2d_description/urdf/robot.xml",
                pos=self.robot_init_pos.cpu().numpy(),
                quat=self.robot_init_quat.cpu().numpy(),
            ),
        )
        
        self.platform: RigidLink = self.robot.get_link("platform")
        self.ball: RigidLink = self.robot.get_link("ball")

        # --- CÁMARA (OBLIGATORIA) ---
        # Siempre añadimos la cámara si use_vision es True, independientemente del viewer
        self.use_vision = add_camera
        self.vision_debug = vision_debug
        self.cam_res = (320, 240)
        
        if self.use_vision:
            self.cam_0 = self.scene.add_camera(
                res=self.cam_res,
                pos=(0.0, 0.0, 0.7),
                lookat=(0.0, 0.0, 0.0),
                fov=40,
                GUI=False,
            )
            self.detector = BallDetector()

        # Buffers visión
        self.cam_obs_curr = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cam_obs_prev = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
            
        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))

        # Índices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["motor_names"]]
        self.joints_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        self.ball_dof_idx = [self.robot.get_joint("ball_joint_x").idx_local, self.robot.get_joint("ball_joint_z").idx_local]

        # Rewards
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            if name != "fall_event":
                self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales["ball_pos_x"]], device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # Estados físicos
        self.platform_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.platform_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.ball_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.ball_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.ball_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        # Randomization Buffers
        self.ball_init_pos = torch.tensor(self.env_cfg["ball_init_pos"], device=self.device)
        self.ball_init_qpos = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.ball_init_qpos[:, 0] = self.ball_init_pos[0]
        self.ball_init_qpos[:, 1] = self.ball_init_pos[2]
        
        rand_cfg = self.env_cfg.get("randomization", {})
        self.rand_ball_init_x_range = torch.tensor(
            rand_cfg.get("ball_init_x_range", [float(self.ball_init_pos[0]), float(self.ball_init_pos[0])]),
            device=self.device, dtype=gs.tc_float
        )
        self.motor_bias_range = torch.tensor(rand_cfg.get("motor_bias_range", [0.0, 0.0]), device=self.device, dtype=gs.tc_float)
        self.motor_bias = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.ball_mass_range = torch.tensor(rand_cfg.get("ball_mass_range", [0.028, 0.077]), device=self.device, dtype=gs.tc_float)
        self.ball_nominal_mass = 0.027
        self.ball_mass_current = torch.full((self.num_envs,), self.ball_nominal_mass, device=self.device)
        
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

        self.extras = {} 
        self.extras["observations"] = {}

    def _sample_commands(self, envs_idx):
        """
        Resample position command targets for the given environment indices.
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
        # print(f"Actions: {actions}. \n Clipped Target DOF Pos: {target_dof_pos}. \n Clipped Actions: {self.actions}")
        self.robot.control_dofs_position(clipped_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update observation buffers
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

        # --- PROCESAMIENTO DE VISIÓN ---
        # Guardamos la observación anterior antes de actualizar
        self.cam_obs_prev[:] = self.cam_obs_curr[:]
        
        if self.use_vision:
            width, height = self.cam_res
            for env_idx in range(self.num_envs):
                # 1. Obtener imagen
                img_bgr = self._grab_camera_frame(env_idx)
                
                # 2. Detectar pelota
                # Usamos gray_background=False porque solo queremos las coordenadas
                _, circle, _, _ = self.detector.detect(img_bgr, gray_background=False)
                
                if circle is not None:
                    x_px, y_px, r_px = circle
                    
                    # 3. NORMALIZACIÓN DE PÍXELES (Input a la Red)
                    # X: [0, Width] -> [-1, 1]
                    norm_x = (float(x_px) / width) * 2.0 - 1.0
                    
                    # Y: [0, Height] -> [-1, 1] (Recuerda: en imagen Y va hacia abajo)
                    # Si quieres que Z "suba" numéricamente, puedes invertirlo: 1.0 - (...)
                    # Aquí lo dejo estándar de imagen: -1 arriba, 1 abajo.
                    norm_y = (float(y_px) / height) * 2.0 - 1.0 
                    
                    # Radio: [0, Width/2] -> [0, 1] (aprox)
                    norm_r = float(r_px) / (width / 2.0)
                    
                    self.cam_obs_curr[env_idx, 0] = norm_x
                    self.cam_obs_curr[env_idx, 1] = norm_y
                    self.cam_obs_curr[env_idx, 2] = norm_r
                    
                    # Debug visual opcional
                    if self.vision_debug and env_idx == 0:
                        cv.imshow("Vision Debug Env 0", img_bgr)
                        cv.waitKey(1)
                else:
                    # Si se pierde la pelota, enviamos ceros o un valor centinela.
                    # 0.0 está en el centro de la imagen, quizás es confuso.
                    # A veces es mejor repetir el último valor conocido o mandar -10.
                    # Por simplicidad, dejamos el buffer como estaba o ponemos 0.
                    # Aquí reseteo a 0 para indicar "pérdida" si el detector falla.
                    self.cam_obs_curr[env_idx, :] = 0.0

        # resample commands
        # envs_idx = (
        #     (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
        #     .nonzero(as_tuple=False)
        #     .flatten()
        # )

        # if is_train:
        #     self._sample_commands(envs_idx)
        #     random_idxs = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
        #     self._sample_commands(random_idxs)
        
        # print("Current commands:", self.commands.cpu().numpy())
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
        
        self.obs_buf = torch.cat(
            [
                (self.platform_euler[:, 1] * self.obs_scales["platform_pitch"]).unsqueeze(-1), # 1
                (self.platform_ang_vel[:, 1] * self.obs_scales["platform_pitch_vel"]).unsqueeze(-1), # 1
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 6
                self.dof_vel * self.obs_scales["dof_vel"],  # 6
                self.cam_obs_curr, # 3
                self.cam_obs_prev, # 3
                self.commands * self.commands_scale, # 1
                self.actions, # 2
            ],
            axis=-1,
        )
        self.last_actions[:] = self.actions[:]
        
        # # Debug individual observation components
        # obs_0 = (self.platform_euler[:, 1] * self.obs_scales["platform_pitch"]).unsqueeze(-1)
        # obs_1 = (self.platform_ang_vel[:, 1] * self.obs_scales["platform_pitch_vel"]).unsqueeze(-1)
        # obs_2 = (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"]
        # obs_3 = self.dof_vel * self.obs_scales["dof_vel"]
        # obs_4 = (self.ball_pos[:, 0] * self.obs_scales["ball_pos_x"]).unsqueeze(-1)
        # obs_5 = (self.ball_pos[:, 2] * self.obs_scales["ball_pos_z"]).unsqueeze(-1)
        # obs_6 = (self.ball_vel[:, 0] * self.obs_scales["ball_vel_x"]).unsqueeze(-1)
        # obs_7 = (self.ball_vel[:, 2] * self.obs_scales["ball_vel_z"]).unsqueeze(-1)
        # obs_8 = self.commands * self.commands_scale
        # obs_9 = self.actions

        # # Imprimir estadísticas de cada componente
        # print(f"obs_0 (platform_pitch): min={obs_0.min():.4f}, max={obs_0.max():.4f}, nan={torch.isnan(obs_0).any().item()}")
        # print(f"obs_1 (platform_pitch_vel): min={obs_1.min():.4f}, max={obs_1.max():.4f}, nan={torch.isnan(obs_1).any().item()}")
        # print(f"obs_2 (dof_pos): min={obs_2.min():.4f}, max={obs_2.max():.4f}, nan={torch.isnan(obs_2).any().item()}")
        # print(f"obs_3 (dof_vel): min={obs_3.min():.4f}, max={obs_3.max():.4f}, nan={torch.isnan(obs_3).any().item()}")
        # print(f"obs_4 (ball_pos_x): min={obs_4.min():.4f}, max={obs_4.max():.4f}, nan={torch.isnan(obs_4).any().item()}")
        # print(f"obs_5 (ball_pos_z): min={obs_5.min():.4f}, max={obs_5.max():.4f}, nan={torch.isnan(obs_5).any().item()}")
        # print(f"obs_6 (ball_vel_x): min={obs_6.min():.4f}, max={obs_6.max():.4f}, nan={torch.isnan(obs_6).any().item()}")
        # print(f"obs_7 (ball_vel_z): min={obs_7.min():.4f}, max={obs_7.max():.4f}, nan={torch.isnan(obs_7).any().item()}")
        # print(f"obs_8 (commands): min={obs_8.min():.4f}, max={obs_8.max():.4f}, nan={torch.isnan(obs_8).any().item()}")
        # print(f"obs_9 (actions): min={obs_9.min():.4f}, max={obs_9.max():.4f}, nan={torch.isnan(obs_9).any().item()}")
        
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

        if is_train:
            x_low, x_high = self.rand_ball_init_x_range[0].item(), self.rand_ball_init_x_range[1].item()
            x0 = gs_rand_float(x_low, x_high, (len(envs_idx),), self.device)
        else:
            x0 = torch.zeros((len(envs_idx),), device=self.device)
     
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
        # Reset Visión Buffers
        self.cam_obs_curr[envs_idx] = 0.0
        self.cam_obs_prev[envs_idx] = 0.0
        
        if is_train:
            # ------------------------------------------------------------
            # 2. DOMAIN RANDOMIZATION (Estilo Ejemplo Oficial)
            # ------------------------------------------------------------
            n_links = self.robot.n_links
            # Generamos el rango de índices de links con PyTorch
            all_links_idx = torch.arange(n_links, device=self.device)
            # --- MASA ---
            m_low, m_high = self.ball_mass_range[0], self.ball_mass_range[1]
            # Generamos una masa absoluta y calculamos el shift
            new_masses = gs_rand_float(m_low, m_high, (len(envs_idx),), self.device)
            self.ball_mass_current[envs_idx] = new_masses
            
            # El ejemplo oficial usa (n_envs, n_links)
            # Creamos un tensor de ceros para todos los links y solo ponemos el valor en la bola
            mass_shifts = torch.zeros((len(envs_idx), self.robot.n_links), device=self.device)
            # El shift es: masa_deseada - masa_original
            mass_shifts[:, self.ball.idx_local] = self.ball_mass_current[envs_idx] - self.ball_nominal_mass
            
            self.robot.set_mass_shift(
                mass_shift = mass_shifts,
                links_idx_local = all_links_idx, # Aplicamos a todos (aunque la mayoría sean 0 shift)
                envs_idx = envs_idx
            )

            # --- FRICCIÓN ---
            # El ejemplo oficial usa ratio: friction = default * ratio
            # Vamos a randomizar el ratio entre 0.7 y 1.3
            friction_ratios = torch.ones((len(envs_idx), self.robot.n_links), device=self.device)
            friction_ratios[:, self.ball.idx_local] = gs_rand_float(0.7, 1.3, (len(envs_idx),), self.device)
            friction_ratios[:, self.platform.idx_local] = gs_rand_float(0.7, 1.3, (len(envs_idx),), self.device)

            self.robot.set_friction_ratio(
                friction_ratio = friction_ratios,
                links_idx_local = all_links_idx,
                envs_idx = envs_idx
            )
            
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

    def reset(self, is_train=True):
        """
        Reset all environments.
        """
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device), is_train)
        return self.obs_buf, None

    # Reward functions
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_tracking_x(self):
        on_x = torch.abs(self.ball_pos[:, 0]) <= 0.11
        on_z = self.ball_pos[:, 2] >= 0.026
        alive = (on_x & on_z).float()

        ex = self.commands[:, 0] - self.ball_pos[:, 0]
        sigma = float(self.reward_cfg.get("tracking_sigma", 0.06))
        r = torch.exp(-torch.square(ex) / (2 * sigma**2))
        return r * alive

    def _reward_stabilize_x_velocity(self):
        s  = self.reward_cfg["stability_sigma"]
        v0 = 0.0
        vx = torch.abs(self.ball_vel[:, 0])
        reward = torch.exp(-torch.square(vx - v0) / (2 * s**2))
        return reward

    def _success_mask(self):
        x_tol = self.reward_cfg.get("success_x_tol", 0.003)
        vz_tol = self.reward_cfg.get("success_vz_tol", 0.003)
        close_x = torch.abs(self.ball_pos[:, 0] - self.commands[:, 0]) < x_tol
        slow_vz = torch.abs(self.ball_vel[:, 2]) < vz_tol
        return close_x & slow_vz

    def _reward_success_event(self):
        # BONUS ∝ tiempo restante (0..1)
        success = self._success_mask()
        remaining_frac = 1.0 - (self.episode_length_buf.float() / float(self.max_episode_length))
        return success.float() * remaining_frac  # ¡no escalar por dt en los scales!

    def _reward_fall_event(self):
        # Caída por X fuera de la plataforma o Z por debajo del umbral
        fell_x = torch.abs(self.ball_pos[:, 0]) > 0.11   # borde en X
        fell_z = self.ball_pos[:, 2] < 0.026             # umbral en Z

        # Fuera de plataforma o por debajo de Z
        x_half = self.reward_cfg.get("platform_x_half", 0.11)
        z_min  = self.reward_cfg.get("z_min_alive", 0.026)

        fell_x = torch.abs(self.ball_pos[:, 0]) > x_half
        fell_z = torch.abs(self.ball_pos[:, 2]) < z_min
        fell = fell_x | fell_z
        return fell.float()