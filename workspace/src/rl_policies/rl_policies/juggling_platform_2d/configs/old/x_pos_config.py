""" Configuration parameters """


def get_train_cfg(exp_name, max_iterations):
    """ 
    Returns training configuration
    """

    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    """ Returns environment, observation, reward and command configurations """
    env_cfg = {
        "num_dof": 6,
        "num_actions": 2,
        "ball_spawn_range": {
            "x": [-0.1, 0.1],
            "z": [0.3, 0.5],
        },
        # joint/link names
        "default_joint_angles": {  # [rad]
            "left_motor_joint": -1.047,
            "right_motor_joint": -1.047,
            "left_arm_joint": 2.094,
            "right_arm_joint": -2.094,
            "left_platform_joint": -1.047,
            "right_platform_joint": 1.047,
        },
        "default_motor_angles": {  # [rad]
            "left_motor_joint": -1.047,
            "right_motor_joint": -1.047,
        },
        "joint_names": [
            "left_motor_joint",
            "right_motor_joint",
            "left_arm_joint",
            "right_arm_joint",
            "left_platform_joint",
            "right_platform_joint",
        ],
        "motor_names": [
            "left_motor_joint",
            "right_motor_joint",
        ],
        # PD
        "kp": 100.0,
        "kd": 0.5,
        # termination
        "termination_if_pitch_greater_than": 0.5236,
        "termination_if_ball_x_greater_than": 0.12,
        "termination_if_ball_falls": 0.025,
        # robot pose
        "robot_init_pos": [0.0, 0.0, 0.0],
        "robot_init_quat": [1.0, 0.0, 0.0, 0.0],
        "ball_init_pos": [0.0, 0.0, 0.2],
        "episode_length_s": 20.0,
        "resampling_time_s": 8.0,
        "action_scale": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        
        "randomization": {
            "ball_init_z_range": [0.14, 0.40],
            "ball_init_x_range": [-0.09, 0.09],
            "action_latency_prob": 0.20,
            "motor_bias_range": [-0.02, 0.02],
            "obs_noise_std": {
                "platform_pitch": 0.002,
                "platform_pitch_vel": 0.01,
                "dof_pos": 0.001,
                "dof_vel": 0.01,
                "ball_pos_x": 0.001,
                "ball_pos_z": 0.001,
                "ball_vel_x": 0.01,
                "ball_vel_z": 0.01,
                "actions": 0.0,
                "command": 0.0
            }
    }
    }

    obs_cfg = {
        "num_obs": 21,
        "obs_scales": {
            "platform_pitch": 2.0,
            "platform_pitch_vel": 0.15,
            "dof_pos": 0.55,
            "dof_vel": 0.04,
            "ball_pos_x": 10.0,
            "ball_pos_z": 0.7,
            "ball_vel_x": 1.5,
            "ball_vel_z": 0.4,  
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.02,        # antes 0.001  -> mucho más ancho
        "stability_sigma": 0.15,       # antes 0.08   -> más tolerancia
        "success_x_tol": 0.01,         # tolerancias más realistas para fase 1
        "success_vz_tol": 0.05,
        "desired_rest_z": 0.030,
        "reward_scales": {
            # "action_rate": -0.005,
            # "similar_to_default": -0.01,
            "tracking_x": 4.0,
            "stabilize_z_velocity": 0.8, #-200.0
            # "platform_quiet_near": -10.0,
            "success_event": 20.0,
            "fall_event": -150.0,
        },
    }
    # reward_cfg = {
    #     "tracking_sigma": 0.010,
    #     "stability_sigma": 0.12,
    #     "desired_rest_z": 0.030,
    #     "dampen_pos_weight": 2.0,
    #     "success_x_tol": 0.008,
    #     "success_z_tol": 0.006,
    #     "success_vz_tol": 0.04,
    #     "reward_scales": {
    #             "dampen_z": 150.0,
    #             "transport_progress": 100.0,
    #             "settle": 120.0,
    #             "stabilize_platform": 8.0,
    #             "alive": 0.5,               # con dt
    #             "clip_penalty": 20.0,
    #             "early_capture": 40.0,      # sin dt
    #             "success_event": 120.0,     # sin dt
    #             "fall_event": -300.0        # sin dt
    #     },
    # }

    command_cfg = {
        "num_commands": 1,
        "target_x": [-0.08, 0.08],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
