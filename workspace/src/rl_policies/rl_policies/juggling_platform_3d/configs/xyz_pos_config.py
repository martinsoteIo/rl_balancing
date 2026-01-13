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
            "learning_rate": 3e-4,
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
        "num_steps_per_env": 48,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    """ Returns environment, observation, reward and command configurations """
    env_cfg = {
        "num_dof": 6,
        "num_actions": 3,
        "ball_spawn_range": {
            "x": [-0.1, 0.1],
            "y": [-0.1, 0.1],
            "z": [0.3, 0.5],
        },
        # joint/link names
        "default_joint_angles": {  # [rad]
            "first_motor_joint": -0.799,
            "second_motor_joint": -0.756,
            "third_motor_joint": -0.756,
            "first_arm_joint": 1.58,
            "second_arm_joint": 1.51,
            "third_arm_joint": 1.51,
        },
        "default_motor_angles": {  # [rad]
            "first_motor_joint": -0.799,
            "second_motor_joint": -0.756,
            "third_motor_joint": -0.756,
        },
        "joint_names": [
            "first_motor_joint",
            "second_motor_joint",
            "third_motor_joint",
            "first_arm_joint",
            "second_arm_joint",
            "third_arm_joint",
        ],
        "motor_names": [
            "first_motor_joint",
            "second_motor_joint",
            "third_motor_joint",
        ],
        # PD
        "kp": 50.0,
        "kd": 2.5,
        # termination
        "termination_if_pitch_greater_than": 0.523599,
        "termination_if_roll_greater_than": 0.523599,
        "termination_if_ball_x_greater_than": 0.10,
        "termination_if_ball_y_greater_than": 0.10,
        "termination_if_ball_too_high": 0.7,
        "termination_if_ball_falls": 0.025,
        # robot pose
        "robot_init_pos": [0.0, 0.0, 0.0],
        "robot_init_quat": [1.0, 0.0, 0.0, 0.0],
        "ball_init_pos": [0.0, 0.0, 0.2],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
    }

    obs_cfg = {
        "num_obs": 28,
        "obs_scales": {
            "platform_pitch": 0.9,
            "platform_roll": 0.9,
            "platform_pitch_vel": 0.05,
            "platform_roll_vel": 0.1,
            "dof_pos": 0.5,
            "dof_vel": 0.02,
            "ball_pos_x": 5,
            "ball_pos_y": 5,
            "ball_pos_z": 1.0,
            "ball_vel_x": 1.0,
            "ball_vel_y": 1.0,
            "ball_vel_z": 0.4,  
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "reward_scales": {
            # "action_rate": -0.005,
            # "similar_to_default": -0.01,
            "tracking_xy": 2.0,
            "tracking_z": 1.0,
            "apex_height": 2.0,
            # "ball_lin_vel_z": -1.0,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "target_x": [-0.095, 0.095],
        "target_y": [-0.095, 0.095],
        "target_z": [0.24, 0.28],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
