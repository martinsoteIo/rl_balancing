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
        "num_dof": 15,
        "num_actions": 3,
        "ball_spawn_range": {
            "x": [-0.1, 0.1],
            "y": [-0.1, 0.1],
            "z": [0.18, 0.3],
        },
        # joint/link names
        "default_joint_angles": {  # [rad]
            "first_motor_joint": 0.0,
            "second_motor_joint": 0.0,
            "third_motor_joint": 0.0,
            "first_arm_joint": 0.0,
            "second_arm_joint": 0.0,
            "third_arm_joint": 0.0,
            "first_platform_joint_yaw": 0.0,
            "first_platform_joint_pitch": 0.0,
            "first_platform_joint_roll": 0.0,
            "second_platform_joint_yaw": 0.0,
            "second_platform_joint_pitch": 0.0,
            "second_platform_joint_roll": 0.0,
            "third_platform_joint_yaw": 0.0,
            "third_platform_joint_pitch": 0.0,
            "third_platform_joint_roll": 0.0,
        },
        "default_motor_angles": {  # [rad]
            "first_motor_joint": 0.0,
            "second_motor_joint": 0.0,
            "third_motor_joint": 0.0,
        },
        "joint_names": [
            "first_motor_joint",
            "second_motor_joint",
            "third_motor_joint",
            "first_arm_joint",
            "second_arm_joint",
            "third_arm_joint",
            "first_platform_joint_yaw",
            "first_platform_joint_pitch",
            "first_platform_joint_roll",
            "second_platform_joint_yaw",
            "second_platform_joint_pitch",
            "second_platform_joint_roll",
            "third_platform_joint_yaw",
            "third_platform_joint_pitch",
            "third_platform_joint_roll",
        ],
        "motor_names": [
            "first_motor_joint",
            "second_motor_joint",
            "third_motor_joint",
        ],
        # PD
        "kp": 50.0,
        "kd": 10,
        # termination
        "termination_if_pitch_greater_than": 0.5236,
        "termination_if_roll_greater_than": 0.5236,
        "termination_if_ball_x_greater_than": 0.24,
        "termination_if_ball_y_greater_than": 0.24,
        "termination_if_ball_falls": 0.050,
        # robot pose
        "robot_init_pos": [0.0, 0.0, 0.0],
        "robot_init_quat": [1.0, 0.0, 0.0, 0.0],
        "ball_init_pos": [0.0, 0.0, 0.18],
        "episode_length_s": 10.0,
        "resampling_time_s": 8.0,
        "action_scale": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        
        "randomization": {
            "ball_init_z_range": [0.17, 0.30],
            "ball_init_y_range": [-0.1, 0.1],
            "ball_init_x_range": [-0.1, 0.1],
            "ball_mass_range": [0.001, 0.030],
            "ball_friction_range": [0.1, 0.5],
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
        "num_obs": 45,
        "obs_scales": {
            "platform_pitch": 0.9,
            "platform_roll": 0.9,
            "platform_pitch_vel": 0.05,
            "platform_roll_vel": 0.1,
            "dof_pos": 0.5,
            "dof_vel": 0.02,
            "ball_pos_x": 4.0,
            "ball_pos_y": 4.0,
            "ball_pos_z": 1.0,
            "ball_vel_x": 1.0,
            "ball_vel_y": 1.0,
            "ball_vel_z": 0.1,  
        },
    }

    reward_cfg = {
        "platform_x_half": 0.11,
        "platform_y_half": 0.11,
        "z_min_alive": 0.026,
        "tracking_sigma_x": 0.25,
        "tracking_sigma_y": 0.25,

        "stability_sigma": 0.05,

        "success_x_tol": 0.003,
        "success_y_tol": 0.003,
        "success_vz_tol": 0.003,

        "reward_scales": {
            "similar_to_default": 0.1,
            "tracking_xy": 5.0,
            "stabilize_z_velocity": 1.0,
            "success_event": 10.0,
            "fall_event": -150.0,
        },
    }


    command_cfg = {
        "num_commands": 2,
        "target_x": [-0.08, 0.08],
        "target_y": [-0.08, 0.08],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
