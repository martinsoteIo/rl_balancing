""" Configuration parameters """

def get_env_cfg():
    """
    Returns environment configuration.
    """
    env_cfg = {
        "num_dof": 6,
        "num_actions": 3,
        "default_joint_angles": {  # [rad]
            "first_motor_joint": 0.0,
            "second_motor_joint": 0.0,
            "third_motor_joint": 0.0,
            "first_arm_joint": 0.0,
            "second_arm_joint": 0.0,
            "third_arm_joint": 0.0,
        },
        "default_motor_angles": {  # [rad]
            "first_motor_joint": 0.0,
            "second_motor_joint": 0.0,
            "third_motor_joint": 0.0,
        },
        # joint/link names
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
        # termination
        "termination_if_pitch_greater_than": 10.0,
        "termination_if_roll_greater_than": 10.0,
        "termination_if_ball_x_greater_than": 10.0,
        "termination_if_ball_y_greater_than": 10.0,
        "termination_if_ball_falls": 0.025,
        # robot pose
        "ball_init_pos": [0.0, 0.0, 0.4],
        "episode_length_s": 60.0,
        "action_scale": 0.5,
        "simulate_action_latency": False,
        "motor_ranges": [
            (-1.0, 0.3),
            (-1.0, 0.3),
            (-0.3, 1.0),
        ],
    }

    return env_cfg
