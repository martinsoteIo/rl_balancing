""" Balancing Platform training script """

import os
import argparse
import pickle
import shutil
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

def train(env_class,
          get_cfgs_fn,
          get_train_cfg_fn,
          exp_name: str = "juggling_platform_2d",
          num_envs: int = 4096,
          max_iterations: int = 101,
          resume: bool = True,
          ckpt: int = 100):
    """
    Generic and environment-agnostic policy training method.
    """

    gs.init(precision="64", logging_level="warning")
    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs_fn()
    train_cfg = get_train_cfg_fn(exp_name, max_iterations)
    
    if not resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    env = env_class(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False
    )

    if not resume:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    if resume:
        ckpt_path = os.path.join(log_dir, f"model_{ckpt}.pt")
        print(f"Loading checkpoint from {ckpt_path}...")
        runner.load(ckpt_path)
    
    runner.learn(num_learning_iterations=max_iterations,
                 init_at_random_ep_len=True,
    )


def main():
    """
    Main training script
    """
    parser = argparse.ArgumentParser(description="Train PPO agent on juggling platform environments")

    parser.add_argument(
        "-en", "--env_name",
        type=str,
        default="x_pos",
        choices=["x_pos", "x_genesis_mass_randomization"],
        help="Environment name: 'x_pos' (track x), 'x_genesis_mass_randomization' (track x with Genesis simulator and ball mass randomization)"
    )

    parser.add_argument(
        "-ex", "--exp_name",
        type=str,
        default="x_pos_2d_juggling",
        help="Name of the experiment for logging and checkpointing"
    )

    parser.add_argument(
        "-n", "--num_envs",
        type=int,
        default=2048,
        help="Number of parallel environments to simulate"
    )

    parser.add_argument(
        "-i", "--max_iter",
        type=int,
        default=10000,
        help="Maximum number of PPO learning iterations"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-r", "--resume",
        dest='resume',
        action="store_true",
        help="Resume training from the checkpoint specified by --ckpt"
    )
    group.add_argument(
        "-nr", "--no-resume",
        dest='resume',
        action='store_false',
        help="Start training from scratch (ignoring existing checkpoints)"
    )

    parser.set_defaults(resume=True)

    parser.add_argument(
        "-c", "--ckpt",
        type=int,
        default=100,
        help="Checkpoint number to load when resuming (e.g., 100 loads 'model_100.pt')"
    )

    args = parser.parse_args()

    if args.env_name == "x_pos":
        from rl_policies.juggling_platform_2d.envs.x_pos_env_genesis import XPos2DEnv
        from rl_policies.juggling_platform_2d.configs.x_pos_genesis import get_cfgs, get_train_cfg
        env_class = XPos2DEnv
        get_cfgs_fn = get_cfgs
        get_train_cfg_fn = get_train_cfg
        
    elif args.env_name == "x_genesis_mass_randomization":
        from rl_policies.juggling_platform_2d.envs.x_pos_env_genesis_mass_friction import XPos2DEnv
        from rl_policies.juggling_platform_2d.configs.x_pos_genesis import get_cfgs, get_train_cfg
        env_class = XPos2DEnv
        get_cfgs_fn = get_cfgs
        get_train_cfg_fn = get_train_cfg
    else:
        raise ValueError(f"Unknown environment name: {args.env_name}")

    train(env_class=env_class,
          get_cfgs_fn=get_cfgs_fn,
          get_train_cfg_fn=get_train_cfg_fn,
          exp_name=args.exp_name,
          num_envs=args.num_envs,
          max_iterations=args.max_iter,
          resume=args.resume,
          ckpt=args.ckpt)


if __name__ == "__main__":
    main()
