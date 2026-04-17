import argparse
import os
import shutil
from datetime import datetime
from pprint import pformat

import os.path as osp

parser = argparse.ArgumentParser(description="Train dual-flagella single-agent joint-policy senior policy")
parser.add_argument("--translate_ckpt", type=str, required=True, help="Checkpoint path for the translate (forward) primitive policy")
parser.add_argument("--reorien_ckpt", type=str, required=True, help="Checkpoint path for the reorientation primitive policy")
parser.add_argument("--num_cpus", type=int, default=5, help="Number of CPUs for Ray (default: 5)")
parser.add_argument("--num_threads", type=int, default=5, help="Number of PyTorch threads (default: 5)")
args = parser.parse_args()
os.environ["STOKES_NUM_THREADS"] = str(args.num_threads)

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print

import swimmer as swimmer_module
from swimmer import NUM_STRATEGIES, STRATEGY_NAMES, ROBOT_IDS, swimmer_gym


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
POLICY_DIR = os.path.join(os.getcwd(), f"single_policy_{TIMESTAMP}")
TENSORBOARD_DIR = os.path.join(POLICY_DIR, "tensorboard")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_VISUALIZER = os.path.join(CURRENT_DIR, "visualize_dual_flagella.py")
CURRENT_SWIMMER = os.path.join(CURRENT_DIR, "swimmer.py")
class TrainingMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        sub_envs = base_env.get_sub_environments()
        if not sub_envs:
            return

        env_ref = sub_envs[env_index]
        episode.custom_metrics["robot_1_reward"] = float(getattr(env_ref, "last_robot_rewards", [0.0, 0.0])[0])
        episode.custom_metrics["robot_2_reward"] = float(getattr(env_ref, "last_robot_rewards", [0.0, 0.0])[1])
        episode.custom_metrics["robot_1_concentration"] = float(getattr(env_ref, "con1", 0.0))
        episode.custom_metrics["robot_2_concentration"] = float(getattr(env_ref, "con2", 0.0))
        episode.custom_metrics["robot_1_order"] = float(getattr(env_ref, "last_robot_orders", [1, 1])[0])
        episode.custom_metrics["robot_2_order"] = float(getattr(env_ref, "last_robot_orders", [1, 1])[1])
        episode.custom_metrics["robot_1_centroid_x"] = float(getattr(env_ref, "last_centroid1", [0.0, 0.0])[0])
        episode.custom_metrics["robot_1_centroid_y"] = float(getattr(env_ref, "last_centroid1", [0.0, 0.0])[1])
        episode.custom_metrics["robot_2_centroid_x"] = float(getattr(env_ref, "last_centroid2", [0.0, 0.0])[0])
        episode.custom_metrics["robot_2_centroid_y"] = float(getattr(env_ref, "last_centroid2", [0.0, 0.0])[1])
        episode.custom_metrics["episode_steps"] = float(getattr(env_ref, "ep_step", 0))


def maybe_add_scalar(writer, tag, value, step):
    if isinstance(value, bool):
        writer.add_scalar(tag, int(value), step)
        return
    if isinstance(value, (int, float)):
        writer.add_scalar(tag, value, step)
        return
    if hasattr(value, "item"):
        try:
            writer.add_scalar(tag, float(value.item()), step)
        except Exception:
            return


def write_training_scalars(writer, result, iteration):
    maybe_add_scalar(writer, "training/episode_reward_mean", result.get("episode_reward_mean"), iteration)
    maybe_add_scalar(writer, "training/episode_reward_min", result.get("episode_reward_min"), iteration)
    maybe_add_scalar(writer, "training/episode_reward_max", result.get("episode_reward_max"), iteration)
    maybe_add_scalar(writer, "training/episodes_total", result.get("episodes_total"), iteration)
    maybe_add_scalar(writer, "training/num_env_steps_sampled", result.get("num_env_steps_sampled"), iteration)
    maybe_add_scalar(writer, "training/num_env_steps_trained", result.get("num_env_steps_trained"), iteration)
    maybe_add_scalar(writer, "training/num_agent_steps_sampled", result.get("num_agent_steps_sampled"), iteration)
    maybe_add_scalar(writer, "training/num_agent_steps_trained", result.get("num_agent_steps_trained"), iteration)
    maybe_add_scalar(writer, "training/sampler_results/episode_len_mean", result.get("sampler_results", {}).get("episode_len_mean"), iteration)

    learner_info = result.get("info", {}).get("learner", {}).get("default_policy", {})
    for key in ("learner_stats", "stats"):
        stats = learner_info.get(key, {})
        if not isinstance(stats, dict):
            continue
        for name, value in stats.items():
            maybe_add_scalar(writer, f"learner/{name}", value, iteration)

    custom_metrics = result.get("custom_metrics", {})
    if isinstance(custom_metrics, dict):
        for name, value in custom_metrics.items():
            maybe_add_scalar(writer, f"custom_metrics/{name}", value, iteration)


def create_summary_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging requires the 'tensorboard' package. "
            "Install it with 'pip install tensorboard' before training."
        ) from exc
    return SummaryWriter(log_dir=log_dir)


def build_env_config(cli_args, skip_policy_load=False):
    return {
        "translate_ckpt": cli_args.translate_ckpt,
        "reorien_ckpt": cli_args.reorien_ckpt,
        "low_level_hold_steps": swimmer_module.LOW_LEVEL_HOLD_STEPS,
        "macro_horizon": swimmer_module.MACRO_HORIZON,
        "skip_policy_load": skip_policy_load,
    }


def build_ppo_config(cli_args):
    env_stub = swimmer_gym(build_env_config(cli_args, skip_policy_load=True))

    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = swimmer_gym
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["num_rollout_workers"] = 0
    config["framework"] = "torch"
    config["env_config"] = build_env_config(cli_args)

    config["gamma"] = 0.9999
    config["lr"] = 0.0005
    config["horizon"] = swimmer_module.MACRO_HORIZON
    config["rollout_fragment_length"] = swimmer_module.MACRO_HORIZON
    config["evaluation_duration"] = 1000000000

    config["lr_schedule"] = None
    config["use_critic"] = True
    config["use_gae"] = True
    config["lambda_"] = 0.95
    config["kl_coeff"] = 0.2
    config["sgd_minibatch_size"] = 20
    config["train_batch_size"] = 100
    config["num_sgd_iter"] = 5
    config["shuffle_sequences"] = True
    config["vf_loss_coeff"] = 1.0
    config["entropy_coeff"] = 0.0
    config["entropy_coeff_schedule"] = None
    config["clip_param"] = 0.2
    config["vf_clip_param"] = 100000
    config["grad_clip"] = None
    config["kl_target"] = 0.01
    config["evaluation_interval"] = 1000000
    config["evaluation_duration"] = 1
    config["min_sample_timesteps_per_iteration"] = 100
    config["callbacks"] = TrainingMetricsCallback
    config["disable_env_checking"] = False
    return config


def write_training_run_markdown(run_dir, cli_args, trainer_config, visualizer_snapshot_path, swimmer_snapshot_path):
    env_params = {
        "robot1_init": swimmer_module.ROBOT1_INIT,
        "robot2_init": swimmer_module.ROBOT2_INIT,
        "macro_horizon": swimmer_module.MACRO_HORIZON,
        "low_level_hold_steps": swimmer_module.LOW_LEVEL_HOLD_STEPS,
        "ccenter_x": swimmer_module.CCENTER_X,
        "ccenter_y": swimmer_module.CCENTER_Y,
        "con_reward_scale": swimmer_module.CON_REWARD_SCALE,
        "visualizer_snapshot": visualizer_snapshot_path,
        "swimmer_snapshot": swimmer_snapshot_path,
        "robot_ids": ROBOT_IDS,
        "policy_mode": "single-agent joint-policy (default_policy)",
        "observation_space": "Box(12,) — concatenated one-hot of both robots' aprm",
        "action_space": "Discrete(9) — joint action (action//3 for R1, action%3 for R2)",
        "num_strategies": swimmer_module.NUM_STRATEGIES,
        "strategy_names": STRATEGY_NAMES,
        "task_mode": "chemotaxis via concentration gradient",
        "reward_mode": "per-robot concentration increase",
        "reset_behavior": "hard reset to fixed start poses each episode",
    }

    lines = [
        "# Single-Agent Joint-Policy Training Run Parameters",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Policy directory: `{run_dir}`",
        "",
        "## CLI Parameters",
        "",
        f"- `translate_ckpt`: `{cli_args.translate_ckpt}`",
        f"- `reorien_ckpt`: `{cli_args.reorien_ckpt}`",
        f"- `num_cpus`: `{cli_args.num_cpus}`",
        f"- `num_threads`: `{cli_args.num_threads}`",
        "",
        "## PPO / RLlib Parameters",
        "",
        "```python",
        pformat(trainer_config, sort_dicts=True),
        "```",
        "",
        "## Environment Parameters",
        "",
        "```python",
        pformat(env_params, sort_dicts=True),
        "```",
    ]

    with open(os.path.join(run_dir, "TRAINING_PARAMS.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def snapshot_current_visualizer(run_dir):
    destination = os.path.join(run_dir, "visualize_dual_flagella.py")
    shutil.copy2(CURRENT_VISUALIZER, destination)
    return destination


def snapshot_current_swimmer(run_dir):
    destination = os.path.join(run_dir, "swimmer.py")
    shutil.copy2(CURRENT_SWIMMER, destination)
    return destination


def main():
    os.makedirs(POLICY_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    visualizer_snapshot_path = snapshot_current_visualizer(POLICY_DIR)
    swimmer_snapshot_path = snapshot_current_swimmer(POLICY_DIR)

    print(f"Policy save dir: {POLICY_DIR}")
    print(f"TensorBoard log dir: {TENSORBOARD_DIR}")
    print(f"Visualizer snapshot: {visualizer_snapshot_path}")
    print(f"Swimmer snapshot: {swimmer_snapshot_path}")
    print(f"Ray CPUs: {args.num_cpus}, PyTorch threads: {args.num_threads}")
    print(f"Translate primitive: {args.translate_ckpt}")
    print(f"Reorien primitive: {args.reorien_ckpt}")
    print(os.getcwd())

    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)

    config = build_ppo_config(args)
    write_training_run_markdown(POLICY_DIR, args, config, visualizer_snapshot_path, swimmer_snapshot_path)

    trainer = ppo.PPO(config=config, env=swimmer_gym)
    tb_writer = create_summary_writer(TENSORBOARD_DIR)

    now_path = os.getcwd()
    for sub_dir in ("traj", "traj2", "trajp"):
        os.makedirs(os.path.join(now_path, sub_dir), exist_ok=True)

    tb_write_interval = 6

    for i in range(2000):
        print(i)
        result = trainer.train()
        if i % tb_write_interval == 0:
            write_training_scalars(tb_writer, result, i)
            tb_writer.flush()
        print(pretty_print(result))
        if i % 3 == 0:
            ckpt_dir = osp.join(POLICY_DIR, str(i))
            os.makedirs(ckpt_dir, exist_ok=True)
            trainer.save(ckpt_dir)

    tb_writer.close()


if __name__ == "__main__":
    main()
