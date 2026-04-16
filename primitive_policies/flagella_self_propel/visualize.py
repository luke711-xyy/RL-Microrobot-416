import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Primitive Policy Visualizer")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path. If omitted, auto-detect the latest.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Total visualization steps (default: 2000)")
    parser.add_argument("--speed", type=float, default=0.001,
                        help="Refresh interval in seconds (default: 0.001)")
    parser.add_argument("--num_cpus", type=int, default=1,
                        help="Number of CPUs for Ray (default: 1)")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of PyTorch threads (default: 1)")
    parser.add_argument("--view_range", type=float, default=4.0,
                        help="Camera follow half-width (default: 4.0)")
    parser.add_argument("--order", type=int, default=None,
                        help="Override env order: 0=symmetric, 1=turn, -1=mirror turn")
    return parser.parse_args()


ARGS = parse_args()
os.environ["STOKES_NUM_THREADS"] = str(ARGS.num_threads)

os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib

if sys.platform == "darwin":
    matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo

from swimmer import swimmer_gym


# ================= Checkpoint utilities =================

def is_checkpoint_path(path):
    path = Path(path)
    if not path.exists():
        return False
    if path.is_file():
        return path.name.startswith("checkpoint-")
    return (
        path.name.startswith("checkpoint_")
        or (path / "rllib_checkpoint.json").exists()
        or (path / ".is_checkpoint").exists()
    )


def checkpoint_sort_key(path):
    path = Path(path)
    digits = "".join(ch for ch in path.name if ch.isdigit())
    order = int(digits) if digits else -1
    return (order, str(path))


def find_latest_checkpoint(base_dir=None):
    base_dir = Path(base_dir or BASE_DIR)
    if not base_dir.exists():
        return None

    # Support both old "policy" / "policy_*" and new "{timestamp}_policy_{id}" naming
    policy_roots = [
        p for p in base_dir.iterdir()
        if p.is_dir() and ("_policy_" in p.name or p.name.startswith("policy"))
    ]
    if not policy_roots:
        return None

    latest_policy = max(policy_roots, key=lambda p: p.stat().st_mtime)
    iter_dirs = [p for p in latest_policy.iterdir() if p.is_dir() and p.name.isdigit()]
    if not iter_dirs:
        return None

    latest_iter_dir = max(iter_dirs, key=lambda p: int(p.name))
    checkpoint_paths = sorted(
        [c for c in latest_iter_dir.rglob("*") if is_checkpoint_path(c)],
        key=checkpoint_sort_key,
    )
    return checkpoint_paths[-1] if checkpoint_paths else None


def resolve_checkpoint(path_str):
    cp_path = Path(path_str).expanduser().resolve()
    if cp_path.is_file():
        return cp_path
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {cp_path}")
    if is_checkpoint_path(cp_path):
        return cp_path

    direct = sorted(
        [c for c in cp_path.iterdir() if is_checkpoint_path(c)],
        key=checkpoint_sort_key,
    )
    if direct:
        return direct[-1]

    nested = sorted(
        [c for c in cp_path.rglob("*") if is_checkpoint_path(c)],
        key=checkpoint_sort_key,
    )
    if nested:
        return nested[-1]
    raise FileNotFoundError(f"No checkpoint found under: {cp_path}")


# ================= Helper functions =================

def unpack_action_output(action_output, prev_state):
    if not isinstance(action_output, tuple):
        return action_output, prev_state
    if len(action_output) == 0:
        raise ValueError("compute_single_action returned an empty tuple")
    action = action_output[0]
    next_state = prev_state
    if len(action_output) >= 2 and isinstance(action_output[1], (list, tuple)):
        next_state = action_output[1]
    return action, next_state


def compute_average_heading(state_array):
    """Compute average heading angle across all links."""
    head_omega = state_array[2]
    running_angle = head_omega
    angle_sum = head_omega
    for beta in state_array[3:]:
        running_angle += beta
        angle_sum += running_angle
    return angle_sum / (len(state_array) - 2)


def compute_true_centroid(robot_shape):
    return np.mean(robot_shape[:, 0]), np.mean(robot_shape[:, 1])


def apply_order_override(env, order_value):
    """Set env.order and regenerate observation accordingly."""
    env.order = order_value
    if env.order >= 0:
        return env.state[3:]
    else:
        return -env.state[3:]


# ================= Config (must match train.py) =================

def get_config():
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["num_rollout_workers"] = 0
    config["framework"] = "torch"
    config["gamma"] = 0.9999
    config["lr"] = 0.0003
    config["horizon"] = 1000
    config["evaluation_duration"] = 10000000
    config["lr_schedule"] = None
    config["use_critic"] = True
    config["use_gae"] = True
    config["lambda_"] = 0.95
    config["kl_coeff"] = 0.2
    config["sgd_minibatch_size"] = 64
    config["train_batch_size"] = 1000
    config["num_sgd_iter"] = 30
    config["shuffle_sequences"] = True
    config["vf_loss_coeff"] = 1.0
    config["entropy_coeff"] = 0.0
    config["entropy_coeff_schedule"] = None
    config["clip_param"] = 0.1
    config["vf_clip_param"] = 100000
    config["grad_clip"] = None
    config["kl_target"] = 0.01
    config["evaluation_interval"] = 1000000
    config["evaluation_duration"] = 1
    config["use_lstm"] = True
    config["max_seq_len"] = 20
    config["min_sample_timesteps_per_iteration"] = 1000
    config["env"] = swimmer_gym
    return config


# ================= Main =================

def main():
    args = ARGS

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus, log_to_driver=False)

    env = swimmer_gym({})
    obs = env.reset()

    # Override order if specified via CLI
    if args.order is not None:
        obs = apply_order_override(env, args.order)

    config = get_config()
    agent = ppo.PPO(config=config, env=swimmer_gym)

    # Load checkpoint
    if args.checkpoint:
        cp_path = resolve_checkpoint(args.checkpoint)
    else:
        cp_path = find_latest_checkpoint()
        if cp_path is None:
            print("\n[Error] No checkpoint found. Run train.py first or pass --checkpoint.")
            sys.exit(1)

    print(f"Loading checkpoint: {cp_path}")
    print(f"Order: {env.order} | Ray CPUs: {args.num_cpus} | PyTorch threads: {args.num_threads}")
    try:
        agent.restore(str(cp_path))
        print(">>> Checkpoint loaded. Launching visualization...")
    except Exception as e:
        print(f"\n[Error] Failed to restore checkpoint: {e}")
        sys.exit(1)

    # --- Plotting setup ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    order_names = {0: "symmetric", 1: "turn", -1: "mirror turn"}
    ax.set_title(f"Primitive Policy  |  order={env.order} ({order_names.get(env.order, '?')})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    (line,) = ax.plot([], [], "-", lw=2, color="royalblue", label="Robot")
    (trace,) = ax.plot([], [], "-", lw=1, color="crimson", alpha=0.5, label="Trace")
    (avg_line,) = ax.plot([], [], "--", lw=2, color="green", alpha=0.8, label="Avg Heading")
    (centroid_dot,) = ax.plot([], [], "o", color="black", markersize=5, label="Centroid")
    info_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), va="top",
    )
    plt.legend(loc="upper right")
    plt.show(block=False)

    # --- LSTM state ---
    policy = agent.get_policy()
    lstm_state = policy.get_initial_state()

    history_x, history_y = [], []
    total_reward = 0.0
    heading_len = 2.0

    # --- Initial frame ---
    robot_shape = env.XY_positions.copy()
    cx, cy = compute_true_centroid(robot_shape)
    avg_heading = compute_average_heading(env.state)
    history_x.append(cx)
    history_y.append(cy)
    line.set_data(robot_shape[:, 0], robot_shape[:, 1])
    trace.set_data(history_x, history_y)
    centroid_dot.set_data([cx], [cy])
    avg_line.set_data(
        [cx, cx + heading_len * np.cos(avg_heading)],
        [cy, cy + heading_len * np.sin(avg_heading)],
    )
    ax.set_xlim(cx - args.view_range, cx + args.view_range)
    ax.set_ylim(cy - args.view_range, cy + args.view_range)
    info_text.set_text(f"Step: 0\nX: {cx:.2f}\nY: {cy:.2f}\nReward: 0.00\nOrder: {env.order}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)

    # --- Console header ---
    print("-" * 80)
    print(f"{'Step':<8} | {'X':<10} | {'Y':<10} | {'Reward':<10} | {'StepRwd':<10} | {'Order':<6}")
    print("-" * 80)

    # --- Simulation loop ---
    try:
        for i in range(args.steps):
            if not plt.fignum_exists(fig.number):
                print("\nWindow closed.")
                break

            plt.pause(0.001)

            # Compute action
            action_output = agent.compute_single_action(
                observation=obs, state=lstm_state, explore=False,
            )
            action, lstm_state = unpack_action_output(action_output, lstm_state)

            # Step
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Read robot shape
            robot_shape = env.XY_positions.copy()
            cx, cy = compute_true_centroid(robot_shape)
            avg_heading = compute_average_heading(env.state)

            history_x.append(cx)
            history_y.append(cy)

            # Update plot
            line.set_data(robot_shape[:, 0], robot_shape[:, 1])
            trace_len = 1000
            if len(history_x) > trace_len:
                trace.set_data(history_x[-trace_len:], history_y[-trace_len:])
            else:
                trace.set_data(history_x, history_y)
            centroid_dot.set_data([cx], [cy])
            avg_line.set_data(
                [cx, cx + heading_len * np.cos(avg_heading)],
                [cy, cy + heading_len * np.sin(avg_heading)],
            )

            ax.set_xlim(cx - args.view_range, cx + args.view_range)
            ax.set_ylim(cy - args.view_range, cy + args.view_range)

            info_text.set_text(
                f"Step: {i + 1}\nX: {cx:.2f}\nY: {cy:.2f}\n"
                f"Reward: {total_reward:.2f}\nStep Rwd: {reward:.3f}\nOrder: {env.order}"
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Console output
            print(
                f"{i + 1:<8} | {cx:<10.4f} | {cy:<10.4f} | "
                f"{total_reward:<10.4f} | {reward:<10.4f} | {env.order:<6}"
            )

            # Handle episode end
            if done:
                obs = env.reset()
                if args.order is not None:
                    obs = apply_order_override(env, args.order)
                lstm_state = policy.get_initial_state()

            if args.speed > 0:
                plt.pause(args.speed)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print("-" * 80)
    print("Simulation finished. Close the window to exit.")
    plt.ioff()
    plt.show()

    agent.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
