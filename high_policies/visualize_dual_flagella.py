import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread


BASE_DIR = Path(__file__).resolve().parent
SHARED_POLICY_ID = "shared_policy"


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return SHARED_POLICY_ID


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Flagella Shared-Policy Visualizer")
    parser.add_argument("--translate_ckpt", type=str, required=True, help="Checkpoint path for the translate primitive policy")
    parser.add_argument("--reorien_ckpt", type=str, required=True, help="Checkpoint path for the reorientation primitive policy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Senior-policy checkpoint path")
    parser.add_argument("--steps", type=int, default=200, help="Total macro steps to visualize (default: 200)")
    parser.add_argument("--speed", type=float, default=0.01, help="Refresh interval per displayed frame in seconds (default: 0.01)")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs used by Ray (default: 1)")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of PyTorch threads used by the solver (default: 1)")
    parser.add_argument("--view_range", type=float, default=5.0, help="Half-width of the camera-follow window (default: 5.0)")
    parser.add_argument(
        "--reset_free_playback",
        action="store_true",
        help="Continue visualization across episode boundaries without resetting robot geometry",
    )
    parser.add_argument(
        "--prefetch_queue_size",
        type=int,
        default=10,
        help="Maximum number of precomputed macro-step packages kept in the queue (default: 10)",
    )
    parser.add_argument(
        "--prefetch_warmup_steps",
        type=int,
        default=10,
        help="Number of macro-step packages to preload before playback starts (default: 10)",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Enable exploration noise (see strategy switching even with under-trained policies)",
    )
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

from swimmer import (
    CCENTER_X,
    CCENTER_Y,
    LOW_LEVEL_HOLD_STEPS,
    MACRO_HORIZON,
    NUM_STRATEGIES,
    STRATEGY_NAMES,
    ROBOT_IDS,
    compute_average_heading,
    compute_true_centroid,
    swimmer_gym,
)


def build_env_config(cli_args):
    return {
        "translate_ckpt": cli_args.translate_ckpt,
        "reorien_ckpt": cli_args.reorien_ckpt,
        "low_level_hold_steps": LOW_LEVEL_HOLD_STEPS,
        "macro_horizon": MACRO_HORIZON,
    }


def is_checkpoint_path(path_obj):
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return False
    if path_obj.is_file():
        return path_obj.name.startswith("checkpoint-")
    return (
        path_obj.name.startswith("checkpoint_")
        or (path_obj / "rllib_checkpoint.json").exists()
        or (path_obj / ".is_checkpoint").exists()
    )


def checkpoint_sort_key(path_obj):
    path_obj = Path(path_obj)
    digits = "".join(ch for ch in path_obj.name if ch.isdigit())
    order = int(digits) if digits else -1
    return (order, str(path_obj))


def find_latest_checkpoint(base_dir=None):
    base_dir = Path(base_dir or BASE_DIR)
    policy_roots = [item for item in base_dir.iterdir() if item.is_dir() and item.name.startswith("policy_")]
    if not policy_roots:
        return None
    latest_policy = max(policy_roots, key=lambda item: item.stat().st_mtime)
    iter_dirs = [item for item in latest_policy.iterdir() if item.is_dir() and item.name.isdigit()]
    if not iter_dirs:
        return None
    latest_iter = max(iter_dirs, key=lambda item: int(item.name))
    candidates = sorted([item for item in latest_iter.rglob("*") if is_checkpoint_path(item)], key=checkpoint_sort_key)
    return candidates[-1] if candidates else None


def resolve_checkpoint(path_str):
    cp_path = Path(path_str).expanduser().resolve()
    if cp_path.is_file():
        return cp_path
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {cp_path}")
    if is_checkpoint_path(cp_path):
        return cp_path

    direct_candidates = sorted(
        [candidate for candidate in cp_path.iterdir() if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if direct_candidates:
        return direct_candidates[-1]

    nested_candidates = sorted(
        [candidate for candidate in cp_path.rglob("*") if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if nested_candidates:
        return nested_candidates[-1]

    raise FileNotFoundError(f"No checkpoint found under: {cp_path}")


def build_config():
    env_stub = swimmer_gym({**build_env_config(ARGS), "skip_policy_load": True})

    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = swimmer_gym
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["num_rollout_workers"] = 0
    config["framework"] = "torch"
    config["env_config"] = build_env_config(ARGS)
    config["gamma"] = 0.995
    config["lr"] = 0.0003
    config["horizon"] = MACRO_HORIZON
    config["rollout_fragment_length"] = MACRO_HORIZON
    config["evaluation_duration"] = 10000000
    config["lr_schedule"] = None
    config["use_critic"] = True
    config["use_gae"] = True
    config["lambda_"] = 0.95
    config["kl_coeff"] = 0.2
    config["sgd_minibatch_size"] = 100
    config["train_batch_size"] = 500
    config["num_sgd_iter"] = 10
    config["shuffle_sequences"] = True
    config["vf_loss_coeff"] = 1.0
    config["entropy_coeff"] = 0.001
    config["entropy_coeff_schedule"] = None
    config["clip_param"] = 0.1
    config["vf_clip_param"] = 100000
    config["grad_clip"] = None
    config["kl_target"] = 0.01
    config["evaluation_interval"] = 1000000
    config["evaluation_duration"] = 1
    config["min_sample_timesteps_per_iteration"] = 500
    config["multiagent"] = {
        "policies": {
            SHARED_POLICY_ID: (
                None,
                env_stub.observation_space,
                env_stub.action_space,
                {},
            )
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": [SHARED_POLICY_ID],
        "count_steps_by": "env_steps",
    }
    return config


def unpack_action_output(action_output):
    if not isinstance(action_output, tuple):
        return action_output
    if len(action_output) == 0:
        raise ValueError("compute_single_action returned an empty tuple")
    return action_output[0]


def draw_heading(ax, centroid, heading_angle, color):
    length = 0.6
    dx = length * np.cos(heading_angle)
    dy = length * np.sin(heading_angle)
    ax.plot(
        [centroid[0], centroid[0] + dx],
        [centroid[1], centroid[1] + dy],
        color=color,
        linewidth=2.0,
    )


def capture_env_frame(env, substep_index):
    centroid1 = compute_true_centroid(env.XY_positions1)
    centroid2 = compute_true_centroid(env.XY_positions2)
    return {
        "substep_index": int(substep_index),
        "xy1": np.array(env.XY_positions1, copy=True),
        "xy2": np.array(env.XY_positions2, copy=True),
        "state1": np.array(env.state1, copy=True),
        "state2": np.array(env.state2, copy=True),
        "centroid1": np.array(centroid1, copy=True),
        "centroid2": np.array(centroid2, copy=True),
    }


def render_frame(
    ax,
    frame,
    trace1,
    trace2,
    macro_index,
    strategy_pair,
    total_substeps,
    robot_rewards,
    robot_concentrations,
    robot_orders,
    queue_fill,
    queue_capacity,
    append_trace=True,
):
    centroid1 = np.array(frame["centroid1"], copy=True)
    centroid2 = np.array(frame["centroid2"], copy=True)
    if append_trace:
        trace1.append(centroid1)
        trace2.append(centroid2)

    ax.clear()

    # 浓度场等高线（同心圆热力图）
    view_cx = 0.5 * (centroid1[0] + centroid2[0])
    view_cy = 0.5 * (centroid1[1] + centroid2[1])
    vr = ARGS.view_range
    grid_x = np.linspace(view_cx - vr, view_cx + vr, 120)
    grid_y = np.linspace(view_cy - vr, view_cy + vr, 120)
    gx, gy = np.meshgrid(grid_x, grid_y)
    con_field = 1.0 / np.sqrt((gx - CCENTER_X) ** 2 + (gy - CCENTER_Y) ** 2)
    ax.contourf(gx, gy, con_field, levels=20, cmap="YlOrRd", alpha=0.3)
    ax.contour(gx, gy, con_field, levels=8, colors="orange", alpha=0.4, linewidths=0.5)

    # 机器人身体
    ax.plot(frame["xy1"][:, 0], frame["xy1"][:, 1], color="tab:blue", linewidth=2.5)
    ax.plot(frame["xy2"][:, 0], frame["xy2"][:, 1], color="tab:red", linewidth=2.5)
    ax.scatter([centroid1[0]], [centroid1[1]], color="tab:blue", s=40)
    ax.scatter([centroid2[0]], [centroid2[1]], color="tab:red", s=40)

    # 化学源标记
    ax.scatter([CCENTER_X], [CCENTER_Y], color="gold", s=150, marker="*", edgecolors="black", zorder=10)

    if len(trace1) > 1:
        trace1_np = np.array(trace1)
        trace2_np = np.array(trace2)
        ax.plot(trace1_np[:, 0], trace1_np[:, 1], color="tab:blue", alpha=0.5, linewidth=1.0)
        ax.plot(trace2_np[:, 0], trace2_np[:, 1], color="tab:red", alpha=0.5, linewidth=1.0)

    heading1 = compute_average_heading(frame["state1"])
    heading2 = compute_average_heading(frame["state2"])
    draw_heading(ax, centroid1, heading1, "tab:green")
    draw_heading(ax, centroid2, heading2, "tab:green")

    center_x = 0.5 * (centroid1[0] + centroid2[0])
    center_y = 0.5 * (centroid1[1] + centroid2[1])
    ax.set_xlim(center_x - ARGS.view_range, center_x + ARGS.view_range)
    ax.set_ylim(center_y - ARGS.view_range, center_y + ARGS.view_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    substep_index = int(frame.get("substep_index", total_substeps))
    info_text = "\n".join(
        [
            f"Macro step: {macro_index}",
            f"Substep: {substep_index}/{total_substeps}",
            f"Chem Source: ({CCENTER_X:.1f}, {CCENTER_Y:.1f})",
            f"R1 strategy: {strategy_pair[0]}",
            f"R1 reward: {robot_rewards[0]:.4f}, con: {robot_concentrations[0]:.4f}",
            f"R2 strategy: {strategy_pair[1]}",
            f"R2 reward: {robot_rewards[1]:.4f}, con: {robot_concentrations[1]:.4f}",
            f"Buffer: {queue_fill}/{queue_capacity}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title("Dual Flagella Chemotaxis Visualization")


def compute_agent_action(policy, obs, explore=False):
    # Discrete(6) 观测需要手动 one-hot，因为直接调用 policy 不经过 RLlib 预处理器
    if isinstance(obs, (int, np.integer)):
        obs = np.eye(NUM_STRATEGIES, dtype=np.float32)[obs]
    try:
        action_output = policy.compute_single_action(obs, explore=explore)
    except TypeError:
        action_output = policy.compute_single_action(obs)
    action = int(unpack_action_output(action_output))
    return int(np.clip(action, 0, 2))  # Discrete(3): 0,1,2


def compute_macro_package(policy, env, obs_dict):
    action_dict = {}
    for robot_id in ROBOT_IDS:
        action_id = compute_agent_action(policy, obs_dict[robot_id], explore=ARGS.explore)
        action_dict[robot_id] = action_id

    next_obs, reward_dict, done_dict, _ = env.step(action_dict)
    frames = env.last_substep_frames if env.last_substep_frames else [capture_env_frame(env, env.low_level_hold_steps)]

    strategy_pair = (STRATEGY_NAMES[env.order1], STRATEGY_NAMES[env.order2])

    package = {
        "action_dict": dict(action_dict),
        "done": bool(done_dict["__all__"]),
        "next_obs": next_obs,
        "frames": frames,
        "strategy_pair": strategy_pair,
        "robot_rewards": list(env.last_robot_rewards),
        "robot_concentrations": [env.con1, env.con2],
        "robot_orders": [env.order1, env.order2],
        "reward_dict": dict(reward_dict),
    }
    return package


def rollover_env_without_geometry_reset(env):
    env.done = False
    env.reward = 0.0
    env.ep_step = 0
    env.episode_count += 1


def producer_loop(policy, env, initial_obs, output_queue, stop_event, total_steps):
    obs_dict = initial_obs
    produced = 0
    try:
        while produced < total_steps and not stop_event.is_set():
            package = compute_macro_package(policy, env, obs_dict)
            while not stop_event.is_set():
                try:
                    output_queue.put(package, timeout=0.1)
                    break
                except Full:
                    continue
            if stop_event.is_set():
                break
            produced += 1
            obs_dict = package["next_obs"]
            if package["done"]:
                if ARGS.reset_free_playback:
                    rollover_env_without_geometry_reset(env)
                    obs_dict = env._get_obs()
                else:
                    obs_dict = env.reset()
    except Exception:
        error_package = {
            "error": True,
            "traceback": traceback.format_exc(),
        }
        while not stop_event.is_set():
            try:
                output_queue.put(error_package, timeout=0.1)
                break
            except Full:
                continue
    finally:
        sentinel = {"done_producing": True}
        while not stop_event.is_set():
            try:
                output_queue.put(sentinel, timeout=0.1)
                break
            except Full:
                continue


def wait_for_queue_warmup(fig, ax, env, trace1, trace2, package_queue, warmup_steps, producer):
    last_ui_update = 0.0
    while package_queue.qsize() < warmup_steps and producer.is_alive():
        if not plt.fignum_exists(fig.number):
            return False

        now = time.time()
        if now - last_ui_update >= 0.05:
            render_frame(
                ax,
                capture_env_frame(env, substep_index=0),
                trace1,
                trace2,
                macro_index=0,
                strategy_pair=(STRATEGY_NAMES[env.order1], STRATEGY_NAMES[env.order2]),
                total_substeps=max(1, env.low_level_hold_steps),
                robot_rewards=env.last_robot_rewards,
                robot_concentrations=[env.con1, env.con2],
                robot_orders=[env.order1, env.order2],
                queue_fill=package_queue.qsize(),
                queue_capacity=max(1, ARGS.prefetch_queue_size),
                append_trace=True,
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            last_ui_update = now
        plt.pause(0.05)
    return True


def main():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=ARGS.num_cpus, log_to_driver=False)

    env = swimmer_gym(build_env_config(ARGS))
    obs_dict = env.reset()

    agent = ppo.PPO(config=build_config(), env=swimmer_gym)
    checkpoint = resolve_checkpoint(ARGS.checkpoint) if ARGS.checkpoint else find_latest_checkpoint()
    if checkpoint is None:
        print("[Error] No senior-policy checkpoint found. Run train.py first or pass --checkpoint.")
        sys.exit(1)

    print(f"Loading senior checkpoint: {checkpoint}")
    print(f"Ray CPUs: {ARGS.num_cpus}, PyTorch threads: {ARGS.num_threads}")
    print(
        "Playback mode: "
        + ("reset-free visualization rollover" if ARGS.reset_free_playback else "environment reset at episode boundary")
    )
    agent.restore(str(checkpoint))
    shared_policy = agent.get_policy(SHARED_POLICY_ID)
    if shared_policy is None:
        raise RuntimeError(f"Shared policy '{SHARED_POLICY_ID}' not found in checkpoint")
    print(">>> Checkpoint restore succeeded. Launching visualization window...")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    trace1 = []
    trace2 = []
    initial_frame = capture_env_frame(env, substep_index=0)
    render_frame(
        ax,
        initial_frame,
        trace1,
        trace2,
        macro_index=0,
        strategy_pair=(STRATEGY_NAMES[env.order1], STRATEGY_NAMES[env.order2]),
        total_substeps=1,
        robot_rewards=env.last_robot_rewards,
        robot_concentrations=[env.con1, env.con2],
        robot_orders=[env.order1, env.order2],
        queue_fill=0,
        queue_capacity=max(1, ARGS.prefetch_queue_size),
    )
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(ARGS.speed)

    preload_size = max(1, ARGS.prefetch_queue_size)
    package_queue = Queue(maxsize=preload_size)
    stop_event = Event()
    producer = Thread(
        target=producer_loop,
        args=(shared_policy, env, obs_dict, package_queue, stop_event, ARGS.steps),
        daemon=True,
    )
    producer.start()
    macro_index = 0

    warmup_steps = min(max(1, ARGS.prefetch_warmup_steps), preload_size, ARGS.steps)
    print(
        f">>> Warming up playback queue: target {warmup_steps} macro steps "
        f"(queue size {preload_size})"
    )
    keep_open = wait_for_queue_warmup(fig, ax, env, trace1, trace2, package_queue, warmup_steps, producer)
    if not keep_open:
        stop_event.set()
        producer.join(timeout=2.0)
        plt.ioff()
        return

    if package_queue.qsize() > 0:
        print(f">>> Playback queue warmup complete: {package_queue.qsize()} macro steps buffered")

    try:
        while macro_index < ARGS.steps:
            package = package_queue.get()
            if package.get("error"):
                raise RuntimeError(package["traceback"])
            if package.get("done_producing"):
                break

            macro_index += 1
            strategy_pair = package["strategy_pair"]
            r1 = package["robot_rewards"][0]
            r2 = package["robot_rewards"][1]

            print(
                f"[Macro {macro_index:>3d}] "
                f"R1: {strategy_pair[0]}, reward={r1:.4f}, con={package['robot_concentrations'][0]:.4f} | "
                f"R2: {strategy_pair[1]}, reward={r2:.4f}, con={package['robot_concentrations'][1]:.4f}"
            )

            should_stop = False
            for frame in package["frames"]:
                render_frame(
                    ax,
                    frame,
                    trace1,
                    trace2,
                    macro_index=macro_index,
                    strategy_pair=strategy_pair,
                    total_substeps=len(package["frames"]),
                    robot_rewards=package["robot_rewards"],
                    robot_concentrations=package["robot_concentrations"],
                    robot_orders=package["robot_orders"],
                    queue_fill=package_queue.qsize(),
                    queue_capacity=preload_size,
                )
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(ARGS.speed)

                if not plt.fignum_exists(fig.number):
                    should_stop = True
                    break

            if should_stop:
                break
    finally:
        stop_event.set()
        producer.join(timeout=2.0)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
