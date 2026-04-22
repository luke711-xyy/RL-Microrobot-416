import argparse
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Flagella Single-Agent Joint-Policy Visualizer")
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

from matplotlib.colors import LinearSegmentedColormap

from calculate_v import compute_stokeslet_forces, evaluate_stokeslet_velocity
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

FLOW_CMAP = LinearSegmentedColormap.from_list("flow_vel", ["#FFFDE7", "#FF8C00"])


def build_env_config(cli_args):
    return {
        "translate_ckpt": cli_args.translate_ckpt,
        "reorien_ckpt": cli_args.reorien_ckpt,
        "low_level_hold_steps": LOW_LEVEL_HOLD_STEPS,
        "macro_horizon": MACRO_HORIZON,
        "reset_free": getattr(cli_args, 'reset_free_playback', False),
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
    policy_roots = [item for item in base_dir.iterdir() if item.is_dir() and item.name.startswith("single_policy_")]
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
        "ll_action1": np.array(env._last_ll_action1, copy=True),
        "ll_action2": np.array(env._last_ll_action2, copy=True),
        "xfirst1": np.array(env.Xfirst1, copy=True),
        "xfirst2": np.array(env.Xfirst2, copy=True),
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

    # 浓度场等高线（同心圆热力图）— 固定 levels 保证远近一致可见
    view_cx = 0.5 * (centroid1[0] + centroid2[0])
    view_cy = 0.5 * (centroid1[1] + centroid2[1])
    vr = ARGS.view_range
    grid_x = np.linspace(view_cx - vr, view_cx + vr, 120)
    grid_y = np.linspace(view_cy - vr, view_cy + vr, 120)
    gx, gy = np.meshgrid(grid_x, grid_y)
    con_field = 1.0 / np.sqrt((gx - CCENTER_X) ** 2 + (gy - CCENTER_Y) ** 2)
    con_levels = np.linspace(0.05, 2.0, 40)
    ax.contourf(gx, gy, con_field, levels=con_levels, cmap="YlOrRd", alpha=0.55, extend="both")
    ax.contour(gx, gy, con_field, levels=10, colors="orange", alpha=0.5, linewidths=0.6)

    # Stokeslet 流体速度场（机器人扰动水流产生的流速箭头）
    if "flow_data" in frame:
        fp_x, fp_y, f_x, f_y, e_val = frame["flow_data"]
        q_n = 64
        qx = np.linspace(view_cx - vr * 0.95, view_cx + vr * 0.95, q_n)
        qy = np.linspace(view_cy - vr * 0.95, view_cy + vr * 0.95, q_n)
        qgx, qgy = np.meshgrid(qx, qy)
        ux, uy = evaluate_stokeslet_velocity(qgx, qgy, fp_x, fp_y, f_x, f_y, e_val)
        flow_mag = np.sqrt(ux ** 2 + uy ** 2)
        ax.quiver(
            qgx, qgy, ux, uy, flow_mag,
            cmap=FLOW_CMAP, alpha=0.75, scale=1.5, scale_units="width",
            width=0.0015, headwidth=3, headlength=3.5, zorder=2,
            clim=(0, np.percentile(flow_mag, 95)),
        )

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
    ax.set_title("Single-Agent Flagella Chemotaxis Visualization")


def compute_agent_action(policy, obs, explore=False):
    obs = np.asarray(obs, dtype=np.float32)
    try:
        action_output = policy.compute_single_action(obs, explore=explore)
    except TypeError:
        action_output = policy.compute_single_action(obs)
    action = int(unpack_action_output(action_output))
    return int(np.clip(action, 0, 8))  # Discrete(9): 0..8


def compute_macro_package(policy, env, obs):
    joint_action = compute_agent_action(policy, obs, explore=ARGS.explore)

    next_obs, reward, done, info = env.step(joint_action)
    frames = env.last_substep_frames if env.last_substep_frames else [capture_env_frame(env, env.low_level_hold_steps)]

    strategy_pair = (STRATEGY_NAMES[env.order1], STRATEGY_NAMES[env.order2])

    package = {
        "joint_action": joint_action,
        "done": bool(done),
        "next_obs": next_obs,
        "frames": frames,
        "strategy_pair": strategy_pair,
        "robot_rewards": list(env.last_robot_rewards),
        "robot_concentrations": [env.con1, env.con2],
        "robot_orders": [env.order1, env.order2],
        "reward": float(reward),
    }
    return package


def rollover_env_without_geometry_reset(env):
    env.done = False
    env.reward = 0.0
    env.ep_step = 0
    env.episode_count += 1


def main():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=ARGS.num_cpus, log_to_driver=False)

    env = swimmer_gym(build_env_config(ARGS))
    obs = env.reset()

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
    policy = agent.get_policy("default_policy")
    if policy is None:
        raise RuntimeError("Default policy not found in checkpoint")
    print(">>> Checkpoint restore succeeded.")

    # ================= 阶段 1：预计算所有 macro step =================
    print(f">>> Precomputing {ARGS.steps} macro steps (this may take a while)...")
    all_packages = []
    for i in range(ARGS.steps):
        joint_action = compute_agent_action(policy, obs, explore=ARGS.explore)

        next_obs, reward, done, info = env.step(joint_action)
        frames = env.last_substep_frames if env.last_substep_frames else [capture_env_frame(env, env.low_level_hold_steps)]
        strategy_pair = (STRATEGY_NAMES[env.order1], STRATEGY_NAMES[env.order2])

        package = {
            "frames": frames,
            "strategy_pair": strategy_pair,
            "robot_rewards": list(env.last_robot_rewards),
            "robot_concentrations": [env.con1, env.con2],
            "robot_orders": [env.order1, env.order2],
        }
        all_packages.append(package)

        print(
            f"  [{i + 1:>4d}/{ARGS.steps}] "
            f"R1: aprm={env.aprm1}->order={env.order1}({strategy_pair[0]}), rwd={env.last_robot_rewards[0]:>8.2f}, con={env.con1:.4f} | "
            f"R2: aprm={env.aprm2}->order={env.order2}({strategy_pair[1]}), rwd={env.last_robot_rewards[1]:>8.2f}, con={env.con2:.4f}"
        )

        obs = next_obs
        if done:
            if ARGS.reset_free_playback:
                rollover_env_without_geometry_reset(env)
                obs = env._get_obs()
            else:
                obs = env.reset()

    # 预计算 Stokeslet 流场力分布（每帧独立，播放时只需做轻量网格求值）
    total_frames = sum(len(p["frames"]) for p in all_packages)
    print(f">>> Computing Stokeslet flow field for {total_frames} frames...")
    computed = 0
    for package in all_packages:
        for frame in package["frames"]:
            if "ll_action1" in frame and "xfirst1" in frame:
                try:
                    flow_data = compute_stokeslet_forces(
                        frame["state1"], frame["ll_action1"], frame["xfirst1"],
                        frame["state2"], frame["ll_action2"], frame["xfirst2"],
                    )
                    frame["flow_data"] = flow_data
                except Exception:
                    pass
            computed += 1
            if computed % 100 == 0:
                print(f"    flow field: {computed}/{total_frames}")
    print(f">>> Flow field done.")

    print(f">>> Precomputation done. {len(all_packages)} macro steps ready. Launching playback...")

    # ================= 阶段 2：流畅回放 =================
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    trace1 = []
    trace2 = []

    try:
        for macro_index, package in enumerate(all_packages, start=1):
            strategy_pair = package["strategy_pair"]

            for frame in package["frames"]:
                if not plt.fignum_exists(fig.number):
                    raise KeyboardInterrupt

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
                    queue_fill=len(all_packages) - macro_index,
                    queue_capacity=len(all_packages),
                )
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(ARGS.speed)

    except KeyboardInterrupt:
        print("\nPlayback interrupted.")

    print("Playback finished. Close the window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
