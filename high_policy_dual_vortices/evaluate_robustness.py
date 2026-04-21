import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate policy robustness against cellular vortex flow")
    parser.add_argument("--translate_ckpt", type=str, required=True)
    parser.add_argument("--reorien_ckpt", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Senior-policy checkpoint path")
    parser.add_argument("--policy_type", type=str, required=True, choices=["dual", "shared", "single"],
                        help="dual=2 policies, shared=1 policy for both, single=joint centralized policy")
    parser.add_argument("--num_trials", type=int, default=30, help="Trials per Uv value (default: 30)")
    parser.add_argument("--uv_values", type=str, default="0.0,0.005,0.01,0.02,0.03,0.05,0.07,0.1",
                        help="Comma-separated Uv values to test")
    parser.add_argument("--success_radius", type=float, default=1.0, help="Distance threshold for success (default: 1.0)")
    parser.add_argument("--steps", type=int, default=200, help="Macro steps per trial (default: 200)")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--view_range", type=float, default=8.0, help="Half-width of scatter plot view (default: 8.0)")
    parser.add_argument("--no_plot", action="store_true", help="Skip figure generation, only print table")
    return parser.parse_args()


ARGS = parse_args()
os.environ["STOKES_NUM_THREADS"] = str(ARGS.num_threads)

os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
from gym import spaces

import calculate_v
import swimmer as swimmer_module
from calculate_v import vortex_vorticity, vortex_velocity, VORTEX_CELL_L
from swimmer import (
    CCENTER_X,
    CCENTER_Y,
    LOW_LEVEL_HOLD_STEPS,
    NUM_STRATEGIES,
    ROBOT_IDS,
    swimmer_gym,
)

TARGET = (CCENTER_X, CCENTER_Y)
START_DIST = np.sqrt((4 - 2) ** 2 + (4 - 0) ** 2)

POLICY_ID_R1 = "policy_robot_1"
POLICY_ID_R2 = "policy_robot_2"
SHARED_POLICY_ID = "shared_policy"


def policy_mapping_dual(agent_id, episode, worker, **kwargs):
    return POLICY_ID_R1 if agent_id == "robot_1" else POLICY_ID_R2


def policy_mapping_shared(agent_id, episode, worker, **kwargs):
    return SHARED_POLICY_ID


# ==================== Checkpoint 工具 ====================

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


def resolve_checkpoint(path_str):
    cp_path = Path(path_str).expanduser().resolve()
    if cp_path.is_file():
        return cp_path
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {cp_path}")
    if is_checkpoint_path(cp_path):
        return cp_path
    direct_candidates = sorted(
        [c for c in cp_path.iterdir() if is_checkpoint_path(c)],
        key=checkpoint_sort_key,
    )
    if direct_candidates:
        return direct_candidates[-1]
    nested_candidates = sorted(
        [c for c in cp_path.rglob("*") if is_checkpoint_path(c)],
        key=checkpoint_sort_key,
    )
    if nested_candidates:
        return nested_candidates[-1]
    raise FileNotFoundError(f"No checkpoint found under: {cp_path}")


# ==================== PPO Config 构建 ====================

def build_env_config():
    return {
        "translate_ckpt": ARGS.translate_ckpt,
        "reorien_ckpt": ARGS.reorien_ckpt,
        "low_level_hold_steps": LOW_LEVEL_HOLD_STEPS,
        "macro_horizon": ARGS.steps,
    }


def build_config(policy_type):
    env_stub = swimmer_gym({**build_env_config(), "skip_policy_load": True})
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["num_rollout_workers"] = 0
    config["framework"] = "torch"
    config["env_config"] = build_env_config()
    config["horizon"] = ARGS.steps
    config["rollout_fragment_length"] = ARGS.steps

    if policy_type == "dual":
        config["env"] = swimmer_gym
        config["multiagent"] = {
            "policies": {
                POLICY_ID_R1: (None, env_stub.observation_space, env_stub.action_space, {}),
                POLICY_ID_R2: (None, env_stub.observation_space, env_stub.action_space, {}),
            },
            "policy_mapping_fn": policy_mapping_dual,
            "policies_to_train": [POLICY_ID_R1, POLICY_ID_R2],
            "count_steps_by": "env_steps",
        }
    elif policy_type == "shared":
        config["env"] = swimmer_gym
        config["multiagent"] = {
            "policies": {
                SHARED_POLICY_ID: (None, env_stub.observation_space, env_stub.action_space, {}),
            },
            "policy_mapping_fn": policy_mapping_shared,
            "policies_to_train": [SHARED_POLICY_ID],
            "count_steps_by": "env_steps",
        }
    elif policy_type == "single":
        config["env"] = _SingleStubEnv
    return config


class _SingleStubEnv:
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
    action_space = spaces.Discrete(9)


# ==================== 策略加载与动作计算 ====================

def load_policies(agent, policy_type):
    if policy_type == "dual":
        r1 = agent.get_policy(POLICY_ID_R1)
        r2 = agent.get_policy(POLICY_ID_R2)
        if r1 is None or r2 is None:
            raise RuntimeError("Failed to load dual policies from checkpoint")
        return {"robot_1": r1, "robot_2": r2}
    elif policy_type == "shared":
        p = agent.get_policy(SHARED_POLICY_ID)
        if p is None:
            raise RuntimeError("Failed to load shared policy from checkpoint")
        return {"robot_1": p, "robot_2": p}
    elif policy_type == "single":
        p = agent.get_policy("default_policy")
        if p is None:
            raise RuntimeError("Failed to load single/default policy from checkpoint")
        return {"joint": p}


def unpack_action_output(action_output):
    if not isinstance(action_output, tuple):
        return action_output
    if len(action_output) == 0:
        raise ValueError("compute_single_action returned an empty tuple")
    return action_output[0]


def compute_agent_action(policy, obs, explore=False):
    if isinstance(obs, (int, np.integer)):
        obs = np.eye(NUM_STRATEGIES, dtype=np.float32)[obs]
    try:
        action_output = policy.compute_single_action(obs, explore=explore)
    except TypeError:
        action_output = policy.compute_single_action(obs)
    action = int(unpack_action_output(action_output))
    return int(np.clip(action, 0, policy.action_space.n - 1))


def compute_actions(policies, obs_dict, policy_type):
    if policy_type in ("dual", "shared"):
        action_dict = {}
        for robot_id in ROBOT_IDS:
            action_dict[robot_id] = compute_agent_action(
                policies[robot_id], obs_dict[robot_id], explore=False
            )
        return action_dict
    elif policy_type == "single":
        obs1_oh = np.eye(NUM_STRATEGIES, dtype=np.float32)[obs_dict["robot_1"]]
        obs2_oh = np.eye(NUM_STRATEGIES, dtype=np.float32)[obs_dict["robot_2"]]
        joint_obs = np.concatenate([obs1_oh, obs2_oh])
        joint_action = compute_agent_action(policies["joint"], joint_obs, explore=False)
        return {
            "robot_1": joint_action // 3,
            "robot_2": joint_action % 3,
        }


# ==================== 随机起始位置 ====================

def random_start_positions():
    angle1 = np.random.uniform(0, 2 * np.pi)
    angle2 = np.random.uniform(0, 2 * np.pi)
    r1 = (TARGET[0] + START_DIST * np.cos(angle1), TARGET[1] + START_DIST * np.sin(angle1))
    r2 = (TARGET[0] + START_DIST * np.cos(angle2), TARGET[1] + START_DIST * np.sin(angle2))
    return r1, r2


# ==================== 绘图 ====================

def plot_results(results, uv_values, policy_type, success_radius):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7),
                                             gridspec_kw={"width_ratios": [1.2, 1]})

    # 左图：取最大非零 Uv 的散点
    nonzero_uvs = [uv for uv in uv_values if uv > 0 and uv in results]
    plot_uv = max(nonzero_uvs) if nonzero_uvs else uv_values[-1]
    trials = results[plot_uv]

    vr = ARGS.view_range
    cx, cy = TARGET

    # 涡度背景
    v_n = 200
    vx_arr = np.linspace(cx - vr, cx + vr, v_n)
    vy_arr = np.linspace(cy - vr, cy + vr, v_n)
    vgx, vgy = np.meshgrid(vx_arr, vy_arr)
    vort = vortex_vorticity(vgx, vgy, Uv=plot_uv)
    L_v = VORTEX_CELL_L
    vmax = 2.0 * math.pi * plot_uv / L_v
    if vmax > 0:
        ax_left.imshow(
            vort,
            extent=(cx - vr, cx + vr, cy - vr, cy + vr),
            origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax,
            alpha=0.45, zorder=0,
        )

    # 涡胞旋转方向标记
    r_arc = 0.3 * L_v
    nx_lo = int(np.floor((cx - vr) / L_v))
    nx_hi = int(np.ceil((cx + vr) / L_v))
    ny_lo = int(np.floor((cy - vr) / L_v))
    ny_hi = int(np.ceil((cy + vr) / L_v))
    for _nx in range(nx_lo, nx_hi):
        for _ny in range(ny_lo, ny_hi):
            ccx = (_nx + 0.5) * L_v
            ccy = (_ny + 0.5) * L_v
            omega = vortex_vorticity(ccx, ccy, Uv=plot_uv)
            if abs(omega) < 1e-12:
                continue
            if omega > 0:
                angles = np.linspace(np.radians(20), np.radians(340), 50)
            else:
                angles = np.linspace(np.radians(340), np.radians(20), 50)
            arc_x = ccx + r_arc * np.cos(angles)
            arc_y = ccy + r_arc * np.sin(angles)
            ax_left.plot(arc_x, arc_y, color="#444444", lw=0.6, alpha=0.4, zorder=1)
            ax_left.annotate(
                "", xy=(arc_x[-1], arc_y[-1]), xytext=(arc_x[-4], arc_y[-4]),
                arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                                color="#444444", lw=0.6),
                zorder=1,
            )

    # 浓度场
    grid_x = np.linspace(cx - vr, cx + vr, 120)
    grid_y = np.linspace(cy - vr, cy + vr, 120)
    gx, gy = np.meshgrid(grid_x, grid_y)
    con_field = 1.0 / np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    con_levels = np.linspace(0.05, 2.0, 40)
    ax_left.contourf(gx, gy, con_field, levels=con_levels, cmap="YlOrRd", alpha=0.3, extend="both")

    # 起始距离参考圆
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax_left.plot(cx + START_DIST * np.cos(theta_circle),
                 cy + START_DIST * np.sin(theta_circle),
                 'k--', lw=0.8, alpha=0.4, zorder=2)

    # 终点散点
    for trial in trials:
        for key, marker_size in [("final_r1", 50), ("final_r2", 50)]:
            pos = trial[key]
            success_key = "success_r1" if key == "final_r1" else "success_r2"
            color = "tab:cyan" if trial[success_key] else "tab:red"
            ax_left.scatter(pos[0], pos[1], c=color, marker="x", s=marker_size,
                           linewidths=1.2, alpha=0.7, zorder=5)

    # 化学源标记
    ax_left.scatter([cx], [cy], color="gold", s=200, marker="*",
                    edgecolors="black", linewidths=0.8, zorder=10)

    ax_left.set_xlim(cx - vr, cx + vr)
    ax_left.set_ylim(cy - vr, cy + vr)
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("x / L")
    ax_left.set_ylabel("y / L")
    ax_left.set_title(f"Endpoints at Uv = {plot_uv:.4f}  ({policy_type} policy)")
    ax_left.grid(True, alpha=0.15)

    # 右图：成功率曲线
    uv_plot = []
    rate_plot = []
    for uv in sorted(results.keys()):
        trials_uv = results[uv]
        total = len(trials_uv) * 2
        successes = sum(t["success_r1"] + t["success_r2"] for t in trials_uv)
        rate = successes / total if total > 0 else 0.0
        uv_plot.append(uv)
        rate_plot.append(rate)

    ax_right.plot(uv_plot, rate_plot, "ko-", markersize=6, linewidth=1.5)
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.set_xlabel("Uv")
    ax_right.set_ylabel("Success rate")
    ax_right.set_title(f"Robustness ({policy_type} policy)")
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.getcwd(), f"robustness_{policy_type}_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out_path}")
    plt.close(fig)


# ==================== main ====================

def main():
    uv_values = [float(v.strip()) for v in ARGS.uv_values.split(",")]

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=ARGS.num_cpus, log_to_driver=False)

    # 构建 PPO config 并加载 checkpoint
    policy_type = ARGS.policy_type

    if policy_type == "single":
        stub = _SingleStubEnv()
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = 0
        config["num_workers"] = 0
        config["num_rollout_workers"] = 0
        config["framework"] = "torch"
        config["env_config"] = {}
        config["observation_space"] = stub.observation_space
        config["action_space"] = stub.action_space
    else:
        config = build_config(policy_type)

    checkpoint = resolve_checkpoint(ARGS.checkpoint)
    print(f"Policy type: {policy_type}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Uv values: {uv_values}")
    print(f"Trials per Uv: {ARGS.num_trials}")
    print(f"Steps per trial: {ARGS.steps}")
    print(f"Success radius: {ARGS.success_radius}")
    print(f"Start distance: {START_DIST:.4f}")

    if policy_type == "single":
        agent = ppo.PPO(config=config)
    else:
        agent = ppo.PPO(config=config, env=swimmer_gym)

    agent.restore(str(checkpoint))
    policies = load_policies(agent, policy_type)
    print("Checkpoint loaded successfully.")

    # 创建环境（始终使用涡流版 MultiAgentEnv）
    env_cfg = build_env_config()
    env_cfg["macro_horizon"] = ARGS.steps
    env = swimmer_gym(env_cfg)

    # 批量评估
    results = {}
    for uv in uv_values:
        calculate_v.VORTEX_STRENGTH = uv
        trials = []

        for trial_idx in range(ARGS.num_trials):
            r1_init, r2_init = random_start_positions()
            swimmer_module.ROBOT1_INIT = r1_init
            swimmer_module.ROBOT2_INIT = r2_init
            obs_dict = env.reset()

            for step in range(ARGS.steps):
                action_dict = compute_actions(policies, obs_dict, policy_type)
                obs_dict, rewards, dones, infos = env.step(action_dict)
                if dones["__all__"]:
                    break

            d1 = np.sqrt((env.last_centroid1[0] - TARGET[0]) ** 2 +
                         (env.last_centroid1[1] - TARGET[1]) ** 2)
            d2 = np.sqrt((env.last_centroid2[0] - TARGET[0]) ** 2 +
                         (env.last_centroid2[1] - TARGET[1]) ** 2)
            trials.append({
                "success_r1": d1 < ARGS.success_radius,
                "success_r2": d2 < ARGS.success_radius,
                "final_r1": np.array(env.last_centroid1, copy=True),
                "final_r2": np.array(env.last_centroid2, copy=True),
                "start_r1": r1_init,
                "start_r2": r2_init,
                "dist_r1": d1,
                "dist_r2": d2,
            })

        results[uv] = trials

        total = ARGS.num_trials * 2
        successes = sum(t["success_r1"] + t["success_r2"] for t in trials)
        rate = successes / total
        print(f"  Uv={uv:.4f}: success={successes}/{total} = {rate:.1%}")

    # 汇总表格
    print("\n" + "=" * 60)
    print(f"{'Uv':>10s}  {'Success':>8s}  {'Total':>6s}  {'Rate':>8s}")
    print("-" * 60)
    for uv in sorted(results.keys()):
        trials = results[uv]
        total = len(trials) * 2
        successes = sum(t["success_r1"] + t["success_r2"] for t in trials)
        rate = successes / total if total > 0 else 0.0
        print(f"{uv:>10.4f}  {successes:>8d}  {total:>6d}  {rate:>8.1%}")
    print("=" * 60)

    if not ARGS.no_plot:
        plot_results(results, uv_values, policy_type, ARGS.success_radius)

    ray.shutdown()


if __name__ == "__main__":
    main()
