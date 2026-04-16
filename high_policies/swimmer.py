import math
import os
from collections import deque
from pathlib import Path

import numpy as np
from gym import spaces
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy

from calculate_v import NL as PRIMITIVE_LINK_NUM, RK_dual


directory_path = os.getcwd()

DT = 0.01
ENV_LINK_NUM = PRIMITIVE_LINK_NUM

ACTION_LOW = -1
ACTION_HIGH = 1

LOW_LEVEL_HOLD_STEPS = 25
MACRO_HORIZON = 50

# 化学源位置
CCENTER_X = 4.0
CCENTER_Y = 0.0
CON_REWARD_SCALE = 10000.0

ROBOT1_INIT = (-4.0, 0.3)
ROBOT2_INIT = (-4.0, -0.3)

ROBOT_IDS = ("robot_1", "robot_2")

# 策略系统：2 个底层策略 × 变换 = 6 种行为
NUM_STRATEGIES = 6
STRATEGY_NAMES = (
    "turn_a_head",      # order 0: reorien, 恒等
    "forward_head",     # order 1: translate, 恒等
    "turn_b_mirror",    # order 2: reorien, 取反
    "turn_a_tail",      # order 3: reorien, 取反+倒序
    "backward_tail",    # order 4: translate, 倒序
    "turn_b_tail",      # order 5: reorien, 倒序
)
STRATEGY_TO_POLICY = {
    0: "reorien",
    1: "translate",
    2: "reorien",
    3: "reorien",
    4: "translate",
    5: "reorien",
}

traj = []
traj2 = []
trajp = []


def _stack_trace(existing, row):
    row = np.asarray(row, dtype=np.float64).reshape(1, -1)
    if isinstance(existing, list) and len(existing) == 0:
        return row
    return np.concatenate((existing.reshape(-1, row.shape[1]), row), axis=0)


def compute_true_centroid(xy_positions):
    xy_positions = np.asarray(xy_positions, dtype=np.float64)
    return np.mean(xy_positions, axis=0)


def compute_average_heading(state_array):
    head_omega = state_array[2]
    running_angle = head_omega
    angle_sum = head_omega
    for beta in state_array[3:]:
        running_angle += beta
        angle_sum += running_angle
    return angle_sum / (len(state_array) - 2)


def subsample_link_positions(xy_positions):
    """从 Stokeslet 离散点 (N+1) 中取出 NL+1 个链节位置，保证和 reset 时点数一致"""
    n_pts = xy_positions.shape[0]
    if n_pts == ENV_LINK_NUM + 1:
        return xy_positions
    step = (n_pts - 1) // ENV_LINK_NUM
    indices = [step * i for i in range(ENV_LINK_NUM)] + [n_pts - 1]
    return xy_positions[indices]


def compute_concentration(xy_positions, cx=CCENTER_X, cy=CCENTER_Y):
    """计算各链节处的化学浓度 (1/距离)，始终在 NL+1 个链节位置上计算"""
    link_pos = subsample_link_positions(xy_positions)
    dx = link_pos[:, 0] - cx
    dy = link_pos[:, 1] - cy
    return 1.0 / np.sqrt(dx ** 2 + dy ** 2)


def coarse_select_order(xy_positions, cx=CCENTER_X, cy=CCENTER_Y):
    """论文粗选：比较前端 tip 和后端 tip 的浓度，从对应候选集中随机选一个。
    前端浓度 >= 后端 → {0,1,2} (头朝前策略)
    前端浓度 <  后端 → {3,4,5} (尾朝前策略)
    """
    link_pos = subsample_link_positions(xy_positions)
    rear_tip = link_pos[0]
    front_tip = link_pos[-1]
    c_front = 1.0 / np.sqrt((front_tip[0] - cx) ** 2 + (front_tip[1] - cy) ** 2)
    c_rear = 1.0 / np.sqrt((rear_tip[0] - cx) ** 2 + (rear_tip[1] - cy) ** 2)
    if c_front >= c_rear:
        return np.random.randint(3)        # {0, 1, 2}
    else:
        return np.random.randint(3) + 3    # {3, 4, 5}


def transform_obs_for_strategy(hinge_angles, order):
    """按 order 变换铰链角观测（喂给底层策略前）"""
    obs = hinge_angles.copy()
    if order == 2:
        obs = -obs
    elif order == 3:
        obs = -obs[::-1].copy()
    elif order == 4:
        obs = obs[::-1].copy()
    elif order == 5:
        obs = obs[::-1].copy()
    # order 0, 1: 恒等
    return obs


def transform_action_for_strategy(action, order):
    """按 order 反变换底层策略输出的动作（还原到物理空间）"""
    act = action.copy()
    if order == 2:
        act = -act
    elif order == 3:
        act = -(act[::-1].copy())
    elif order == 4:
        act = act[::-1].copy()
    elif order == 5:
        act = act[::-1].copy()
    # order 0, 1: 恒等
    return act


# ================= Checkpoint 工具函数 =================

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
        or (path_obj / "policies" / "default_policy").exists()
    )


def checkpoint_sort_key(path_obj):
    path_obj = Path(path_obj)
    digits = "".join(ch for ch in path_obj.name if ch.isdigit())
    order = int(digits) if digits else -1
    return (order, str(path_obj))


def resolve_policy_checkpoint_dir(path_str):
    path_obj = Path(path_str).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_obj}")

    if path_obj.is_file():
        candidate = path_obj.parent / "policies" / "default_policy"
        if candidate.exists():
            return candidate

    if (path_obj / "policies" / "default_policy").exists():
        return path_obj / "policies" / "default_policy"

    direct_candidates = sorted(
        [candidate for candidate in path_obj.rglob("default_policy") if candidate.is_dir() and candidate.name == "default_policy"],
        key=checkpoint_sort_key,
    )
    if direct_candidates:
        return direct_candidates[-1]

    raise FileNotFoundError(f"No RLlib policy directory found under: {path_obj}")


def restore_policy(path_str):
    policy_dir = resolve_policy_checkpoint_dir(path_str)
    restored = Policy.from_checkpoint(str(policy_dir))
    if isinstance(restored, dict):
        if "default_policy" in restored:
            return restored["default_policy"]
        if len(restored) == 1:
            return next(iter(restored.values()))
        raise ValueError(f"Unexpected policy dictionary keys: {list(restored.keys())}")
    return restored


def get_policy_initial_state(policy):
    try:
        state = policy.get_initial_state()
    except Exception:
        return []
    return [np.array(item, copy=True) for item in state]


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


# ================= 双机器人高层环境 =================

class swimmer_gym(MultiAgentEnv):
    metadata = {
        "render.modes": ["human"],
        "video.frames_per_second": 30,
    }

    def __init__(self, env_config):
        super().__init__()
        env_config = env_config or {}
        self._agent_ids = set(ROBOT_IDS)

        self.dt = DT
        self.low_level_hold_steps = int(env_config.get("low_level_hold_steps", LOW_LEVEL_HOLD_STEPS))
        self.macro_horizon = int(env_config.get("macro_horizon", MACRO_HORIZON))
        self.skip_policy_load = bool(env_config.get("skip_policy_load", False))
        self.reset_free = bool(env_config.get("reset_free", False))
        self.translate_ckpt = env_config.get("translate_ckpt")
        self.reorien_ckpt = env_config.get("reorien_ckpt")

        self.betamax = (2 * math.pi) / ENV_LINK_NUM
        self.betamin = -self.betamax * 0.5

        # 动作：Discrete(3) = RL 微调 aadj ∈ {-1, 0, +1}
        self.action_space = spaces.Discrete(3)
        # 观测：Discrete(6) = 粗选出的 aprm
        self.observation_space = spaces.Discrete(NUM_STRATEGIES)

        self.low_level_policies = {}
        self.low_level_states = [{}, {}]
        if not self.skip_policy_load:
            self._load_low_level_policies()

        self.episode_count = 0
        self.it = 0
        self.low_level_step_count = 0
        self.ep_step = 0
        self.reward = 0.0
        self.done = False

        # 每个机器人独立的状态
        self.order1 = 1          # 实际执行的 order (aprm + aadj)
        self.order2 = 1
        self.aprm1 = 1           # 粗选结果（作为 RL 观测）
        self.aprm2 = 1
        self.con1 = 0.0          # 铰链平均浓度
        self.con2 = 0.0

        # 跟踪变量（用于日志和可视化）
        self.last_robot_rewards = [0.0, 0.0]
        self.last_robot_orders = [1, 1]
        self.last_centroid1 = np.zeros((2,), dtype=np.float64)
        self.last_centroid2 = np.zeros((2,), dtype=np.float64)
        self.last_substep_frames = []

        self.trace1 = deque(maxlen=1000)
        self.trace2 = deque(maxlen=1000)

        self._build_initial_geometry()

    def _build_initial_robot_state(self, init_xy):
        centroid_x, centroid_y = init_xy
        state = np.zeros((ENV_LINK_NUM + 2,), dtype=np.float64)
        state[0] = centroid_x
        state[1] = centroid_y
        state[2] = 0.0

        x_first = np.zeros((2,), dtype=np.float64)
        x_first[0] = centroid_x - 0.5 * math.cos(state[2])
        x_first[1] = centroid_y - 0.5 * math.sin(state[2])

        x_positions = np.zeros((ENV_LINK_NUM + 1,), dtype=np.float64)
        y_positions = np.zeros((ENV_LINK_NUM + 1,), dtype=np.float64)
        for i in range(ENV_LINK_NUM + 1):
            x_positions[i] = x_first[0] + i / ENV_LINK_NUM * math.cos(state[2])
            y_positions[i] = x_first[1] + i / ENV_LINK_NUM * math.sin(state[2])
        xy_positions = np.concatenate((x_positions.reshape(-1, 1), y_positions.reshape(-1, 1)), axis=1)

        true_centroid = compute_true_centroid(xy_positions)
        state[0] = true_centroid[0]
        state[1] = true_centroid[1]
        return state, x_first, xy_positions

    def _build_initial_geometry(self):
        self.state1, self.Xfirst1, self.XY_positions1 = self._build_initial_robot_state(ROBOT1_INIT)
        self.state2, self.Xfirst2, self.XY_positions2 = self._build_initial_robot_state(ROBOT2_INIT)

        self._reset_policy_states()

        centroid1 = compute_true_centroid(self.XY_positions1)
        centroid2 = compute_true_centroid(self.XY_positions2)
        self.last_centroid1 = np.array(centroid1, dtype=np.float64)
        self.last_centroid2 = np.array(centroid2, dtype=np.float64)
        self.trace1.clear()
        self.trace2.clear()
        self.trace1.append(np.array(centroid1))
        self.trace2.append(np.array(centroid2))

        # 初始化铰链平均浓度
        self.con1 = float(np.mean(compute_concentration(self.XY_positions1)))
        self.con2 = float(np.mean(compute_concentration(self.XY_positions2)))

        # 粗选初始 order
        self.aprm1 = coarse_select_order(self.XY_positions1)
        self.aprm2 = coarse_select_order(self.XY_positions2)
        self.order1 = self.aprm1
        self.order2 = self.aprm2

        self.last_robot_rewards = [0.0, 0.0]
        self.last_robot_orders = [self.order1, self.order2]
        self.last_substep_frames = [self._capture_substep_frame(0)]

    def _reset_policy_states(self):
        if self.skip_policy_load:
            self.low_level_states = [{}, {}]
            return
        self.low_level_states = []
        for _robot_idx in range(2):
            robot_states = {}
            for policy_name, policy in self.low_level_policies.items():
                robot_states[policy_name] = get_policy_initial_state(policy)
            self.low_level_states.append(robot_states)

    def _load_low_level_policies(self):
        required = {
            "translate": self.translate_ckpt,
            "reorien": self.reorien_ckpt,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing low-level checkpoint paths for: {', '.join(missing)}")
        for policy_name, ckpt_path in required.items():
            self.low_level_policies[policy_name] = restore_policy(ckpt_path)

    def _get_obs(self):
        """观测 = 粗选出的 aprm（论文设计：RL 只看到粗选结果，决定如何微调）"""
        return {
            ROBOT_IDS[0]: self.aprm1,
            ROBOT_IDS[1]: self.aprm2,
        }

    def _sanitize_low_level_action(self, state, action):
        action = np.asarray(action, dtype=np.float64)
        clipped = np.clip(action, ACTION_LOW, ACTION_HIGH)
        # 逐关节裁剪：每个铰链角独立限制在 [-betamax, betamax]
        predicted = state[3:] + clipped * 0.2
        safe = np.clip(predicted, -self.betamax, self.betamax)
        return (safe - state[3:]) / 0.2

    def _compute_low_level_action(self, robot_idx, strategy_order):
        if self.skip_policy_load:
            return np.zeros((ENV_LINK_NUM - 1,), dtype=np.float64)

        # 1. 选底层策略
        policy_name = STRATEGY_TO_POLICY[strategy_order]
        policy = self.low_level_policies[policy_name]
        recurrent_state = self.low_level_states[robot_idx][policy_name]

        # 2. 取原始铰链角
        raw_hinges = self.state1[3:].copy() if robot_idx == 0 else self.state2[3:].copy()

        # 3. 按 order 变换观测
        obs = transform_obs_for_strategy(raw_hinges, strategy_order)

        # 4. 调用底层策略
        try:
            action_output = policy.compute_single_action(obs, state=recurrent_state, explore=False)
        except TypeError:
            action_output = policy.compute_single_action(obs, state=recurrent_state)

        action, next_state = unpack_action_output(action_output, recurrent_state)
        self.low_level_states[robot_idx][policy_name] = next_state
        action = np.asarray(action, dtype=np.float64).reshape(-1)

        # 5. 按 order 反变换动作
        action = transform_action_for_strategy(action, strategy_order)

        # 6. 关节限位检查
        target_state = self.state1 if robot_idx == 0 else self.state2
        return self._sanitize_low_level_action(target_state, action)

    def _apply_dual_solver(self, action1, action2):
        (
            state1_next,
            _xn1,
            _yn1,
            _r1,
            x_first_delta1,
            x_positions1,
            y_positions1,
            state2_next,
            _xn2,
            _yn2,
            _r2,
            x_first_delta2,
            x_positions2,
            y_positions2,
            _pressure_diff,
            _pressure_end,
            _pressure_all,
        ) = RK_dual(self.state1, action1, self.Xfirst1, self.state2, action2, self.Xfirst2)

        self.state1 = state1_next.copy()
        self.state2 = state2_next.copy()
        self.Xfirst1 = self.Xfirst1 + x_first_delta1
        self.Xfirst2 = self.Xfirst2 + x_first_delta2
        self.XY_positions1 = np.concatenate((np.array(x_positions1).reshape(-1, 1), np.array(y_positions1).reshape(-1, 1)), axis=1)
        self.XY_positions2 = np.concatenate((np.array(x_positions2).reshape(-1, 1), np.array(y_positions2).reshape(-1, 1)), axis=1)

        centroid1 = compute_true_centroid(self.XY_positions1)
        centroid2 = compute_true_centroid(self.XY_positions2)
        self.state1[0] = centroid1[0]
        self.state1[1] = centroid1[1]
        self.state2[0] = centroid2[0]
        self.state2[1] = centroid2[1]

    def _capture_substep_frame(self, substep_index):
        centroid1 = compute_true_centroid(self.XY_positions1)
        centroid2 = compute_true_centroid(self.XY_positions2)
        return {
            "substep_index": int(substep_index),
            "xy1": np.array(self.XY_positions1, copy=True),
            "xy2": np.array(self.XY_positions2, copy=True),
            "state1": np.array(self.state1, copy=True),
            "state2": np.array(self.state2, copy=True),
            "centroid1": np.array(centroid1, copy=True),
            "centroid2": np.array(centroid2, copy=True),
        }

    def _record_macro_step(self):
        global traj
        global traj2
        global trajp

        combined_state = np.concatenate((self.state1.copy(), self.state2.copy()), axis=0)
        summary_row = np.array(
            [
                self.state1[0],
                self.state1[1],
                self.state2[0],
                self.state2[1],
                self.last_robot_rewards[0],
                self.last_robot_rewards[1],
                float(self.order1),
                float(self.order2),
                self.con1,
                self.con2,
            ],
            dtype=np.float64,
        )

        traj = _stack_trace(traj, combined_state)
        traj2 = _stack_trace(traj2, summary_row)

        if self.ep_step > 0 and self.ep_step % 100 == 0:
            path1 = os.path.join(directory_path, "traj")
            path2 = os.path.join(directory_path, "traj2")
            os.makedirs(path1, exist_ok=True)
            os.makedirs(path2, exist_ok=True)

            np.savetxt(os.path.join(path1, f"traj_{len(os.listdir(path1))}.pt"), traj, delimiter=",")
            np.savetxt(os.path.join(path2, f"traj2_{len(os.listdir(path2))}.pt"), traj2, delimiter=",")

            traj = []
            traj2 = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_dict):
        self.it += 1
        self.ep_step += 1
        self.reward = 0.0

        # ========== 论文两步法：粗选 + RL 微调 ==========
        # 上一步（或 reset）已粗选出 aprm1/aprm2 并作为观测返回给 RL
        # RL 返回 aadj ∈ {0,1,2} 映射到 {-1, 0, +1}
        action_dict = action_dict or {}
        aadj1 = int(action_dict.get(ROBOT_IDS[0], 1)) - 1   # {-1, 0, +1}
        aadj2 = int(action_dict.get(ROBOT_IDS[1], 1)) - 1

        # 微调: a'prm = (aprm + aadj) % 6
        self.order1 = (self.aprm1 + aadj1 + NUM_STRATEGIES) % NUM_STRATEGIES
        self.order2 = (self.aprm2 + aadj2 + NUM_STRATEGIES) % NUM_STRATEGIES
        self.last_robot_orders = [self.order1, self.order2]

        self.last_substep_frames = []

        # 底层执行子循环
        for substep_index in range(self.low_level_hold_steps):
            action1 = self._compute_low_level_action(0, self.order1)
            action2 = self._compute_low_level_action(1, self.order2)
            self._apply_dual_solver(action1, action2)
            self.low_level_step_count += 1
            self.last_substep_frames.append(self._capture_substep_frame(substep_index + 1))

        # 奖励：铰链平均浓度变化量 × 缩放系数（论文: hinge-averaged concentration）
        con1_new = float(np.mean(compute_concentration(self.XY_positions1)))
        con2_new = float(np.mean(compute_concentration(self.XY_positions2)))
        reward1 = (con1_new - self.con1) * CON_REWARD_SCALE
        reward2 = (con2_new - self.con2) * CON_REWARD_SCALE
        self.con1 = con1_new
        self.con2 = con2_new

        centroid1 = compute_true_centroid(self.XY_positions1)
        centroid2 = compute_true_centroid(self.XY_positions2)
        self.last_centroid1 = np.array(centroid1, dtype=np.float64)
        self.last_centroid2 = np.array(centroid2, dtype=np.float64)
        self.trace1.append(np.array(centroid1))
        self.trace2.append(np.array(centroid2))

        self.last_robot_rewards = [reward1, reward2]

        # ========== 为下一步做新的粗选 ==========
        self.aprm1 = coarse_select_order(self.XY_positions1)
        self.aprm2 = coarse_select_order(self.XY_positions2)

        print(
            f"[Macro {self.ep_step:>3d}] "
            f"R1: aprm={self.aprm1}→order={self.order1}({STRATEGY_NAMES[self.order1]}), "
            f"reward={reward1:>8.4f}, con={self.con1:>8.4f}, "
            f"pos=({centroid1[0]:>8.4f}, {centroid1[1]:>8.4f}) | "
            f"R2: aprm={self.aprm2}→order={self.order2}({STRATEGY_NAMES[self.order2]}), "
            f"reward={reward2:>8.4f}, con={self.con2:>8.4f}, "
            f"pos=({centroid2[0]:>8.4f}, {centroid2[1]:>8.4f})"
        )

        self._record_macro_step()

        if self.ep_step >= self.macro_horizon:
            self.done = True

        # 观测 = 新粗选结果（下一步 RL 用来决定微调方向）
        obs = self._get_obs()
        rewards = {
            ROBOT_IDS[0]: float(reward1),
            ROBOT_IDS[1]: float(reward2),
        }
        dones = {robot_id: self.done for robot_id in ROBOT_IDS}
        dones["__all__"] = self.done
        infos = {
            ROBOT_IDS[0]: {
                "reward": float(reward1),
                "concentration": float(self.con1),
                "aprm": int(self.aprm1),
                "order": int(self.order1),
                "strategy": STRATEGY_NAMES[self.order1],
            },
            ROBOT_IDS[1]: {
                "reward": float(reward2),
                "concentration": float(self.con2),
                "aprm": int(self.aprm2),
                "order": int(self.order2),
                "strategy": STRATEGY_NAMES[self.order2],
            },
        }
        return obs, rewards, dones, infos

    def reset(self):
        if self.episode_count == 0 or not self.reset_free:
            # 硬重置：回到初始位置（训练时默认）
            self._build_initial_geometry()
        else:
            # reset-free：不重置位置，只重新粗选（可视化回放时用）
            self.aprm1 = coarse_select_order(self.XY_positions1)
            self.aprm2 = coarse_select_order(self.XY_positions2)
            self.order1 = self.aprm1
            self.order2 = self.aprm2
            self.con1 = float(np.mean(compute_concentration(self.XY_positions1)))
            self.con2 = float(np.mean(compute_concentration(self.XY_positions2)))
            self.last_robot_orders = [self.order1, self.order2]

        self.reward = 0.0
        self.done = False
        self.ep_step = 0
        self.episode_count += 1
        return self._get_obs()

    def render(self):
        return None
