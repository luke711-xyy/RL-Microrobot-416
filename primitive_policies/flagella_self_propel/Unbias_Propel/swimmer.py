from collections import deque
from pathlib import Path
import math

import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding

from calculate_v import RK


BASE_DIR = Path(__file__).resolve().parent

N = 10
DT = 0.01
ACTION_LOW = -1
ACTION_HIGH = 1

traj = []
traj2 = []
trajp = []
d = torch.device('cpu')
dtype = torch.double


def build_unbias_reward_config():
    return {
        'pressure_reward_weight': 10.0,
        'pressure_reward_coupling_ref': 80.0,
        'direction_error_weight_max': 6.0,
        'direction_weight_recent_ref': 0.005,
        'direction_weight_prev_ref': 0.01,
        'anchor_direction_error_weight_max': 8.0,
        'anchor_direction_weight_ref': 0.015,
        'recent_window': 30,
        'prev_window': 60,
        'anchor_window': 150,
        'reset_free': True,
    }


class swimmer_gym(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 30,
    }

    def __init__(self, env_config):
        env_config = env_config or {}
        reward_defaults = build_unbias_reward_config()

        self.betamax = (2 * math.pi) / N
        self.betamin = -self.betamax * 0.5
        self.dt = DT
        self.action_space = spaces.Box(low=-1, high=1, shape=((N - 1),), dtype=np.float64)
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(N - 1,), dtype=np.float64)

        self.pressure_reward_weight = env_config.get('pressure_reward_weight', reward_defaults['pressure_reward_weight'])
        self.pressure_reward_coupling_ref = env_config.get('pressure_reward_coupling_ref', reward_defaults['pressure_reward_coupling_ref'])
        self.direction_error_weight_max = env_config.get('direction_error_weight_max', reward_defaults['direction_error_weight_max'])
        self.direction_weight_recent_ref = env_config.get('direction_weight_recent_ref', reward_defaults['direction_weight_recent_ref'])
        self.direction_weight_prev_ref = env_config.get('direction_weight_prev_ref', reward_defaults['direction_weight_prev_ref'])
        self.anchor_direction_error_weight_max = env_config.get('anchor_direction_error_weight_max', reward_defaults['anchor_direction_error_weight_max'])
        self.anchor_direction_weight_ref = env_config.get('anchor_direction_weight_ref', reward_defaults['anchor_direction_weight_ref'])
        self.recent_window = env_config.get('recent_window', reward_defaults['recent_window'])
        self.prev_window = env_config.get('prev_window', reward_defaults['prev_window'])
        self.anchor_window = env_config.get('anchor_window', reward_defaults['anchor_window'])
        self.reset_free = env_config.get('reset_free', reward_defaults['reset_free'])

        self.output_dir = Path(env_config.get('output_dir', BASE_DIR))
        self.save_trajectories = env_config.get('save_trajectories', True)
        if self.save_trajectories:
            self._ensure_output_dirs()

        self.viewer = None
        self.X_ini = -4.0
        self.Y_ini = 4.0
        self.X = 0.0
        self.reward = 0.0
        self.done = False
        self.pressure_diff = 0.0
        self.episode_reward = 0.0
        self.episode_count = 0
        self.episode_step_count = 0
        self.it = 0
        self.istore = 0
        self.order = 0

        self.state = np.zeros((N + 2), dtype=np.float64)
        self.state[0] = self.X_ini
        self.state[1] = self.Y_ini
        self.state[2] = 0.0

        self.Xfirst = np.zeros((2), dtype=np.float64)
        self.Xfirst[0] = self.X_ini - 0.5 * math.cos(self.state[2])
        self.Xfirst[1] = self.Y_ini - 0.5 * math.sin(self.state[2])

        self.last_pressure_reward = 0.0
        self.last_direction_reward = 0.0
        self.last_anchor_direction_reward = 0.0
        self.last_abs_direction_error = 0.0
        self.last_signed_direction_error = 0.0
        self.last_anchor_abs_error = 0.0
        self.last_anchor_signed_error = 0.0
        self.last_recent_direction_angle = 0.0
        self.last_previous_direction_angle = 0.0
        self.last_anchor_direction_angle = 0.0
        self.last_recent_displacement = 0.0
        self.last_previous_displacement = 0.0
        self.last_anchor_displacement = 0.0
        self.last_direction_weight = 0.0
        self.last_anchor_direction_weight = 0.0
        self.last_motion_angle = 0.0
        self.last_limit_clipped = False

        self.centroid_history = deque(maxlen=self.recent_window + self.prev_window + self.anchor_window + 1)
        self.centroid_history.append(self.state[:2].copy())

        self.XY_positions = self._build_robot_positions(self.state[2], self.Xfirst)
        self.pressure_index = 1
        self.escape_count = 0

    def _ensure_output_dirs(self):
        for folder in ('traj', 'traj2', 'trajp'):
            (self.output_dir / folder).mkdir(parents=True, exist_ok=True)

    def angle_normalize(self, value):
        return ((value + math.pi) % (2 * math.pi)) - math.pi

    def _build_robot_positions(self, initial_angle, x_first):
        Xp = np.zeros((N + 1), dtype=np.float64)
        Yp = np.zeros((N + 1), dtype=np.float64)
        for i in range(N + 1):
            Xp[i] = x_first[0] + i / N * math.cos(initial_angle)
            Yp[i] = x_first[1] + i / N * math.sin(initial_angle)
        return np.concatenate(((Xp).reshape(-1, 1), (Yp).reshape(-1, 1)), axis=1)

    def _compute_window_direction_feedback(self):
        points = np.asarray(self.centroid_history, dtype=np.float64)
        transition_count = len(points) - 1
        if transition_count <= 0:
            return (0.0,) * 13

        # Recent window
        recent_span = min(self.recent_window, transition_count)
        # Prev window
        prev_available = max(transition_count - recent_span, 0)
        prev_span = min(self.prev_window, prev_available)
        # Anchor window
        anchor_span = min(self.anchor_window, transition_count)

        end_idx = len(points) - 1

        # Recent direction
        recent_start_idx = end_idx - recent_span
        recent_vec = points[end_idx] - points[recent_start_idx]
        recent_disp = float(np.linalg.norm(recent_vec))
        recent_angle = math.atan2(float(recent_vec[1]), float(recent_vec[0]))

        # Previous direction
        if prev_span > 0:
            prev_end_idx = recent_start_idx
            prev_start_idx = prev_end_idx - prev_span
            prev_vec = points[prev_end_idx] - points[prev_start_idx]
            prev_disp = float(np.linalg.norm(prev_vec))
            previous_angle = math.atan2(float(prev_vec[1]), float(prev_vec[0]))
        else:
            prev_disp = 0.0
            previous_angle = recent_angle

        # Anchor direction
        anchor_start_idx = end_idx - anchor_span
        anchor_vec = points[end_idx] - points[anchor_start_idx]
        anchor_disp = float(np.linalg.norm(anchor_vec))
        anchor_angle = math.atan2(float(anchor_vec[1]), float(anchor_vec[0]))

        # Signed errors
        signed_error = self.angle_normalize(recent_angle - previous_angle)
        abs_error = abs(signed_error)
        anchor_signed_error = self.angle_normalize(recent_angle - anchor_angle)
        anchor_abs_error = abs(anchor_signed_error)

        # Displacement-based scales
        recent_scale = 1.0 if self.direction_weight_recent_ref <= 1e-12 else float(np.clip(recent_disp / self.direction_weight_recent_ref, 0.0, 1.0))
        previous_scale = 1.0 if self.direction_weight_prev_ref <= 1e-12 else float(np.clip(prev_disp / self.direction_weight_prev_ref, 0.0, 1.0))
        anchor_scale = 1.0 if self.anchor_direction_weight_ref <= 1e-12 else float(np.clip(anchor_disp / self.anchor_direction_weight_ref, 0.0, 1.0))

        direction_base_weight = float(self.direction_error_weight_max * recent_scale * previous_scale)
        anchor_direction_base_weight = float(self.anchor_direction_error_weight_max * recent_scale * anchor_scale)

        return (abs_error, signed_error, recent_angle, previous_angle, anchor_angle,
                recent_disp, prev_disp, anchor_disp,
                direction_base_weight, anchor_direction_base_weight,
                anchor_abs_error, anchor_signed_error,
                recent_scale)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.it += 1
        self.episode_step_count += 1
        global traj
        global traj2
        global trajp

        self.reward = 0.0

        if self.order >= 0:
            actionx = action.copy()
        else:
            actionx = -action.copy()

        w_tmp = np.clip(actionx, ACTION_LOW, ACTION_HIGH)

        state_predict = self.state.copy()
        state_predict[3:] += w_tmp * 0.2
        reward_limit_penalty = 0.0
        limit_clipped = False
        if self.order == 1:
            for i in range(N - 1):
                if state_predict[i + 3] > self.betamax or state_predict[i + 3] < self.betamin:
                    w_tmp = np.clip(actionx, 0, 0)
                    reward_limit_penalty = -1.0
                    limit_clipped = True
                    break
        elif self.order == -1:
            for i in range(N - 1):
                if state_predict[i + 3] < -self.betamax or state_predict[i + 3] > -self.betamin:
                    w_tmp = np.clip(actionx, 0, 0)
                    reward_limit_penalty = -1.0
                    limit_clipped = True
                    break
        else:
            for i in range(N - 1):
                if abs(state_predict[i + 3]) > self.betamax:
                    w_tmp = np.clip(actionx, 0, 0)
                    reward_limit_penalty = -1.0
                    limit_clipped = True
                    break

        self.Xn = self.X + 0.0
        self.state_n = self.state.copy()

        staten, Xn, r, x_first_delta, Xpositions, Ypositions, pressure_diff, pressure_end, pressure_all = RK(self.state_n, w_tmp, self.Xfirst)
        self.pressure_diff += pressure_diff
        self.state_n = staten.copy()
        self.XY_positions = np.concatenate((np.array(Xpositions).reshape(-1, 1), np.array(Ypositions).reshape(-1, 1)), axis=1)

        self.Xn += Xn
        self.Xfirst += x_first_delta
        self.state = self.state_n.copy()
        self.X = self.Xn
        self.pressure_end = pressure_end

        # Update centroid history
        self.centroid_history.append(self.state[:2].copy())

        # Compute direction feedback
        (abs_direction_error, signed_direction_error, recent_direction_angle, previous_direction_angle, anchor_direction_angle,
         recent_disp, previous_disp, anchor_disp,
         direction_base_weight, anchor_direction_base_weight,
         anchor_abs_error, anchor_signed_error,
         recent_scale) = self._compute_window_direction_feedback()

        # Pressure reward
        pressure_reward = float(pressure_diff.item() * self.pressure_reward_weight)

        # Pressure-based coupling scale
        penalty_pressure_scale = float(np.clip(abs(pressure_reward) / self.pressure_reward_coupling_ref, 0.0, 1.0))

        # Direction weights with pressure coupling
        direction_weight = direction_base_weight * penalty_pressure_scale
        anchor_direction_weight = anchor_direction_base_weight * penalty_pressure_scale

        # Direction penalties
        recent_error_fraction = float(np.clip(abs_direction_error / math.pi, 0.0, 1.0))
        anchor_error_fraction = float(np.clip(anchor_abs_error / math.pi, 0.0, 1.0))
        direction_reward = -direction_weight * recent_error_fraction
        anchor_direction_reward = -anchor_direction_weight * anchor_error_fraction

        # Limit clipping overrides
        if limit_clipped:
            pressure_reward = 0.0
            direction_reward = 0.0
            anchor_direction_reward = 0.0
            direction_weight = 0.0
            anchor_direction_weight = 0.0
            reward = -1.0
        else:
            reward = pressure_reward + direction_reward + anchor_direction_reward

        self.reward += reward
        self.episode_reward += reward

        # Store diagnostics
        self.last_pressure_reward = pressure_reward
        self.last_direction_reward = direction_reward
        self.last_anchor_direction_reward = anchor_direction_reward
        self.last_abs_direction_error = abs_direction_error
        self.last_signed_direction_error = signed_direction_error
        self.last_anchor_abs_error = anchor_abs_error
        self.last_anchor_signed_error = anchor_signed_error
        self.last_recent_direction_angle = recent_direction_angle
        self.last_previous_direction_angle = previous_direction_angle
        self.last_anchor_direction_angle = anchor_direction_angle
        self.last_recent_displacement = recent_disp
        self.last_previous_displacement = previous_disp
        self.last_anchor_displacement = anchor_disp
        self.last_direction_weight = direction_weight
        self.last_anchor_direction_weight = anchor_direction_weight
        self.last_motion_angle = 0.0
        self.last_limit_clipped = limit_clipped

        # Trajectory saving
        if self.save_trajectories:
            m = np.zeros((1, 4))
            m[:, 0] = self.state[0]
            m[:, 1] = self.state[1]
            m[:, 2] = self.Xfirst[0]
            m[:, 3] = self.Xfirst[1]

            if traj == []:
                traj = self.state_n.copy()
            else:
                traj = np.concatenate((traj.reshape(-1, N + 2), self.state_n.reshape(1, -1)), axis=0)

            if traj2 == []:
                traj2 = m
            else:
                traj2 = np.concatenate((traj2.reshape(-1, 4), m.reshape(1, -1)), axis=0)

            if trajp == []:
                trajp = pressure_all.reshape(1, -1)
            else:
                trajp = np.concatenate((trajp, pressure_all.reshape(1, -1)), axis=0)

            if self.it % 4000 == 0:
                path1 = self.output_dir / 'traj' / f'traj_{self.istore}.pt'
                path2 = self.output_dir / 'traj2' / f'traj2_{self.istore}.pt'
                pathp = self.output_dir / 'trajp' / f'trajp_{self.istore}.pt'
                np.savetxt(path1, traj, delimiter=',')
                np.savetxt(path2, traj2, delimiter=',')
                np.savetxt(pathp, trajp, delimiter=',')
                self.istore += 1
                traj = []
                traj2 = []
                trajp = []

        if self.it % 100 == 0:
            print(
                f'[Progress | GlobalStep {self.it} | ResetEp {self.episode_count}] '
                f'ep_step={self.episode_step_count} '
                f'pos=({self.state[0]:.2f},{self.state[1]:.2f}) '
                f'pressure_r={self.last_pressure_reward:.4f} '
                f'direction_r={self.last_direction_reward:.4f} '
                f'anchor_dir_r={self.last_anchor_direction_reward:.4f} '
                f'direction_w={self.last_direction_weight:.4f} '
                f'anchor_dir_w={self.last_anchor_direction_weight:.4f} '
                f'recent_disp={self.last_recent_displacement:.4f} '
                f'prev_disp={self.last_previous_displacement:.4f} '
                f'anchor_disp={self.last_anchor_displacement:.4f} '
                f'signed_dir_err_deg={math.degrees(self.last_signed_direction_error):.2f} '
                f'anchor_err_deg={math.degrees(self.last_anchor_signed_error):.2f} '
                f'ep_rwd_so_far={self.episode_reward:.2f}'
            )

        info = {
            'pressure_reward': float(self.last_pressure_reward),
            'direction_reward': float(self.last_direction_reward),
            'anchor_direction_reward': float(self.last_anchor_direction_reward),
            'direction_weight': float(self.last_direction_weight),
            'anchor_direction_weight': float(self.last_anchor_direction_weight),
            'recent_displacement': float(self.last_recent_displacement),
            'previous_displacement': float(self.last_previous_displacement),
            'anchor_displacement': float(self.last_anchor_displacement),
            'signed_direction_error': float(self.last_signed_direction_error),
            'abs_direction_error': float(self.last_abs_direction_error),
            'anchor_signed_error': float(self.last_anchor_signed_error),
            'anchor_abs_error': float(self.last_anchor_abs_error),
            'recent_direction_angle': float(self.last_recent_direction_angle),
            'previous_direction_angle': float(self.last_previous_direction_angle),
            'anchor_direction_angle': float(self.last_anchor_direction_angle),
            'global_step': int(self.it),
            'reset_ep': int(self.episode_count),
            'episode_step': int(self.episode_step_count),
            'position_x': float(self.state[0]),
            'position_y': float(self.state[1]),
            'reset_free': bool(self.reset_free),
        }

        if self.order >= 0:
            return self.state[3:], float(self.reward), self.done, info
        else:
            return -self.state[3:], float(self.reward), self.done, info

    def reset(self):
        self.reward = 0.0
        self.done = False
        self.pressure_diff = 0.0
        self.episode_reward = 0.0
        self.episode_count += 1
        self.episode_step_count = 0
        self.order = 0

        self.last_pressure_reward = 0.0
        self.last_direction_reward = 0.0
        self.last_anchor_direction_reward = 0.0
        self.last_direction_weight = 0.0
        self.last_anchor_direction_weight = 0.0
        self.last_recent_displacement = 0.0
        self.last_previous_displacement = 0.0
        self.last_anchor_displacement = 0.0
        self.last_abs_direction_error = 0.0
        self.last_signed_direction_error = 0.0
        self.last_anchor_abs_error = 0.0
        self.last_anchor_signed_error = 0.0
        self.last_recent_direction_angle = 0.0
        self.last_previous_direction_angle = 0.0
        self.last_anchor_direction_angle = 0.0
        self.last_motion_angle = 0.0
        self.last_limit_clipped = False

        if self.order >= 0:
            return self.state[3:]
        else:
            return -self.state[3:]

    def _get_obs(self):
        if self.order >= 0:
            return self.state[3:]
        else:
            return -self.state[3:]

    def render(self):
        return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
