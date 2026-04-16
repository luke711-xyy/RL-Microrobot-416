import argparse
from pathlib import Path
import time
import math

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from swimmer import swimmer_gym, build_unbias_reward_config
from calculate_v import set_solver_num_threads, get_solver_num_threads


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_DIR / 'experiments'
TRAINING_ITERATIONS = 2000
CHECKPOINT_INTERVAL = 10


def parse_args():
    parser = argparse.ArgumentParser(description='Unbias Propel Trainer')
    parser.add_argument('--cpus', type=int, default=None, help='Ray CPU resource count for this training run')
    parser.add_argument('--threads', type=int, default=None, help='Torch solver thread count for physics solves')
    return parser.parse_args()


class UnbiasPropelCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data['pressure_reward'] = []
        episode.user_data['direction_reward'] = []
        episode.user_data['anchor_direction_reward'] = []
        episode.user_data['direction_weight'] = []
        episode.user_data['anchor_direction_weight'] = []
        episode.user_data['recent_displacement'] = []
        episode.user_data['previous_displacement'] = []
        episode.user_data['anchor_displacement'] = []
        episode.user_data['signed_direction_error_deg'] = []
        episode.user_data['anchor_signed_error_deg'] = []
        episode.user_data['position_x'] = []
        episode.user_data['position_y'] = []
        episode.user_data['global_step'] = []
        episode.user_data['reset_ep'] = []

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        try:
            info = episode.last_info_for()
        except TypeError:
            info = episode.last_info_for(None)
        if not info:
            return

        episode.user_data['pressure_reward'].append(float(info.get('pressure_reward', 0.0)))
        episode.user_data['direction_reward'].append(float(info.get('direction_reward', 0.0)))
        episode.user_data['anchor_direction_reward'].append(float(info.get('anchor_direction_reward', 0.0)))
        episode.user_data['direction_weight'].append(float(info.get('direction_weight', 0.0)))
        episode.user_data['anchor_direction_weight'].append(float(info.get('anchor_direction_weight', 0.0)))
        episode.user_data['recent_displacement'].append(float(info.get('recent_displacement', 0.0)))
        episode.user_data['previous_displacement'].append(float(info.get('previous_displacement', 0.0)))
        episode.user_data['anchor_displacement'].append(float(info.get('anchor_displacement', 0.0)))
        episode.user_data['signed_direction_error_deg'].append(math.degrees(float(info.get('signed_direction_error', 0.0))))
        episode.user_data['anchor_signed_error_deg'].append(math.degrees(float(info.get('anchor_signed_error', 0.0))))
        episode.user_data['position_x'].append(float(info.get('position_x', 0.0)))
        episode.user_data['position_y'].append(float(info.get('position_y', 0.0)))
        episode.user_data['global_step'].append(int(info.get('global_step', 0)))
        episode.user_data['reset_ep'].append(int(info.get('reset_ep', 0)))

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        def safe_mean(values):
            return sum(values) / len(values) if values else 0.0

        def safe_std(values):
            if not values:
                return 0.0
            mean_value = safe_mean(values)
            return (sum((value - mean_value) ** 2 for value in values) / len(values)) ** 0.5

        pressure_reward_mean = safe_mean(episode.user_data['pressure_reward'])
        direction_reward_mean = safe_mean(episode.user_data['direction_reward'])
        anchor_direction_reward_mean = safe_mean(episode.user_data['anchor_direction_reward'])
        direction_weight_mean = safe_mean(episode.user_data['direction_weight'])
        anchor_direction_weight_mean = safe_mean(episode.user_data['anchor_direction_weight'])
        recent_displacement_mean = safe_mean(episode.user_data['recent_displacement'])
        previous_displacement_mean = safe_mean(episode.user_data['previous_displacement'])
        anchor_displacement_mean = safe_mean(episode.user_data['anchor_displacement'])
        signed_err_deg_mean = safe_mean(episode.user_data['signed_direction_error_deg'])
        signed_err_deg_std = safe_std(episode.user_data['signed_direction_error_deg'])
        anchor_err_deg_mean = safe_mean(episode.user_data['anchor_signed_error_deg'])
        final_pos_x = episode.user_data['position_x'][-1] if episode.user_data['position_x'] else 0.0
        final_pos_y = episode.user_data['position_y'][-1] if episode.user_data['position_y'] else 0.0
        global_step = episode.user_data['global_step'][-1] if episode.user_data['global_step'] else 0
        reset_ep = episode.user_data['reset_ep'][-1] if episode.user_data['reset_ep'] else 0

        episode.custom_metrics['pressure_reward_mean'] = pressure_reward_mean
        episode.custom_metrics['direction_reward_mean'] = direction_reward_mean
        episode.custom_metrics['anchor_direction_reward_mean'] = anchor_direction_reward_mean
        episode.custom_metrics['direction_weight_mean'] = direction_weight_mean
        episode.custom_metrics['anchor_direction_weight_mean'] = anchor_direction_weight_mean
        episode.custom_metrics['recent_displacement_mean'] = recent_displacement_mean
        episode.custom_metrics['previous_displacement_mean'] = previous_displacement_mean
        episode.custom_metrics['anchor_displacement_mean'] = anchor_displacement_mean
        episode.custom_metrics['signed_direction_error_deg_mean'] = signed_err_deg_mean
        episode.custom_metrics['signed_direction_error_deg_std'] = signed_err_deg_std
        episode.custom_metrics['anchor_signed_error_deg_mean'] = anchor_err_deg_mean

        print(
            f'[EpisodeEnd | ResetEp {reset_ep} | GlobalStep {global_step}]\n'
            f'len={episode.length}\n'
            f'ep_reward={episode.total_reward:.2f}\n'
            f'final_pos=({final_pos_x:.2f}, {final_pos_y:.2f})\n'
            f'pressure_r_mean={pressure_reward_mean:.4f}\n'
            f'direction_r_mean={direction_reward_mean:.4f}\n'
            f'anchor_dir_r_mean={anchor_direction_reward_mean:.4f}\n'
            f'direction_w_mean={direction_weight_mean:.4f}\n'
            f'anchor_dir_w_mean={anchor_direction_weight_mean:.4f}\n'
            f'recent_disp_mean={recent_displacement_mean:.4f}\n'
            f'prev_disp_mean={previous_displacement_mean:.4f}\n'
            f'anchor_disp_mean={anchor_displacement_mean:.4f}\n'
            f'signed_dir_err_deg_mean={signed_err_deg_mean:.2f}\n'
            f'signed_dir_err_deg_std={signed_err_deg_std:.2f}\n'
            f'anchor_err_deg_mean={anchor_err_deg_mean:.2f}'
        )


def build_config(output_dir):
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_gpus'] = 0
    config['num_workers'] = 0
    config['num_rollout_workers'] = 0
    config['framework'] = 'torch'
    config['gamma'] = 0.995
    config['lr'] = 0.0003
    config['horizon'] = 2000
    config['evaluation_duration'] = 10000000

    config['lr_schedule'] = None
    config['use_critic'] = True
    config['use_gae'] = True
    config['lambda_'] = 0.95
    config['kl_coeff'] = 0.2
    config['sgd_minibatch_size'] = 64
    config['train_batch_size'] = 2000
    config['num_sgd_iter'] = 30
    config['shuffle_sequences'] = True
    config['vf_loss_coeff'] = 1.0
    config['entropy_coeff'] = 0.01
    config['entropy_coeff_schedule'] = None
    config['clip_param'] = 0.1
    config['vf_clip_param'] = 100000
    config['grad_clip'] = None
    config['kl_target'] = 0.01
    config['evaluation_interval'] = 1000000
    config['evaluation_duration'] = 1
    config['use_lstm'] = True
    config['max_seq_len'] = 20
    config['min_sample_timesteps_per_iteration'] = 2000
    config['callbacks'] = UnbiasPropelCallbacks

    config['env_config'] = {
        'output_dir': str(output_dir),
        'save_trajectories': True,
        **build_unbias_reward_config(),
    }
    return config


def main():
    args = parse_args()
    set_solver_num_threads(args.threads)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    experiment_name = f'unbias_propel_{timestamp}'
    base_dir = EXPERIMENTS_DIR / experiment_name
    policy_dir = base_dir / 'policy'

    print('========================================')
    print('当前任务: 无偏直线推进 (Unbias Propel)')
    print('奖励 = 压力奖励 + 近期方向惩罚(recent vs prev) + 锚点方向惩罚(recent vs anchor)')
    print('惩罚幅度与 pressure 大小和位移大小形成比例关联')
    print(f'CLI resources: cpus={args.cpus}, solver_threads={get_solver_num_threads()}')
    print('观测 = 9 个关节角')
    print('环境 = reset-free，只切 episode 统计，不重置机器人位置/姿态')
    print(f'实验输出目录:\n{base_dir}')
    print('========================================')
    print(f'无偏奖励配置: {build_unbias_reward_config()}')

    for folder in (policy_dir, base_dir / 'traj', base_dir / 'traj2', base_dir / 'trajp'):
        folder.mkdir(parents=True, exist_ok=True)

    ray_kwargs = {'ignore_reinit_error': True}
    if args.cpus is not None:
        ray_kwargs['num_cpus'] = args.cpus
    ray.init(**ray_kwargs)
    trainer = None
    try:
        config = build_config(base_dir)
        trainer = ppo.PPO(config=config, env=swimmer_gym)

        for i in range(TRAINING_ITERATIONS):
            result = trainer.train()
            ep_reward_mean = result.get('episode_reward_mean', 0.0)
            ep_reward_min = result.get('episode_reward_min', 0.0)
            ep_reward_max = result.get('episode_reward_max', 0.0)
            ep_len_mean = result.get('episode_len_mean', 0.0)
            learner_stats = result.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {})
            total_loss = learner_stats.get('total_loss', 0.0)
            policy_loss = learner_stats.get('policy_loss', 0.0)
            vf_loss = learner_stats.get('vf_loss', 0.0)
            entropy = learner_stats.get('entropy', 0.0)
            sampled_steps = result.get('num_env_steps_sampled_this_iter', result.get('num_agent_steps_sampled_this_iter', 0))
            completed_eps = result.get('episodes_this_iter', 0)
            time_this_iter = result.get('time_this_iter_s', 0.0)
            print(
                f'[IterSummary | Iter {i}] '
                f'sampled_steps={sampled_steps} '
                f'completed_eps={completed_eps} '
                f'ep_reward_mean={ep_reward_mean:.2f} '
                f'ep_reward_min={ep_reward_min:.2f} '
                f'ep_reward_max={ep_reward_max:.2f} '
                f'ep_len_mean={ep_len_mean:.0f} '
                f'total_loss={total_loss:.4f} '
                f'policy_loss={policy_loss:.4f} '
                f'vf_loss={vf_loss:.4f} '
                f'entropy={entropy:.4f} '
                f'time_this_iter_s={time_this_iter:.2f}'
            )
            if i % CHECKPOINT_INTERVAL == 0:
                checkpoint_dir = policy_dir / str(i)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                trainer.save(str(checkpoint_dir))
    finally:
        if trainer is not None:
            trainer.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()
