import gym, ray
#from gym_particle.envs.swimmer import swimmer_gym
from swimmer import swimmer_gym
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.algorithms.ppo import PPO
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import os
import numpy as np
import math
from os import path
from datetime import datetime
# class MyEnv(gym.Env):
#     def __init__(self, env_config):
#         self.action_space = <gym.Space>
#         self.observation_space = <gym.Space>
#     def reset(self):
#         return <obs>
#     def step(self, action):
#         return <obs>, <reward: float>, <done: bool>, <info: dict>
POLICY_ID = 0  # 策略编号: 0=转向(reorien)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
policy_dir_name = f"{timestamp}_policy_{POLICY_ID}"
cwd = os.path.join(os.getcwd(), policy_dir_name)
cwd2 = os.path.join(os.getcwd(),"policy/checkpoint_000000/checkpoint-0")
print(os.getcwd())
ray.init(ignore_reinit_error=True, num_cpus=3)
# trainer = ppo.PPOTrainer(env=swimmer_gym, config={
#     "env_config": {},  # config to pass to env class
# })

# while True:
#     print(trainer.train())
#     
env=swimmer_gym
#BaseEnv.to_base_env(env)
# obs, rewards, dones, infos, off_policy_actions = env.poll() 
# 
# print(obs)     


#config = PPOConfig()
config =ppo.DEFAULT_CONFIG.copy()
#config = config.training()
config["num_gpus"] = 0
config["num_workers"] = 0
config["num_rollout_workers"]=0
config["framework"]= "torch"
config['gamma']=0.9999
config['lr']=0.0003
config['horizon']=1000
config["evaluation_duration"]= 10000000

config['lr_schedule'] = None
config['use_critic']  = True
config['use_gae']= True
config['lambda_']= 0.95
config['kl_coeff']= 0.2
config['sgd_minibatch_size']= 64
config["train_batch_size"]= 1000
config['num_sgd_iter']= 30
config['shuffle_sequences']= True
config['vf_loss_coeff']= 1.0
config['entropy_coeff'] = 0.0
config['entropy_coeff_schedule'] = None
config['clip_param']=0.1
config['vf_clip_param']=100000
config['grad_clip']= None
config['kl_target']=0.01






# config['soft_horizon'] = True
# config['no_done_at_end'] =   True

config["evaluation_interval"]=1000000
config["evaluation_duration"]=1
# config["actor_hiddens"]=[100, 100]
# config["actor_hidden_activation"]="relu"
# config["critic_hiddens"]=[100, 100]
# config["critic_hidden_activation"]="relu"
config["use_lstm"]=True
config["max_seq_len"]= 20

# config[ "timesteps_per_iteration"]=500

config["min_sample_timesteps_per_iteration"]= 1000

directory_path = os.getcwd()
folder_name = path.basename(directory_path)


# N=int(int(folder_name))
# N=3
# X_ini = 0.0
# Y_ini = 0.3*N
# 
# Xfirst=np.zeros((2),dtype=np.float64)
# Xfirst[1]=  Y_ini      
# state = np.zeros(N+1,dtype=np.float64)
#         #self.state[0]=X_ini         
# state[0]=Y_ini 
# 
# X=0
# Y=0
# m=np.zeros((1,2))
# m[:,0]=X
# m[:,1]=Y       
# mm=np.zeros((1,2))
# mm=Xfirst.copy()
# np.savetxt('state.pt',state , delimiter=',')
# np.savetxt('XY.pt',m, delimiter=',')
# np.savetxt('Xfirst.pt',mm, delimiter=',')
#config["train_batch_size"]=100

#config["num_workers"]= 0
# config["env"]=env
# config["env_config"]={}

#config = config.resources(num_gpus=0)
#config = config.rollouts(num_rollout_workers=1)
#trainer = config.build()
#trainer = PPO()
#print(config.to_dict())   
env.__init__(env,{})
#trainer.restore(cwd2)
#trainer = config.build(env=env)
trainer= ppo.PPO(config=config, env=env)
now_path=os.getcwd()
path1 = os.path.join(now_path,'traj')
path2 = os.path.join(now_path, 'traj2')
pathp = os.path.join(now_path,'trajp')
if os.path.isdir(path1)<1:
    os.mkdir(path1)
    
if os.path.isdir(path2)<1:
    os.mkdir(path2)

if os.path.isdir(pathp)<1:
    os.mkdir(pathp)
#i=199
#print(i)
#cwd_restore = os.path.join(os.getcwd(),"policy2",str(i),"checkpoint_000200")
#trainer.restore(cwd_restore)

for i in range(2000):
    print(i)
#     env.reset(env)
    #print(env.X)
#     trainer.config["critic_lr"]=(1e-3)*(1-i/100)
#     trainer.config["actor_lr"]=(1e-3)*(1-i/100)    
#     trainer.config["tau"]=(1e-3)*(1-i/50)   
#     if i>0 and i%50==0:
#         env.__init__(env,{})
#         trainer=None
#         trainer= ppo.PPO(config=config, env=env)    
    result = trainer.train()
    if i%10==0:
        path = os.path.join(cwd, str(i))
        if os.path.isdir(path)<0:
            os.mkdir(path)
        checkpoint = trainer.save(path)
#     print(i)
#     trainer.evaluate()
    #print("checkpoint saved at", checkpoint)
#trainer.export_policy_model(cwd)
