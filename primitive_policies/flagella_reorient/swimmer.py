import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import os
from os import path
import math
import numpy.matlib
from math import sin
from math import cos
import random
import torch
#import numba
#from numba import jit, njit
from calculate_v import RK
directory_path = os.getcwd()
folder_name = path.basename(directory_path)


#N=int(int(folder_name))
N=10
M=1
K_RWD = 4;
L=1
DIST_EPSI_GOAL = 0.004
COEF_EXCEED=-0.1
DIST_EPSI_INLET=0.05 # need to prevent the particle from too close to the inlets.
MAX_STEP=10000
DT = 0.01

ACTION_LOW = -1
ACTION_HIGH = 1
ACTION_MEAN = (ACTION_LOW + ACTION_HIGH)/2



#theta_ini=M/18.0*math.pi
theta_ini1=0
theta_ini2=0
# beta1_ini=0
# beta2_ini=0
# beta3_ini=0
# 2.6589477  0.6732641  0.67776    0.6898408  0.68705076 0.7361291
#  0.733033   0.6177958  0.6838072
beta1_ini1=0
beta2_ini1=0

beta1_ini2=0
beta2_ini2=0







traj=[]
traj2=[]
traj3=[]
traj4=[]
trajp=[]
d = torch.device("cpu")
dtype = torch.double






class swimmer_gym(gym.Env):
    metadata = {
        'render.modes' : ['human'],
        'video.frames_per_second' : 30
    }
   
    def __init__(self, env_config):

        self.max_rate = 1
#         self.betamax = (1.0*math.pi)/6.0
#         self.betamin = -(1.0*math.pi)/6.0
        self.betamax = (2*math.pi)/(N)
        self.betamin = -self.betamax*0.5
        
        self.dt=DT
        self.action_space = spaces.Box(low=-1, high=1, shape=((N-1),), dtype=np.float64)
        self.observation_space = spaces.Box(low=-10000, high=10000,shape=(N-1,), dtype=np.float64)
        self.viewer = None
        self.X = 0
        self.Y = 0

#         self.theta_ini1=theta_ini1
#         self.beta1_ini1=beta1_ini1
#         self.beta2_ini1=beta2_ini1
#         self.theta_ini2=theta_ini2
#         self.beta1_ini2=beta1_ini2
#         self.beta2_ini2=beta2_ini2
        #self.state = np.array([X_ini1,Y_ini1,X_ini2,Y_ini2,theta_ini1,beta1_ini1,beta2_ini1,theta_ini2,beta1_ini2,beta2_ini2],dtype=np.float64)
#         self.state = np.zeros((N+2),dtype=np.float64)
#         self.X_ini=-2
#         self.Y_ini=3.5
        self.X_ini= -4    
        self.Y_ini=4
        
        
#         self.state=np.loadtxt('state.pt', delimiter=',')
#         XY=np.loadtxt('XY.pt', delimiter=',')
#         self.X=XY[0]
#         self.Y=XY[1]
#         self.Xfirst=np.loadtxt('Xfirst.pt', delimiter=',')
#         self.Xfirst=np.squeeze(self.Xfirst) 
        #self.it=0
#         self.X=0
        self.reward=0
        self.state = np.zeros((N+2),dtype=np.float64)
        self.state[0]=(self.X_ini)        
        self.state[1]=self.Y_ini
        self.state[2]= 0 
        self.Xfirst=np.zeros((2),dtype=np.float64)
        self.Xfirst[0]=  self.X_ini -1/2*math.cos(self.state[2])       
        self.Xfirst[1]=  self.Y_ini -1/2*math.sin(self.state[2])        
        
        #self.state[0]= (X_ini+N/2)
        #self.state[1]= Y_ini        
        
#         self.state[0]=X_ini1         
#         self.state[1]=Y_ini1         
#         self.state[2]=X_ini2
#         self.state[3]=Y_ini2        
        self.dist_epsi_goal = DIST_EPSI_GOAL
        self.ACTION_LOW = ACTION_LOW
        self.ACTION_HIGH = ACTION_HIGH
        self.ACTION_MEAN = (self.ACTION_LOW + self.ACTION_HIGH)/2.0
        #self.seed()
        #self.state = np.array([theta_ini,beta1_ini,beta2_ini,beta3_ini,self.beta4_ini,self.beta5_ini,self.beta6_ini,self.beta7_ini,self.beta8_ini],dtype=np.float64)

#        print('herer')
#        print(self.x_tar)

        self.it = 0
        self.istore = 0
        self.traj=[]
        self.traj2=[]
        self.done=False
        self.pressure_diff=0
   
        Xp=np.zeros((N+1),dtype=np.float64)
        Yp=np.zeros((N+1),dtype=np.float64)
        for i in range(N+1):
            Xp[i]=self.Xfirst[0]+i/N*math.cos(self.state[2])
            Yp[i]=self.Xfirst[1]+i/N*math.sin(self.state[2])
            
        self.XY_positions=np.concatenate(((Xp).reshape(-1,1),(Yp).reshape(-1,1)),axis=1)            
        self.pressure_index=1
        self.order=1
        self.escape_count=0
        

    

    
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
 
    def step(self,action):
#         if self.it==0:
#             self.state=np.loadtxt('state.pt', delimiter=',')
#             XY=np.loadtxt('XY.pt', delimiter=',')
#             self.X=XY[0]
#             self.Y=XY[1]
#             self.Xfirst=np.loadtxt('Xfirst.pt', delimiter=',')
#             self.Xfirst=np.squeeze(self.Xfirst) 
#             #self.it=0
#     #         self.X=0
#             self.reward=0        
        self.it += 1
#         done   = False
        reach  = False
        global traj
        global traj2
#         global traj3
        #global traj4
        global trajp        
        
        self.reward=0
#         add_reward=0        
        
        if self.order>=0:                
            actionx=action.copy()
        else:
            actionx=-action.copy()
            
            
        w_tmp  = np.clip(actionx, ACTION_LOW, ACTION_HIGH) ## compared clip, tanh is much better for control purpose.
        
        state_predict=self.state.copy()
        
        state_predict[3:]+=w_tmp*0.2
        reward=0        
        if self.order==1:
            for i in range(N-1):
                if (state_predict[i+3])>self.betamax or (state_predict[i+3])<self.betamin:            
                     w_tmp=   np.clip(actionx, 0, 0)
                     reward=-1
                     break
        elif self.order==-1:            
            for i in range(N-1):
                if (state_predict[i+3])<-self.betamax or (state_predict[i+3])>-self.betamin:            
                     w_tmp=   np.clip(actionx, 0, 0)
                     reward=-1                     
                     break            
        else:
            for i in range(N-1):
                if abs(state_predict[i+3])>self.betamax:            
                     w_tmp=   np.clip(actionx, 0, 0)
                     reward=-1
                     break    




        self.Xn=self.X+0.0

    
            #print(w_tmpx)
        self.state_n=self.state.copy()
        
        
        
        
#         print(w_tmp)
        staten ,Xn ,r,x_first_delta,Xpositions,Ypositions,pressure_diff,pressure_end,pressure_all=RK(self.state_n,w_tmp,self.Xfirst)
        self.pressure_diff+=pressure_diff
    #         if Xn>0.05:
    #             print(self.state_n,actionx)
        self.state_n=staten.copy()
        self.XY_positions=np.concatenate((np.array(Xpositions).reshape(-1,1),np.array(Ypositions).reshape(-1,1)),axis=1)
#         self.con_next=self.get_concentraion(XY_positions)
#         self.con_next=np.squeeze(self.con_next)
        #print((self.con_next*0.9999- self.con))
#         self.con=np.squeeze(self.con)        
#         add_reward= ((self.con_next[-1]*0.9999- self.con[-1]))*5000
#         self.con=self.con_next.copy()        
        #print(self.con)
        
        
        self.Xn+=Xn
        
        #dis1=math.sqrt((self.state_n[0]-self.X_ini)**2+(self.state_n[1]-self.Y_ini)**2)
        #dis2=math.sqrt((self.state[0]-self.X_ini)**2+(self.state[1]-self.Y_ini)**2)
#         if r==0:
#             reward+=-N
#         else:
#         reward+=(((r)/N)*100)      
        reward+=  pressure_diff.item()*10      
        
        self.Xfirst+=x_first_delta
            

        self.reward+=(reward)
#         self.reward+=  add_reward              
        self.state = self.state_n.copy()
        self.X = self.Xn
#         print(Xpositions,Ypositions)
#         for i in range(N+1):
#             if Ypositions[i]<0.05 or Ypositions[i]>4.95:
#                 
#                 self.done =True
#                 self.reward=-100
#                 print(self.done)
#                 self.Xn=self.X                 
#                 self.state_n=self.state.copy()                
#                 break
# 
# 
# #         for i in range(N+1):
# #             if 1-math.cos(math.pi*Xpositions[i]+math.pi)+Ypositions[i]-5>0.1 and abs(Xpositions[i])<1:
# #                 self.reward-=1
# 
#         for i in range(N+1):
#             if 1-math.cos(math.pi*Xpositions[i]+math.pi)+Ypositions[i]-5>0 and abs(Xpositions[i])<1:
#                 
#                 self.done =True
#                 self.reward=-100
#                 print(self.done)
#                 self.Xn=self.X                 
#                 self.state_n=self.state.copy()                
#                 break         
        self.pressure_end=pressure_end
#         print(self.pressure_end)
#         if abs(pressure_end)>0.9:
# #             self.done =True
# #             self.Xn=self.X                 
# #             self.state_n=self.state.copy()            
# #             self.reward=-100
#             print(pressure_end)
#             self.pressure_index=0
        m=np.zeros((1,4))
        m[:,0]=self.state[0]
        m[:,1]=self.state[1]
        #m[:,2]= self.pressure_end.item()      

        m[:,2]=self.Xfirst[0]
        m[:,3]=self.Xfirst[1]       
        #mm=self.Xfirst.copy()
        
        if traj==[]:
            traj=self.state_n.copy()
          
        else:
            traj=np.concatenate((traj.reshape(-1,N+2),self.state_n.reshape(1,-1)),axis=0)
            
        if traj2==[]:
            traj2=m
        else:
            traj2=np.concatenate((traj2.reshape(-1,4),m.reshape(1,-1)),axis=0)
            
#         if traj3==[]:
#             traj3=mm
#         else:
#             traj3=np.concatenate((traj3.reshape(-1,2),mm.reshape(1,-1)),axis=0)
#             
#         if traj4==[]:
#             traj4=action
#         else:
#             traj4=np.concatenate((traj4.reshape(-1,N-1),action.reshape(1,-1)),axis=0)
            
        if trajp==[]:
            trajp=pressure_all.reshape(1,-1)
        else:
            trajp=np.concatenate((trajp,pressure_all.reshape(1,-1)),axis=0)            
#         if     self.it%1000==0 :
#             np.savetxt('traj.pt', traj, delimiter=',')        
#             np.savetxt('traj2.pt', traj2, delimiter=',')            
#             np.savetxt('traj3.pt', traj3, delimiter=',') 
#             np.savetxt('traj4.pt', traj4, delimiter=',')
#             np.savetxt('trajp.pt', trajp, delimiter=',')            
#             np.savetxt('state.pt',self.state_n , delimiter=',')
#             np.savetxt('XY.pt',m, delimiter=',')
#             np.savetxt('Xfirst.pt',mm, delimiter=',')        
        
        if     self.it%4000==0 :
            path1 = os.path.join(directory_path , 'traj','traj_'+str(self.istore)+'.pt')
            path2 = os.path.join(directory_path , 'traj2','traj2_'+str(self.istore)+'.pt')
            pathp = os.path.join(directory_path , 'trajp','trajp_'+str(self.istore)+'.pt')            
            np.savetxt(path1, traj, delimiter=',')        
            np.savetxt(path2, traj2, delimiter=',')
            np.savetxt(pathp, trajp, delimiter=',')            
            self.istore+=1
            traj=[]
            traj2=[]
            trajp=[]           
#         self.Y = self.Yn
        #self.reward+=0.9999**(self.it-1)*reward
#         print(self.state[3:] )
        if self.it%1000==0 :
            print(self.state[:2],self.X ,self.it,self.reward)
            print(self.state)
# #         if self.it%100==0 and reward>=0:
# #             reward+=self.X*100#done = True
            #print(self.state,self.X,self.Y)
        #if self.it==1000 :
        
        #print(self.it,self.pressure_diff,pressure_end,self.state[:2])
        if self.order>=0:
            return self.state[3:],self.reward,self.done,{}
        else:
            return -self.state[3:],self.reward,self.done,{}



    def reset(self):
        


        
        self.reward=0
        

        self.order=0


           
            

        self.reward=0

        self.done=False
        
        
        
 
        


        
        if self.order>=0:
            return self.state[3:]
        else:
            return -self.state[3:]

    def _get_obs(self):
        return np.concatenate((self.state, self.reach_targets)) 
    
    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


