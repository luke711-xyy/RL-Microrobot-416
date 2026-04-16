import numpy as np
import os
from os import path
import math
import numpy.matlib
from math import sin
from math import cos
from scipy.sparse.linalg import gmres
from numpy import linalg as LA
import torch
import time


directory_path = os.getcwd()
folder_name = path.basename(directory_path)


# N=int(int(folder_name)*10)
# NL=int(folder_name)





Xf_match_q_fila=torch.load(  'Xf_match_q_fila.pt')
Yf_match_q_fila=torch.load( 'Yf_match_q_fila.pt')
Zf_match_q_fila=torch.load( 'Zf_match_q_fila.pt')
Xf_all_fila=torch.load( 'Xf_all_fila.pt')
Yf_all_fila=torch.load( 'Yf_all_fila.pt')
Zf_all_fila=torch.load( 'Zf_all_fila.pt')
Label_Matrix_fila=torch.load( 'Min_Distance_Label_Fila.pt')
Min_Distance_num_fila=torch.load( 'Min_Distance_num_fila.pt')
Min_Distance_Label_fila=torch.load( 'Correponding_label_fila.pt')  #labels of stokeslet points correpsonding to the force points in fila



device = torch.device('cpu')

NL=10
N_dense=int(NL*8)
N=int(NL*4)
torch.set_num_threads(5)

Fila_point_num=Xf_all_fila.shape[0]


S_fila_fila=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],Xf_match_q_fila.shape[1],3,3),dtype=torch.double)
B_fila_fila=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],Xf_match_q_fila.shape[1],3,3),dtype=torch.double)
S_fila_fila_sum=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],3,3),dtype=torch.double)
B_fila_fila_sum=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],3,3),dtype=torch.double)

P_fila_fila=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],Xf_match_q_fila.shape[1],3),dtype=torch.double)
P_fila_fila_sum=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],3),dtype=torch.double)







Xf_match_q_fila=Xf_match_q_fila.view(1,Xf_match_q_fila.shape[0],Xf_match_q_fila.shape[1])
Yf_match_q_fila=Yf_match_q_fila.view(1,Yf_match_q_fila.shape[0],Yf_match_q_fila.shape[1])
Zf_match_q_fila=Zf_match_q_fila.view(1,Zf_match_q_fila.shape[0],Zf_match_q_fila.shape[1])




Xf_all_fila=Xf_all_fila.view(-1,1,1)
Yf_all_fila=Yf_all_fila.view(-1,1,1)
Zf_all_fila=Zf_all_fila.view(-1,1,1)




A_fila_fila=    torch.zeros(((Fila_point_num)*3,(Fila_point_num)*3),dtype=torch.double,device=device)





PA_fila_fila=    torch.zeros(((Fila_point_num),(Fila_point_num)*3),dtype=torch.double,device=device)





delta_x_fila_fila=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],Xf_match_q_fila.shape[1]),dtype=torch.double)
delta_y_fila_fila=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],Xf_match_q_fila.shape[1]),dtype=torch.double)
delta_z_fila_fila=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],Xf_match_q_fila.shape[1]),dtype=torch.double)
delta_z_I_fila_fila=torch.zeros((Fila_point_num,Xf_match_q_fila.shape[0],Xf_match_q_fila.shape[1]),dtype=torch.double)








mu=1

def pressurelet_fila_fila(x,y,z,e):

    global P_fila_fila
    R=torch.sqrt(x**2+y**2+z**2+e**2)    
    RD=1/R
    RD5=RD**5 
    R2=2*R**2+3*e**2
    RD5R2=R2*RD5
    P_fila_fila[:,:,:,0]=x*RD5R2*Min_Distance_Label_fila
    P_fila_fila[:,:,:,2]=y*RD5R2*Min_Distance_Label_fila
    P_fila_fila[:,:,:,1]=z*RD5R2*Min_Distance_Label_fila
    
    
 



def stokeslet_fila_fila(x,y,z,e):
    global S_fila_fila

    R=torch.sqrt(x**2+y**2+z**2+e**2)
    
    RD=1/R
    RD3=RD**3    
#    X=torch.cat((x.view(1,-1),y.view(1,-1),z.view(1,-1)),dim=1)
#     print(R.shape)
#     print(Min_Distance_Label_fila.shape)
#     Min_Distance_Label_fila_expand=Min_Distance_Label_fila.clone()
#     Min_Distance_Label_fila_expand=Min_Distance_Label_fila_expand.view(1,Min_Distance_Label_fila_expand.shape[0],Min_Distance_Label_fila_expand.shape[1])\
#                                     .repeat(RD.shape[0],1,1)
    
    
    #print(torch.matmul((X).view(-1,1),(X).view(1,-1)))
    S_fila_fila[:,:,:,0,0]=(RD+e**2*RD3+x*x*RD3)*Min_Distance_Label_fila
    S_fila_fila[:,:,:,0,2]=(x*y*RD3)*Min_Distance_Label_fila
    S_fila_fila[:,:,:,0,1]=(x*z*RD3)*Min_Distance_Label_fila
    S_fila_fila[:,:,:,2,0]=S_fila_fila[:,:,:,0,2]
    S_fila_fila[:,:,:,2,2]=(RD+e**2*RD3+y*y*RD3)*Min_Distance_Label_fila
    S_fila_fila[:,:,:,2,1]=(y*z*RD3)*Min_Distance_Label_fila
    S_fila_fila[:,:,:,1,0]=S_fila_fila[:,:,:,0,1]
    S_fila_fila[:,:,:,1,2]=S_fila_fila[:,:,:,2,1]
    S_fila_fila[:,:,:,1,1]= (RD+e**2*RD3+z*z*RD3)*Min_Distance_Label_fila 
       
    




def blakelet_fila_fila(x1,x2,x3,h,e):
    global B_fila_fila    
    R=torch.sqrt(x1**2+x2**2+x3**2+e**2)
    #print(h)
    RD=1/R
    RD3=RD**3
    RD5=RD**5    
    H2=h**2
    B_fila_fila[:,:,:,0,0]=(-H2 *((6* e**2)*RD5 - 2*RD3) - e**2*RD3 - RD - (6 *(H2)* x1**2)*RD5 - x1**2*RD3 + (6 *e**2 *h *x3)*RD5\
                + 2* h *(x3*RD3 - (3* x1 *x1* x3)*RD5))*Min_Distance_Label_fila
    B_fila_fila[:,:,:,0,2]=(-((6 *H2 *x1 *x2)*RD5) - (x1 *x2)*RD3 -(6* h* x1 *x2 *x3)*RD5)*Min_Distance_Label_fila
    B_fila_fila[:,:,:,0,1]=  ((6 *H2* x1 *x3)*RD5 - (x1* x3)*RD3 - 2 *h *(x1*RD3 - (3 *x1 *x3 *x3)*RD5))*Min_Distance_Label_fila
    B_fila_fila[:,:,:,2,0]=  (-((6* H2 *x1 *x2)*RD5) - (x1* x2)*RD3 - (6 *h *x1* x2 *x3)*RD5)*Min_Distance_Label_fila
    B_fila_fila[:,:,:,2,2]=   (-H2 *((6 *e**2)*RD5 - 2*RD3) - e**2*RD3 - RD - (6 *H2 *x2**2)*RD5 - x2**2*RD3 +\
                   (6 *e**2 *h *x3)*RD5 + 2* h *(x3*RD3 - (3 *x2 *x2 *x3)*RD5))*Min_Distance_Label_fila
    B_fila_fila[:,:,:,2,1]=((6* H2 *x2 *x3)*RD5 - (x2 *x3)*RD3 - 2 *h *(x2*RD3 - (3 *x3 *x2* x3)*RD5))*Min_Distance_Label_fila
    B_fila_fila[:,:,:,1,0]=   (-((6 *e**2* h *x1)*RD5) - (6 *H2 *x1 *x3)*RD5 - (x1 *x3)*RD3 + 2 *h *(-(x1*RD3) - (3 *(e**2 *x1 + x1 *x3 *x3))*RD5))*Min_Distance_Label_fila
    B_fila_fila[:,:,:,1,2]=    (-((6 *e**2* h *x2)*RD5) - (6 *H2 *x2* x3)*RD5 - (x2 *x3)*RD3 + 2 *h *(-(x2*RD3) - (3 *(e**2 *x1 + x3 *x2 *x3))*RD5))*Min_Distance_Label_fila
    B_fila_fila[:,:,:,1,1]=   (H2 *((6 *e**2)*RD5 - 2*RD3) - e**2*RD3 - RD + (6 *H2 *x3**2)*RD5 - x3**2*RD3 - \
                   2 *h *(x3*RD3 - (3 *(e**2 *x1 + x3 *x3 *x3))*RD5))*Min_Distance_Label_fila

    


    
  


def M1M2(e):
    global S_fila_fila
  
    
    global B_fila_fila
  
    
    
    global S_fila_fila_sum

    
    global B_fila_fila_sum

    
    global P_fila_fila_sum
   
    
   
    
    
    
    global A

    global A_fila_fila

    
    global PA

    global PA_fila_fila

   
    
    
    
    global delta_x_fila_fila
    global delta_y_fila_fila
    global delta_z_fila_fila
    global delta_z_I_fila_fila


    

   
   
    global Xf_match_q_fila
    global Yf_match_q_fila    
    global Zf_match_q_fila   
    global Xf_all_fila
    global Yf_all_fila    
    global Zf_all_fila 
    

    Fila_point_num=Xf_all_fila.shape[0]

    #print(Xf_all_fila.shape,Xf_match_q_fila.shape)
    
    

    delta_x_fila_fila=Xf_all_fila-Xf_match_q_fila
    
    delta_z_fila_fila=Zf_all_fila-Zf_match_q_fila    
    delta_y_fila_fila=Yf_all_fila-Yf_match_q_fila
        
    delta_z_I_fila_fila=Zf_all_fila+Zf_match_q_fila
    
#     delta_x_fila_wall=Xf_all_fila-Xf_match_q_wall
#     delta_y_fila_wall=Yf_all_fila-Yf_match_q_wall    
#     delta_z_fila_wall=Zf_all_fila-Zf_match_q_wall    
#     delta_z_I_fila_wall=Zf_all_fila+Zf_match_q_wall
# 
#     delta_x_wall_wall=Xf_all_wall-Xf_match_q_wall
#     delta_y_wall_wall=Yf_all_wall-Yf_match_q_wall    
#     delta_z_wall_wall=Zf_all_wall-Zf_match_q_wall    
#     delta_z_I_wall_wall=Zf_all_wall+Zf_match_q_wall
#     
#     delta_x_wall_fila=Xf_all_wall-Xf_match_q_fila
#     delta_y_wall_fila=Yf_all_wall-Yf_match_q_fila    
#     delta_z_wall_fila=Zf_all_wall-Zf_match_q_fila    
#     delta_z_I_wall_fila=Zf_all_wall+Zf_match_q_fila    
    
    
   
    
    pressurelet_fila_fila(delta_x_fila_fila,delta_y_fila_fila,delta_z_fila_fila,e)    
#     pressurelet_fila_wall(delta_x_fila_wall,delta_y_fila_wall,delta_z_fila_wall,e)    


    P_fila_fila_sum=torch.sum(P_fila_fila,dim=2)    
#     P_fila_wall_sum=torch.sum(P_fila_wall,dim=2)    
    
    
    PA_fila_fila[:,0:Fila_point_num]=P_fila_fila_sum[:,:,0]
    PA_fila_fila[:,Fila_point_num:Fila_point_num*2]=P_fila_fila_sum[:,:,1]    
    PA_fila_fila[:,Fila_point_num*2:Fila_point_num*3]=P_fila_fila_sum[:,:,2]     
    
    
#     PA_fila_wall[:,0:Wall_point_num]=P_fila_wall_sum[:,:,0]
#     PA_fila_wall[:,Wall_point_num:Wall_point_num*2]=P_fila_wall_sum[:,:,1]
#     PA_fila_wall[:,Wall_point_num*2:Wall_point_num*3]=P_fila_wall_sum[:,:,2]
    

    #PA=torch.cat((PA_fila_fila,PA_fila_wall),dim=1) 
    PA=PA_fila_fila
    stokeslet_fila_fila(delta_x_fila_fila,delta_y_fila_fila,delta_z_fila_fila,e)
#     stokeslet_fila_wall(delta_x_fila_wall,delta_y_fila_wall,delta_z_fila_wall,e)    
#     stokeslet_wall_wall(delta_x_wall_wall,delta_y_wall_wall,delta_z_wall_wall,e)
#     stokeslet_wall_fila(delta_x_wall_fila,delta_y_wall_fila,delta_z_wall_fila,e)    
    
    
    
#    blakelet_fila_fila(delta_x_fila_fila,delta_y_fila_fila,delta_z_I_fila_fila,-Zf_match_q_fila,e)
#     blakelet_fila_wall(delta_x_fila_wall,delta_y_fila_wall,delta_z_I_fila_wall,-Zf_match_q_wall,e)    
#     blakelet_wall_wall(delta_x_wall_wall,delta_y_wall_wall,delta_z_I_wall_wall,-Zf_match_q_wall,e)
#     blakelet_wall_fila(delta_x_wall_fila,delta_y_wall_fila,delta_z_I_wall_fila,-Zf_match_q_fila,e)   
#     print(-Zf_match_q_fila)
    
    S_fila_fila_sum=torch.sum(S_fila_fila,dim=2)
#    B_fila_fila_sum=torch.sum(B_fila_fila,dim=2)
#    S_fila_fila_sum+= B_fila_fila_sum
    
#     S_fila_wall_sum=torch.sum(S_fila_wall,dim=2)
#     B_fila_wall_sum=torch.sum(B_fila_wall,dim=2)
#     S_fila_wall_sum+= B_fila_wall_sum      
#     
#     S_wall_wall_sum=torch.sum(S_wall_wall,dim=2)
#     B_wall_wall_sum=torch.sum(B_wall_wall,dim=2)
#     S_wall_wall_sum+= B_wall_wall_sum  
#     
#     S_wall_fila_sum=torch.sum(S_wall_fila,dim=2)
#     B_wall_fila_sum=torch.sum(B_wall_fila,dim=2)
#     S_wall_fila_sum+= B_wall_fila_sum
    


    
    A_fila_fila[0:Fila_point_num,0:Fila_point_num]=S_fila_fila_sum[:,:,0,0]
    A_fila_fila[0:Fila_point_num,Fila_point_num:Fila_point_num*2]=S_fila_fila_sum[:,:,0,1]                    
    A_fila_fila[0:Fila_point_num,Fila_point_num*2:Fila_point_num*3]=S_fila_fila_sum[:,:,0,2]    
    A_fila_fila[Fila_point_num:Fila_point_num*2,0:Fila_point_num]=S_fila_fila_sum[:,:,1,0]
    A_fila_fila[Fila_point_num:Fila_point_num*2,Fila_point_num:Fila_point_num*2]=S_fila_fila_sum[:,:,1,1]                    
    A_fila_fila[Fila_point_num:Fila_point_num*2,Fila_point_num*2:Fila_point_num*3]=S_fila_fila_sum[:,:,1,2] 
    A_fila_fila[Fila_point_num*2:Fila_point_num*3,0:Fila_point_num]=S_fila_fila_sum[:,:,2,0]
    A_fila_fila[Fila_point_num*2:Fila_point_num*3,Fila_point_num:Fila_point_num*2]=S_fila_fila_sum[:,:,2,1]                    
    A_fila_fila[Fila_point_num*2:Fila_point_num*3,Fila_point_num*2:Fila_point_num*3]=S_fila_fila_sum[:,:,2,2]
    
    
#     A_fila_wall[0:Fila_point_num,0:Wall_point_num]=S_fila_wall_sum[:,:,0,0]
#     A_fila_wall[0:Fila_point_num,Wall_point_num:Wall_point_num*2]=S_fila_wall_sum[:,:,0,1]                    
#     A_fila_wall[0:Fila_point_num,Wall_point_num*2:Wall_point_num*3]=S_fila_wall_sum[:,:,0,2]    
#     A_fila_wall[Fila_point_num:Fila_point_num*2,0:Wall_point_num]=S_fila_wall_sum[:,:,1,0]
#     A_fila_wall[Fila_point_num:Fila_point_num*2,Wall_point_num:Wall_point_num*2]=S_fila_wall_sum[:,:,1,1]                    
#     A_fila_wall[Fila_point_num:Fila_point_num*2,Wall_point_num*2:Wall_point_num*3]=S_fila_wall_sum[:,:,1,2] 
#     A_fila_wall[Fila_point_num*2:Fila_point_num*3,0:Wall_point_num]=S_fila_wall_sum[:,:,2,0]
#     A_fila_wall[Fila_point_num*2:Fila_point_num*3,Wall_point_num:Wall_point_num*2]=S_fila_wall_sum[:,:,2,1]                    
#     A_fila_wall[Fila_point_num*2:Fila_point_num*3,Wall_point_num*2:Wall_point_num*3]=S_fila_wall_sum[:,:,2,2]    
#     
#     
#     A_wall_wall[0:Wall_point_num,0:Wall_point_num]=S_wall_wall_sum[:,:,0,0]
#     A_wall_wall[0:Wall_point_num,Wall_point_num:Wall_point_num*2]=S_wall_wall_sum[:,:,0,1]                    
#     A_wall_wall[0:Wall_point_num,Wall_point_num*2:Wall_point_num*3]=S_wall_wall_sum[:,:,0,2]    
#     A_wall_wall[Wall_point_num:Wall_point_num*2,0:Wall_point_num]=S_wall_wall_sum[:,:,1,0]
#     A_wall_wall[Wall_point_num:Wall_point_num*2,Wall_point_num:Wall_point_num*2]=S_wall_wall_sum[:,:,1,1]                    
#     A_wall_wall[Wall_point_num:Wall_point_num*2,Wall_point_num*2:Wall_point_num*3]=S_wall_wall_sum[:,:,1,2] 
#     A_wall_wall[Wall_point_num*2:Wall_point_num*3,0:Wall_point_num]=S_wall_wall_sum[:,:,2,0]
#     A_wall_wall[Wall_point_num*2:Wall_point_num*3,Wall_point_num:Wall_point_num*2]=S_wall_wall_sum[:,:,2,1]                    
#     A_wall_wall[Wall_point_num*2:Wall_point_num*3,Wall_point_num*2:Wall_point_num*3]=S_wall_wall_sum[:,:,2,2]        
    
    
#     A_wall_fila[0:Wall_point_num,0:Fila_point_num]=S_wall_fila_sum[:,:,0,0]
#     A_wall_fila[0:Wall_point_num,Fila_point_num:Fila_point_num*2]=S_wall_fila_sum[:,:,0,1]                    
#     A_wall_fila[0:Wall_point_num,Fila_point_num*2:Fila_point_num*3]=S_wall_fila_sum[:,:,0,2]    
#     A_wall_fila[Wall_point_num:Wall_point_num*2,0:Fila_point_num]=S_wall_fila_sum[:,:,1,0]
#     A_wall_fila[Wall_point_num:Wall_point_num*2,Fila_point_num:Fila_point_num*2]=S_wall_fila_sum[:,:,1,1]                    
#     A_wall_fila[Wall_point_num:Wall_point_num*2,Fila_point_num*2:Fila_point_num*3]=S_wall_fila_sum[:,:,1,2] 
#     A_wall_fila[Wall_point_num*2:Wall_point_num*3,0:Fila_point_num]=S_wall_fila_sum[:,:,2,0]
#     A_wall_fila[Wall_point_num*2:Wall_point_num*3,Fila_point_num:Fila_point_num*2]=S_wall_fila_sum[:,:,2,1]                    
#     A_wall_fila[Wall_point_num*2:Wall_point_num*3,Fila_point_num*2:Fila_point_num*3]=S_wall_fila_sum[:,:,2,2]     
#     
#     
#     

    
    A=A_fila_fila
    #print(A_fila_fila)
    return A/(8*math.pi*mu),PA/(8*math.pi*mu)




def MatrixQ(L,theta,Qu,Q1,Ql,Q2):

    
#     Q=torch.cat((Qu,-Q2,Ql,Q1),dim=1)    
#     Q=Q.reshape(2*(N+1),-1)
    Q_up=torch.cat((Qu,-Q2),dim=1)
    Q_down=torch.cat((Ql,Q1),dim=1)
    Q=torch.cat((Q_up,Q_down),dim=0)
    return Q


def MatrixQp(L,theta):
    
    Qu=torch.cat((torch.ones((N+1),dtype=torch.double,device=device).reshape(-1,1),torch.zeros((N+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    Ql=torch.cat((torch.zeros((N+1),dtype=torch.double,device=device).reshape(-1,1),torch.ones((N+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    q1=L*torch.cos(theta[2:])
    q2=L*torch.sin(theta[2:])
   
    

    Q1=q1.reshape(1,-1).repeat(N+1,1)
    
    Q1=torch.tril(Q1,-1)
    
    Q2=q2.reshape(1,-1).repeat(N+1,1)
    
    Q2=torch.tril(Q2,-1)    
    

    
    Q=torch.cat((Qu,Q1,Ql,Q2),dim=1)
#     Q_up=torch.cat((Qu,Q1),dim=1)
#     Q_down=torch.cat((Ql,Q2),dim=1)
#     Q=torch.cat((Q_up,Q_down),dim=0)
    Q=Q.reshape(2*(N+1),-1)
    
    return Q,Qu,Q1,Ql,Q2

def MatrixQp_dense(L,theta):
    
    Qu=torch.cat((torch.ones((N_dense+1),dtype=torch.double,device=device).reshape(-1,1),torch.zeros((N_dense+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    Ql=torch.cat((torch.zeros((N_dense+1),dtype=torch.double,device=device).reshape(-1,1),torch.ones((N_dense+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    q1=L*torch.cos(theta[2:])
    q2=L*torch.sin(theta[2:])
   
    

    Q1=q1.reshape(1,-1).repeat(N_dense+1,1)
    
    Q1=torch.tril(Q1,-1)
    
    Q2=q2.reshape(1,-1).repeat(N_dense+1,1)
    
    Q2=torch.tril(Q2,-1)    
    

    
    Q=torch.cat((Qu,Q1,Ql,Q2),dim=1) 
    Q=Q.reshape(2*(N_dense+1),-1)

    return Q,Qu,Q1,Ql,Q2


def MatrixB(L,theta,Y):
    
    B1=0.5*torch.cat((2*torch.ones((N+1),dtype=torch.double,device=device).reshape(-1,1),torch.zeros((N+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    B1[0,0]=0.5
    B1[-1,0]=0.5
    B1=B1.reshape(1,-1)
#     
    B2=0.5*torch.cat((torch.zeros((N+1),dtype=torch.double,device=device).reshape(-1,1),2*torch.ones((N+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    B2[0,1]=0.5
    B2[-1,1]=0.5
    B2=B2.reshape(1,-1)
    
#     Y1=torch.cat((Y[:-1,:],torch.zeros((2),dtype=torch.double,device=device).reshape(1,-1)),dim=0)
#     
#     Y2=torch.cat((torch.zeros((2),dtype=torch.double,device=device).reshape(1,-1),Y[:-1,:]),dim=0)
# 
#     
# #     Y01=torch.repeat(Y[0,:],N+1)
# #     Y01=Y01.reshape(-1,N+1).T
#     #print(Y.shape)
#     Y01=Y[0,:].reshape(1,-1).repeat(N+1,1)
#     #Y02=Y01.detach()    
#     Y01[-1,0]=0
#     Y01[-1,1]=0
# # torch.repeat(Y[:,0],N+1) XX.reshape(-1,N+1)
# 
# #     Y02=torch.repeat(Y[0,:],N+1)
# #     Y02=Y02.reshape(-1,N+1).T
#     Y02=Y[0,:].reshape(1,-1).repeat(N+1,1)
#     Y02[0,0]=0
#     Y02[0,1]=0
    #np.savetxt('Y02.out', Y02.numpy(), delimiter=',')
#     t=torch.cat((torch.cos(theta[2:]).reshape(-1,1),torch.sin(theta[2:]).reshape(-1,1)),dim=1)
#      
#     t1= torch.cat((t,torch.zeros((2),dtype=torch.double,device=device).reshape(1,-1)),dim=0)
#    
#     t2= torch.cat((torch.zeros((2),dtype=torch.double,device=device).reshape(1,-1),t),dim=0)
#     B3=0.5*L*(Y1-Y01)+(L**2)/6.0*t1+0.5*L*(Y2-Y02)+(L**2)/3.0*t2
#     #np.savetxt('B3.out', B3.numpy(), delimiter=',')  
#     B3=torch.cat((-B3[:,1].reshape(-1,1),B3[:,0].reshape(-1,1)),dim=1)
    
    B3=torch.cat((-(Y[:,1]-Y[0,1]).view(1,-1),(Y[:,0]-Y[0,0]).view(1,-1)),dim=1)
    #B3=B3.reshape(2,-1).T
    B3=B3.reshape(1,-1)
#     print(B3)
    #np.savetxt('B3.out', B3.numpy(), delimiter=',')     
    B=torch.cat((B1,B2),dim=0)
    
    Bx=B[:,0::2]
    Bx=torch.cat((Bx,B3[:,:Bx.shape[1]]),dim=0)
    By=B[:,1::2]
    By=torch.cat((By,B3[:,Bx.shape[1]:]),dim=0)    
    Min_Distance_num_fila_copy=Min_Distance_num_fila.clone()
    Min_Distance_num_fila_copy=Min_Distance_num_fila_copy.reshape(1,-1).repeat(3,1)
    Bx=Bx*Min_Distance_num_fila_copy
    By=By*Min_Distance_num_fila_copy

    B=torch.cat((Bx,By),dim=1)

    #print(torch.mean(B1),torch.mean(B2),torch.mean(B3),'B')  
    return B

def MatrixC(action_absolute):
    C1=torch.zeros((N+2,3),dtype=torch.double,device=device)
    C1[0,0]=1
    C1[1,1]=1
    C1[2:,2]=1
    C2=torch.zeros((N+2,1),dtype=torch.double,device=device)
    C2[3:,:]=action_absolute.view(-1,1) #N-1,1, start's rotation velocity removed
    #print(C1,C2)
    return C1, C2




def MatrixD_sum(beta_ini,absU):
    D1=torch.zeros((NL*2+2,3),dtype=torch.double,device=device)
    D2=torch.zeros((NL*2+2,1),dtype=torch.double,device=device)
    for i in range(NL+1):
        if i==0:
            D1[i*2,0]=1
            D1[i*2+1,1]=1            
           
            D2[i*2,:]=0
            D2[i*2+1,:]=0            
                       
        elif i==1:
            D1[i*2,0]=1
            D1[i*2,2]=  D1[(i-1)*2,2] -sin(beta_ini[i-1])*1.0/NL
            D1[i*2+1,1]=1             
            D1[i*2+1,2]= D1[(i-1)*2+1,2] +cos(beta_ini[i-1])*1.0/NL           
           
            D2[i*2,:]=D2[(i-1)*2,:]
            D2[i*2+1,:]=D2[(i-1)*2+1,:]          
        else:
            D1[i*2,0]=1
            D1[i*2,2]=  D1[(i-1)*2,2] -sin(beta_ini[i-1])*1.0/NL
            D1[i*2+1,1]=1             
            D1[i*2+1,2]= D1[(i-1)*2+1,2] +cos(beta_ini[i-1])*1.0/NL        
           
            D2[i*2,:]=D2[(i-1)*2,:]-absU[i-2]*sin(beta_ini[i-1])*1.0/NL
            D2[i*2+1,:]=D2[(i-1)*2+1,:]+absU[i-2]*cos(beta_ini[i-1])*1.0/NL
   
    return D1, D2


def MatrixD_position(beta_ini,Xini,Yini,L):
    D1=torch.zeros((NL+1,NL),dtype=torch.double,device=device)
    D2=torch.ones((NL+1,1),dtype=torch.double,device=device)*Yini
    D1x=torch.zeros((NL+1,NL),dtype=torch.double,device=device)
    D2x=torch.ones((NL+1,1),dtype=torch.double,device=device)*Xini    
    for i in range(NL+1):
        if i==0:
            D1[i,:]=0
            D1x[i,:]=0
        else:
            D1[i,:i]=torch.sin(beta_ini[:i])*L
            D1x[i,:i]=torch.cos(beta_ini[:i])*L   
    return D1, D2,D1x,D2x



def Calculate_velocity(x,w,x_first):
    global Xf_match_q_fila
    global Yf_match_q_fila    
    global Zf_match_q_fila   
    global Xf_all_fila
    global Yf_all_fila    
    global Zf_all_fila    
    
    L,e,Y,theta,action_absolute,Qu,Q1,Ql,Q2,action,beta_ini,absU,Xini,Yini=initial(x,w,x_first)
    Y_dense=initial_dense(x,w,x_first)
    
    Xf_all_fila=Y[:,0].clone()
    
#     Yf_fila=torch.zeros_like(Xf_fila)    
    Zf_all_fila=Y[:,1].clone()
    Xf_all_fila=Xf_all_fila.view(-1,1,1)
#     Yf_all_fila=Yf_all_fila.view(-1,1,1)    
    Zf_all_fila=Zf_all_fila.view(-1,1,1)
#     Xf_q_fila=torch.zeros_like(Xf_match_q_fila)
#     Yf_q_fila=torch.zeros_like(Xf_match_q_fila)    
#     Zf_q_fila=torch.zeros_like(Xf_match_q_fila)    
    
    
    #print(Min_Distance_Label_fila.shape,Y_dense.shape)
    for m in range(Xf_match_q_fila.shape[1]):
        
        selected_x=Label_Matrix_fila[:,m]*Y_dense[:,0]
        
        selected_z=Label_Matrix_fila[:,m]*Y_dense[:,1]        
        Xf_match_q_fila[:,m,0:Min_Distance_num_fila[m]]=selected_x[np.nonzero(Label_Matrix_fila[:,m])].view(1,-1)
        Zf_match_q_fila[:,m,0:Min_Distance_num_fila[m]]=selected_z[np.nonzero(Label_Matrix_fila[:,m])].view(1,-1) 


    
    B=MatrixB(L,theta,Y)
    
#     print(B)
#     
    
    
    
    A,Ap=M1M2(e)
    
    B_supply=torch.zeros((3,A.shape[0]-B.shape[1]))
    B_all=torch.cat((B,B_supply),dim=1)
    
    
  
    Q=MatrixQ(L,theta,Qu,Q1,Ql,Q2)

    C1,C2=MatrixC(action_absolute)
 
    AB=torch.zeros((3,A.shape[0]),dtype=torch.double,device=device)
    
    AB = torch.linalg.solve(A.T, B_all.T)
    AB=(AB.T).double()
#     Ainv=torch.linalg.inv(A)
    #print(Ainv[:B.shape[1],:B.shape[1]])
#     AB=torch.matmul(B,Ainv[:B.shape[1],:B.shape[1]])    
    AB=AB[:,:B.shape[1]]    
        
    MT=torch.matmul(AB,Q)     


       
    M=torch.matmul(MT,C1)

    R=-torch.matmul(MT,C2)
    D1,D2=MatrixD_sum(beta_ini,absU)     
    velo=torch.matmul(torch.linalg.inv(M),R)
    
    velo_points=torch.matmul(C1,(velo))+C2    
#     print(velo)
    veloall=torch.matmul(D1,velo)+D2    
    velo=velo.numpy()
    velo=np.squeeze(velo)
    omega=velo[2]    


#     print(C1.shape,C2.shape)
    velo_points_all=torch.matmul(Q,velo_points)    
#    veloall=torch.matmul(D1,velo)+D2
#     velo_points_filawall=torch.zeros(((Wall_point_num+Fila_point_num)*3,1),dtype=torch.double,device=device)   
#     velo_points_filawall[:Fila_point_num*2,:]=velo_points_all
    velo_points_fila=torch.zeros(((Fila_point_num)*3,1),dtype=torch.double,device=device)
    velo_points_fila[:Fila_point_num*2,:]=velo_points_all
    force_points_fila= torch.linalg.solve(A, velo_points_fila)
    
        
    pressure_all=torch.matmul(Ap,force_points_fila.reshape(-1,1))
    pressure_end=(pressure_all.view(1,-1))[0,-1]
    pressure_start=(pressure_all.view(1,-1))[0,0]
    
    veloall[0]=0.5*(veloall[0]+veloall[-2])
    veloall[1]=0.5*(veloall[1]+veloall[-1])
    veloall=veloall[:-2]
    velon=velo.copy()
    
    velon[0]=torch.mean(veloall[::2])*1.1
    velon[1]=torch.mean(veloall[1::2])*1.1
    
    output=np.concatenate((velon,action))
    
    
    D1y,D2y,D1x,D2x=MatrixD_position(beta_ini,Xini,Yini,1.0/NL)
    #print(D1y)
    Yp=torch.matmul(D1y,torch.ones((NL,1),dtype=torch.double,device=device))+D2y
    Xp=torch.matmul(D1x,torch.ones((NL,1),dtype=torch.double,device=device))+D2x   
    #print(Yp)
    return output,velo,np.squeeze(Xp.numpy()),np.squeeze(Yp.numpy()),pressure_end-pressure_start,pressure_end,pressure_all




def initial(x,w,x_first):




#     N1=int(N/4)-1
#     N2=int(N/4)*2-1
#     N3=int(N/4)*3-1
#     N4=int(N/4)*4-1 
# 
#     #L=2.0/N
#     L=4.0/N
    L=1.0/N   
    e=L*0.1
    
    Xini=x_first[0]
    Yini=x_first[1]
    #Yini=9*10000
    #print(Xini**2+Yini**2)
    # beta1_ini=math.pi*0.0
    # beta2_ini=math.pi/2
    # beta3_ini=math.pi
    # beta4_ini=math.pi*1.5


     
#     beta_ini=torch.tensor(x[1:].copy(),dtype=torch.double,device=device)
#     beta_ini[1]+=beta_ini[0]
#     beta_ini[2]+=beta_ini[1]
#     beta_ini[3]+=beta_ini[2]   
    beta_ini=torch.tensor(x[2:].copy(),dtype=torch.double,device=device)
    NI=np.zeros(NL,dtype=np.double)
    for i in range(NL):
        NI[i]=int(int(N/NL)*(i+1)-1)
        if i>0:
            beta_ini[i]+=beta_ini[i-1]    
   
    # beta1_ini=0
    # beta2_ini=0
    # beta3_ini=0
    # beta4_ini=0
    theta=torch.zeros((N+2),dtype=torch.double,device=device)
    forQp=torch.ones((N+2),dtype=torch.double,device=device)
    forQp[0]=Xini
    forQp[1]=Yini

    theta[0]=Xini
    theta[1]=Yini

    for i in range(N):
        theta[i+2]=beta_ini[int((i)/(N/NL))]    


   
    Q,Qu,Q1,Ql,Q2=MatrixQp(L,theta)
    Yposition=torch.matmul(Q,forQp)            

    
    
    Yposition=Yposition.reshape(-1,2)
    
    #print(Yposition,theta)

    absU=w.copy()
    action=absU.copy() 
#     absU[1]=absU[1]+absU[0]
#     absU[2]=absU[2]+absU[1]
    #absU[8]=absU[8]+absU[7]    
    
    for i in range(NL):
     
        if i>0 and i<NL-1:
            absU[i]+=absU[i-1]
    


    
    action_absolute=torch.zeros((N-1),dtype=torch.double,device=device)
#     action_absolute[N1:N2]=absU[0]
#     action_absolute[N2:N3]=absU[1]
#     action_absolute[N3:]=absU[2]
#    


    for i in range(NL-1):
        if i==NL-2:
            a=int(NI[i])            
            action_absolute[a:]=absU[i]
        else:
            a=int(NI[i])
            b=int(NI[i+1])
            
            action_absolute[a:b]=absU[i]
    action_absolute.view(-1,1)

    return L,e,Yposition,theta,action_absolute,Qu,Q1,Ql,Q2,action,beta_ini,absU,Xini,Yini




def initial_dense(x,w,x_first):
    
    L=1.0/N_dense  
    e=L*0.1
    
    Xini=x_first[0]
    Yini=x_first[1]
  
    beta_ini=torch.tensor(x[2:].copy(),dtype=torch.double,device=device)
    NI=np.zeros(NL,dtype=np.double)
    for i in range(NL):
        NI[i]=int(int(N_dense/NL)*(i+1)-1)
        if i>0:
            beta_ini[i]+=beta_ini[i-1]    
   

    theta=torch.zeros((N_dense+2),dtype=torch.double,device=device)
    forQp=torch.ones((N_dense+2),dtype=torch.double,device=device)
    forQp[0]=Xini
    forQp[1]=Yini

    theta[0]=Xini
    theta[1]=Yini

    for i in range(N_dense):
        theta[i+2]=beta_ini[int((i)/(N_dense/NL))]    


   
    Q,Qu,Q1,Ql,Q2=MatrixQp_dense(L,theta)
    Yposition=torch.matmul(Q,forQp)            

    
    
    Yposition=Yposition.reshape(-1,2)
    
    #print(Yposition)
  
    return Yposition





def RK(x,w,x_first):
    Xn=0.0
    Yn=0.0
    r=0.0
    xc=x+1.0
    xc=xc-1.0
    x_first_delta=np.zeros((2))
    x_fc=x_first.copy()
    Ypositions=np.zeros((NL+1))
    Ntime=10
    whole_time=0.2
    part_time=whole_time/Ntime
#     print(x)
    for i in range(Ntime):
        #print(xc.shape,w.shape)
        V,Vo,Xp,Yp,pressure_diff,pressure_end,pressure_all=Calculate_velocity(xc, w,x_fc)
        k1=part_time*V
        V,Vo,Xp,Yp,pressure_diff,pressure_end,pressure_all=Calculate_velocity(xc+0.5*k1, w,x_fc+0.5*part_time*Vo[:2])        
        
        #print(x_fc[0]**2+x_fc[1]**2)
        k2=part_time*V
        
        #k2=0.01*Calculate_velocity(xc+0.5*k1[2:], w)
        xc+=k2
        xc[2]=(xc[2]+math.pi)%(2*math.pi)-math.pi
          
        Xn+=k2[0]
        Yn+=k2[1]
        x_first_delta+=part_time*Vo[:2]
        x_fc+=part_time*Vo[:2]

        r+=(k2[0])/(whole_time)
          
#     Ypositions[0]=xc[0]-sin(xc[1])-sin(xc[1]+xc[2])
#     Ypositions[1]=Ypositions[0]+sin(xc[1])
#     Ypositions[2]= Ypositions[1] + sin(xc[1]+xc[2]) 
#     Ypositions[3]= Ypositions[2] + sin(xc[1]+xc[2]+xc[3])
#     Ypositions[4]= Ypositions[3] + sin(xc[1]+xc[2]+xc[3]+xc[4])
    #print(Yp.shape)
    #print(xc[0],xc[1])
    #print(pressure_diff)
    pressure_all=np.array(pressure_all)
    pressure_all=np.squeeze(pressure_all)    
    return xc , Xn ,r  ,x_first_delta,Xp,Yp,pressure_diff,pressure_end,pressure_all
    
    
    
    



    
    
    
    
    
