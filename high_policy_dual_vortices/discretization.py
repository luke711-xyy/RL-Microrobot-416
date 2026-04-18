import numpy as np
import torch
N=3
NL=10
dense_size=int(NL*8)
sparse_size=int(NL*4)
Xf=np.zeros((sparse_size+1),dtype=np.float64)
Yf=np.zeros((sparse_size+1),dtype=np.float64)
Zf=np.ones((sparse_size+1),dtype=np.float64)*0.01

Xq=np.zeros((dense_size+1),dtype=np.float64)
Yq=np.zeros((dense_size+1),dtype=np.float64)
Zq=np.ones((dense_size+1),dtype=np.float64)*0.01

Xf_match_q=np.zeros((sparse_size+1,N),dtype=np.float64)
Yf_match_q=np.zeros((sparse_size+1,N),dtype=np.float64)
Zf_match_q=np.zeros((sparse_size+1,N),dtype=np.float64)
Delta_x=np.zeros((dense_size+1,sparse_size+1),dtype=np.float64)
Delta_y=np.zeros((dense_size+1,sparse_size+1),dtype=np.float64)
Delta_z=np.zeros((dense_size+1,sparse_size+1),dtype=np.float64)
Distance=np.zeros((dense_size+1,sparse_size+1),dtype=np.float64)
Min_Distance_Label=np.zeros((dense_size+1,sparse_size+1),dtype=np.int16)
Min_Distance_Index=np.zeros((sparse_size+1,N),dtype=np.int16)
Min_Distance_num=np.zeros((sparse_size+1),dtype=np.int16)
index_equal_index=np.arange((dense_size+1))
Xf = np.arange(0, 1.0+0.1*1.0/sparse_size, 1.0/sparse_size)
Xq = np.arange(0, 1.0+0.1/dense_size, 1.0/dense_size)
    
#print(Xf.shape,Xq.shape)    
    
Xf=Xf.reshape(1,-1)
Yf=Yf.reshape(1,-1)
Zf=Zf.reshape(1,-1)
Xq=Xq.reshape(-1,1)
Yq=Yq.reshape(-1,1)
Zq=Zq.reshape(-1,1)
Xf_all=np.squeeze(Xf.reshape(1,-1))
Yf_all=np.squeeze(Yf.reshape(1,-1))
Zf_all=np.squeeze(Zf.reshape(1,-1))
Xq_all=np.squeeze(Xq.reshape(-1,1))
Yq_all=np.squeeze(Yq.reshape(-1,1))
Zq_all=np.squeeze(Zq.reshape(-1,1))

Delta_x[:,:]=Xf[:,:]-Xq[:,:]     
Delta_y[:,:]=Yf[:,:]-Yq[:,:]
Delta_z[:,:]=Zf[:,:]-Zq[:,:]        
Distance[:,:]=Delta_x[:,:]**2+Delta_y[:,:]**2+Delta_z[:,:]**2
for k in range(Distance[:,:].shape[0]):
    Min_Distance_Label[k,np.argmin(Distance[k,:])]=1
#     Min_Distance_Index[k,np.argmin(Distance[k,:])]=k
#     Min_Distance_Label[k,np.argmin(Distance[k,:])]=1print(Min_Distance_Label[i,:,:])
print(Min_Distance_Label.shape,Xq_all.shape)
for m in range(Distance[:,:].shape[1]):
    Min_Distance_num[m]=np.sum(Min_Distance_Label[:,m])
    selected_x=Min_Distance_Label[:,m]*Xq_all[:]
    selected_y=Min_Distance_Label[:,m]*Yq_all[:]
    selected_z=Min_Distance_Label[:,m]*Zq_all[:]
    

    print((np.nonzero(Min_Distance_Label[:,m])))
    Min_Distance_Index[m,0:Min_Distance_num[m]]=index_equal_index[np.nonzero(Min_Distance_Label[:,m])]
    Xf_match_q[m,0:Min_Distance_num[m]]=selected_x[np.nonzero(Min_Distance_Label[:,m])]
    Yf_match_q[m,0:Min_Distance_num[m]]=selected_y[np.nonzero(Min_Distance_Label[:,m])]       
    Zf_match_q[m,0:Min_Distance_num[m]]=selected_z[np.nonzero(Min_Distance_Label[:,m])]
print(Min_Distance_Index)
Correponding_label=np.zeros((sparse_size+1,N),dtype=np.int16)

for m in range(Distance[:,:].shape[1]):
    selected=Min_Distance_Label[:,m]
    Correponding_label[m,0:Min_Distance_num[m]]=selected[np.nonzero(Min_Distance_Label[:,m])]
    
print(Correponding_label)
print(np.max(Min_Distance_num),np.min(Min_Distance_num))
print(Min_Distance_num)
Xf_match_q=Xf_match_q.reshape(sparse_size+1,np.max(Min_Distance_num))
Yf_match_q=Yf_match_q.reshape(sparse_size+1,np.max(Min_Distance_num))
Zf_match_q=Zf_match_q.reshape(sparse_size+1,np.max(Min_Distance_num))
Min_Distance_num=Min_Distance_num.reshape(sparse_size+1,1)

Xf_all=Xf_all.reshape(sparse_size+1,1)
Yf_all=Yf_all.reshape(sparse_size+1,1)
Zf_all=Zf_all.reshape(sparse_size+1,1)        
torch.save( torch.from_numpy(Xf_match_q), 'Xf_match_q_fila.pt')
torch.save(torch.from_numpy(Yf_match_q), 'Yf_match_q_fila.pt')
torch.save(torch.from_numpy(Zf_match_q), 'Zf_match_q_fila.pt')
torch.save(torch.from_numpy(Min_Distance_Label), 'Min_Distance_Label_Fila.pt')

torch.save(torch.from_numpy(Xf_all), 'Xf_all_fila.pt')
torch.save(torch.from_numpy(Yf_all), 'Yf_all_fila.pt')
torch.save(torch.from_numpy(Zf_all), 'Zf_all_fila.pt')
torch.save(torch.from_numpy(Min_Distance_num), 'Min_Distance_num_fila.pt')
torch.save(torch.from_numpy(Correponding_label), 'Correponding_label_fila.pt')
print(Correponding_label)
#print(Xf_match_q)
# print(Yf_all)
# print(Zf_all)



#     print(Min_Distance_num[i,:])
# print(np.max(Min_Distance_num))
# print(np.min(Min_Distance_num))    
#     for m in range(Distance[i,j,:,:].shape[1]):
#         #Min_Distance_Label[i,j,k,np.argmin(Distance[i,j,k,:])]=1
#         Min_Distance_num[i,j,m]=np.sum(Min_Distance_Label[i,j,:,m])
#         selected_x=Min_Distance_Label[i,j,:,m]*Xq_all[i,j,:]
#         selected_y=Min_Distance_Label[i,j,:,m]*Yq_all[i,j,:]
#         selected_z=Min_Distance_Label[i,j,:,m]*Zq_all[i,j,:]            
#         
#         Xf_match_q[i,j,m,0:Min_Distance_num[i,j,m]]=selected_x[np.nonzero(Min_Distance_Label[i,j,:,m])]
#         Yf_match_q[i,j,m,0:Min_Distance_num[i,j,m]]=selected_y[np.nonzero(Min_Distance_Label[i,j,:,m])]       
#         Zf_match_q[i,j,m,0:Min_Distance_num[i,j,m]]=selected_z[np.nonzero(Min_Distance_Label[i,j,:,m])]    
    
    