import torch
from torch import nn

class MultiHeadAttention(torch.nn.Module):
	def __init__(self,n_dim=128,n_heads=8):
		super().__init__()
		self.K=torch.nn.Linear(n_dim,n_dim)
		self.Q=torch.nn.Linear(n_dim,n_dim)
		self.V=torch.nn.Linear(n_dim,n_dim)
		self.Wo=torch.nn.Linear(n_dim,n_dim)
		self.heads=n_heads
	def forward(self,x,mask=None):
		'''x- input sequence (batch,n_city,n_embedding)'''
		keys=self.K(x) #(batch,n_city,n_embedding)
		values=self.V(x) ##(batch,n_city,n_embedding)
		queries=self.Q(x)#(batch,n_city,n_embedding)
		batch,n_city,dim=x.size()
		dim_h=int(dim/self.heads)
		# print('keys-----------',keys)
		keys_h=torch.reshape(keys,(batch,n_city,self.heads,dim_h))
		# print('keys_h--------',keys_h)
		values_h=torch.reshape(values,(batch,n_city,self.heads,dim_h)) #(batch,n_city,heads,dim_h)
		queries_h=torch.reshape(queries,(batch,n_city,self.heads,dim_h))
		# print('---------keys',keys_h)
		# print('----------queries',queries_h)
		dot=torch.matmul(queries_h.permute(0,2,1,3),keys_h.permute(0,2,3,1)) # (batch,heads,n_city,n_city)
		# print('---------dot',dot)
		dot=dot/(dim_h**(1/2))
		dot_s=torch.softmax(dot,dim=-1)
		context=torch.matmul(dot_s,values_h.permute(0,2,1,3)) #(batch,heads,n_city,dim_h)
		context=context.permute(0,2,1,3).reshape(batch,n_city,self.heads*dim_h) #(batch,n_city,dim)
		out=self.Wo(context)
		return out
class BatchNorm(torch.nn.Module):
	def __init__(self,dim=128):
		super().__init__()
		self.m=nn.BatchNorm1d(dim)
	def forward(self,x):
		x=x.permute(0,2,1)
		out=self.m(x).permute(0,2,1)
		return out
class FF(torch.nn.Module):
	def __init__(self,dim=128,hidden_dim=512):
		super().__init__()
		self.FC=nn.Sequential(nn.Linear(dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,dim))
	def forward(self,x):
		return self.FC(x)

		
if __name__=="__main__":
	n_batch=2
	n_city=5
	torch.manual_seed(2)
	x=torch.randn((n_batch,n_city,4))
	attn=MultiHeadAttention(4,2)
	out=attn(x)