import torch
from torch import nn
from attention import MultiHeadAttention, BatchNorm, FF

class DecoderMHA(nn.Module):
	def __init__(self,n_dim=128,n_heads=8):
		super().__init__()
		self.K=torch.nn.Linear(n_dim,n_dim)
		self.Q=torch.nn.Linear(3*n_dim,n_dim)
		self.V=torch.nn.Linear(n_dim,n_dim)
		self.Wo=torch.nn.Linear(n_dim,n_dim)
		self.Qf=torch.nn.Linear(n_dim,n_dim)
		self.Kf=torch.nn.Linear(n_dim,n_dim)
		self.heads=n_heads
		self.inf=10**8
	def forward(self,h,h_N,h_prev,h_0,mask=None):
		'''h: (batch,city, embed) - final encoder hidden state
		   h_N: (batch,1,embed)- average final enc hidden state
		   h_prev,h_0: (batch,1,embed) previous city's enc state and first city's enc state
		   mask: (batch,city) or None
		'''
		batch,city,embed=h.size()
		h_c=torch.concat((h_N,h_prev,h_0),2)
		q=self.Q(h_c) #(batch,embed)
		v=self.V(h) #(batch,city,embed)
		k=self.K(h) #(batch,city,embed)
		embed_h=int(embed/self.heads)
		q_h=torch.reshape(q,(batch,1,self.heads,embed_h)) #(batch,1,head,embed_h)
		v_h=torch.reshape(v,(batch,city,self.heads,embed_h))#(batch,city,head,embed_h)
		k_h=torch.reshape(v,(batch,city,self.heads,embed_h))
		# print(q_h.permute(0,2,1,3).size(),k_h.permute(0,2,3,1).size())
		dot=torch.matmul(q_h.permute(0,2,1,3),k_h.permute(0,2,3,1)) #(batch,head,1,embed_h)x(batch,head,embed_h,city)->(batch,head,1,city)
		dot=dot/(embed_h**(1/2))
		mask_h=mask.unsqueeze(1).unsqueeze(1).repeat(1,self.heads,1,1)
		dot=dot-mask_h*self.inf
		#softmax
		dot_s=torch.softmax(dot,dim=-1)# (batch,head,1,city)
		context=torch.matmul(dot_s,v_h.permute(0,2,1,3))#(batch,head,1,city)x(batch,head,city,embed_h)-> (batch,head,1,embed_h)
		# print(context.size())
		context=context.permute(0,2,1,3).reshape(batch,1,self.heads*embed_h) #(batch,1,embed)
		dec_out=self.Wo(context)
		qf=self.Qf(dec_out)#(batch,1,embed)
		kf=self.Kf(h)#(batch,city,embed)
		dot=torch.matmul(qf,kf.permute(0,2,1)).squeeze() #(batch,1,embed)x(batch,embed,city)->(batch,city)
		dot=dot/(embed**(1/2))
		dot=10*torch.tanh(dot)
		dot=dot-mask*self.inf
		# print(dot.size())
		return dot

if __name__=="__main__":
	n_batch=2
	n_city=5
	torch.manual_seed(2)
	x=torch.randn((n_batch,n_city,4))
	h_N=x.mean(dim=1,keepdim=True)
	h_0=x[:,2:3,:]

	attn=DecoderMHA(4,2)
	out=attn(x,h_N,h_0,h_0)