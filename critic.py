import torch
from torch import nn
from encoder import EncoderBlock
from decoder_MHA import DecoderMHA

class TransformerCritic(nn.Module):
	def __init__(self,Nx,embed,heads,sample=True):
		super().__init__()
		self.Nx=Nx
		encoder_blocks=[]
		self.embedding=nn.Linear(2,embed)
		for i in range(self.Nx):
			encoder_blocks.append(EncoderBlock(embed,heads))
		self.encoder=nn.Sequential(*encoder_blocks)
		self.FF=nn.Sequential(nn.Linear(embed,128),nn.ReLU(),nn.Linear(128,1))
		self.heads=heads
		self.embed=embed
		self.sample=sample
	def forward(self,x):
		batch,city,_=x.size()
		x_embed=self.embedding(x)
		enc_out=self.encoder(x_embed)#(batch,city,embed)
		enc_out_mean=torch.mean(x_embed,dim=1,keepdim=True).squeeze() #(batch,embed)
		state_value=self.FF(enc_out_mean).squeeze()#(batch,1)
		return state_value

if __name__=="__main__":
	n_batch=2
	n_city=5
	torch.manual_seed(2)
	x=torch.randn((n_batch,n_city,2))
	TB=TransformerCritic(4,4,4,False)
	out=TB(x)
	print(out)
	# print(out.size(),out_mean.size())