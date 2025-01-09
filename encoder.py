import torch
from torch import nn
from attention import MultiHeadAttention, BatchNorm, FF

class EncoderBlock(nn.Module):
	def __init__(self,embed=128,heads=8):
		super().__init__()
		self.MHA=MultiHeadAttention(embed,heads)
		self.BN1=BatchNorm(embed)
		self.BN2=BatchNorm(embed)
		self.FF=FF(embed,512)
	def forward(self,x):
		context=self.MHA(x)
		context=self.BN1(x+context)
		FF_out=self.FF(context)
		FF_out=self.BN2(FF_out+context)
		return FF_out


if __name__=="__main__":
	n_batch=2
	n_city=5
	torch.manual_seed(2)
	x=torch.randn((n_batch,n_city,4))
	EB=EncoderBlock(4,2)
	out=EB(x)
	print(out.size())