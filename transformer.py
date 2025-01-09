import torch
from torch import nn
from encoder import EncoderBlock
from decoder_MHA import DecoderMHA

class Transformer(nn.Module):
	def __init__(self,Nx,embed,heads,sample=True):
		super().__init__()
		self.Nx=Nx
		encoder_blocks=[]
		self.embedding=nn.Linear(2,embed)
		for i in range(self.Nx):
			encoder_blocks.append(EncoderBlock(embed,heads))
		self.encoder=nn.Sequential(*encoder_blocks)
		self.v1=nn.Parameter(nn.Parameter(torch.FloatTensor(embed)))
		self.vf=nn.Parameter(nn.Parameter(torch.FloatTensor(embed)))
		self.decoder=DecoderMHA(embed,heads)
		self.heads=heads
		self.embed=embed
		self.sample=sample
	def forward(self,x):
		batch,city,_=x.size()
		x_embed=self.embedding(x)
		enc_out=self.encoder(x_embed)#(batch,city,embed)
		enc_out_mean=torch.mean(x_embed,dim=1,keepdim=True) #(batch,1,embed)
		h0=self.v1.unsqueeze(0).repeat(batch,1).unsqueeze(1)
		hprev=self.vf.unsqueeze(0).repeat(batch,1).unsqueeze(1)
		mask=torch.zeros(batch,city)#(batch,city)
		tour=[]
		log_l=torch.zeros(batch,city)
		# print(enc_out,enc_out_mean)
		for i in range(city):
			# print(enc_out_mean.size())
			logits=self.decoder(enc_out,enc_out_mean,hprev,h0,mask) #(batch,city)
			log_p=torch.log_softmax(logits,dim=1)#(n_batch,city)
			# print('log_p',log_p.exp())
			# next_node=torch.argmax(log_p,dim=1)#(n_batch)
			if self.sample:
				next_city=torch.multinomial(log_p.exp(), 1).long() #(n_batch,1)
				# print(next_city.size(),'here')
			else:
				next_city=torch.argmax(log_p.exp(), dim=1,keepdim=True)
				# print(next_city.size(),'here')
			if i==0:
				h0=torch.gather(enc_out,index=next_city.unsqueeze(1).repeat(1,1,self.embed),dim=1)#(batch,1,embed)
			mask=mask+torch.zeros(batch,city).scatter_(dim=-1,index=next_city,value=1)
			hprev=torch.gather(enc_out,index=next_city.unsqueeze(1).repeat(1,1,self.embed),dim=1)#(batch,1,embed)
			log_l=log_l+torch.gather(log_p,index=next_city,dim=1)
			tour.append(next_city)
		log_l_sum=log_l.sum(dim=1)
		tour=torch.stack(tour,dim=1)#(batch,city,1)
		# print(tour)
		out_tour=torch.gather(x,index=tour.repeat(1,1,2),dim=1)
		# print(out_tour,x)
		# print(log_l_sum,out_tour)
		return out_tour,log_l_sum

if __name__=="__main__":
	n_batch=2
	n_city=5
	torch.manual_seed(2)
	x=torch.randn((n_batch,n_city,2))
	TB=Transformer(4,4,4,False)
	out,ll=TB(x)
	# print(out.size(),out_mean.size())