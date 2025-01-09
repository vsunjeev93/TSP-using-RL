import torch
from transformer import Transformer
from torch.utils.data import DataLoader
from critic import TransformerCritic
torch.manual_seed(2)
epoch=2
city=20
batch=512
instances=2500
params={'Nx':3, # number of MHA layers in encoder sequentially
'embed':128, #embedding dimension
'heads':8}
mse_loss=torch.nn.MSELoss()
actor=Transformer(**params,sample=True)
critic=TransformerCritic(**params,sample=False)
LR=0.0001
actor_optim=torch.optim.Adam(actor.parameters(), lr = LR)
critic_optim=torch.optim.Adam(critic.parameters(),lr=LR)
steps_per_epoch=20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for e in range(epoch):
	training_set=torch.rand((512*2500,city,2)).to(device)
	for t in range(steps_per_epoch):	
		data=DataLoader(training_set,batch_size=512,shuffle=True)
		for instance,state in enumerate(data):
			state_actor,ll=actor(state)
			distance=torch.sum(torch.linalg.vector_norm(state_actor[:,:-1,:]-state_actor[:,1:,:],dim=2),dim=1)+torch.linalg.vector_norm(state_actor[:,0,:]-state_actor[:,-1,:],dim=1)
			state_value=critic(state)
			loss=((distance.detach()-state_value.detach())*ll).mean()
			critic_loss=mse_loss(distance.detach(),state_value).mean()
			critic_optim.zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm = 1, norm_type = 2)
			critic_optim.step()
			actor_optim.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm = 1, norm_type = 2)
			actor_optim.step()
			print(distance.mean(dim=0),'critic loss',critic_loss,'actor_loss',loss,'epcoh',e,t,instance)
torch.save(actor.state_dict(), 'actor_20.pt')
	


	

