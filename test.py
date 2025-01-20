import torch
from transformer import Transformer
params={'Nx':3, # number of MHA layers in encoder sequentially
'embed':128, #embedding dimension
'heads':8}
city=20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Transformer(**params)
model.load_state_dict(torch.load('actor_20.pt'))
model.eval()
test_input=torch.rand((1,city,2)).to(device).repeat(512,1,1)
with torch.no_grad():
    out_tour,_=model(test_input)
distance=torch.sum(torch.linalg.vector_norm(out_tour[:,:-1,:]-out_tour[:,1:,:],dim=2),dim=1)+torch.linalg.vector_norm(out_tour[:,0,:]-out_tour[:,-1,:],dim=1)
print(distance)