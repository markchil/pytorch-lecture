import torch
from module import ScaledLinear

layer = ScaledLinear(16, 4, 2.0)
state_dict = layer.state_dict()
torch.save(state_dict, 'model.pt')

state_dict = torch.load('model.pt')
layer.load_state_dict(state_dict)
# <All keys matched successfully>
