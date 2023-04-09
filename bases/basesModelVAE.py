import torch
from torch import nn
from mlagents.trainers.torch_modules.encoders import SimpleVariationnalInputBase
from mlagents.trainers.torch_modules.decoders import SimpleVariationnalOutputBase

class BasesVAE(nn.Module):
	def __init__(self,state_dim,latent_dim,hidden_dim):
		super(BasesVAE, self).__init__()
		self.state_dim  = state_dim
		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim
		self.in_ = SimpleVariationnalInputBase(state_dim,latent_dim,hidden_dim)
		self.out = SimpleVariationnalOutputBase(state_dim,latent_dim,hidden_dim)

	def forward(self, x):
		pass
