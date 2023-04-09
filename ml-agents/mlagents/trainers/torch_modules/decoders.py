from typing import List, Dict
import numpy as np
from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_modules.layers import linear_layer,Initialization


class ValueHeads(nn.Module):
    def __init__(self, stream_names: List[str], input_size: int, output_size: int = 1):
        super().__init__()
        self.stream_names = stream_names
        _value_heads = {}

        for name in stream_names:
            value = linear_layer(input_size, output_size)
            _value_heads[name] = value
        self.value_heads = nn.ModuleDict(_value_heads)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        value_outputs = {}
        for stream_name, head in self.value_heads.items():
            value_outputs[stream_name] = head(hidden).squeeze(-1)
        return value_outputs


class SimpleVariationnalOutputBase(nn.Module):
    def __init__(self,output_size : int, latent_dim : int ,hidden_dim : int): 
        super(SimpleVariationnalOutputBase, self).__init__() 
        self.output_size = output_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.decoder = nn.Sequential(
            linear_layer(
                self.latent_dim,
                self.hidden_dim//4,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
            linear_layer(
                self.hidden_dim//4,
                self.hidden_dim//2,   # c'était 4 ici avant
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
            linear_layer(
                self.hidden_dim//2,
                self.output_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
        )


    def forward(self,inputs : torch.Tensor) -> torch.Tensor:
        decoding = self.decoder(inputs)
        return decoding

    def get_joints_velocity(self, inputs : torch.Tensor) -> torch.Tensor:
        decoding = self.decoder(inputs)
        return decoding[:,self.output_size//2:]

    def get_joints_position(self, inputs : torch.Tensor) -> torch.Tensor:
        decoding = self.decoder(inputs)
        return decoding[:,:self.output_size//2]

