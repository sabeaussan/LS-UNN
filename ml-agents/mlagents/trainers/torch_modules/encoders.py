from typing import Tuple, Optional, Union

from mlagents.trainers.torch_modules.layers import linear_layer, Initialization, Swish

from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_modules.model_serialization import exporting_to_onnx


class Normalizer(nn.Module):
    def __init__(self, vec_obs_size: int):
        super().__init__()
        self.register_buffer("normalization_steps", torch.tensor(1))
        self.register_buffer("running_mean", torch.zeros(vec_obs_size))
        self.register_buffer("running_variance", torch.ones(vec_obs_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized_state = torch.clamp(
            (inputs - self.running_mean)
            / torch.sqrt(self.running_variance / self.normalization_steps),
            -5,
            5,
        )
        return normalized_state

    def update(self, vector_input: torch.Tensor) -> None:
        with torch.no_grad():
            steps_increment = vector_input.size()[0]
            total_new_steps = self.normalization_steps + steps_increment

            input_to_old_mean = vector_input - self.running_mean
            new_mean: torch.Tensor = self.running_mean + (
                input_to_old_mean / total_new_steps
            ).sum(0)

            input_to_new_mean = vector_input - new_mean
            new_variance = self.running_variance + (
                input_to_new_mean * input_to_old_mean
            ).sum(0)
            # Update references. This is much faster than in-place data update.
            self.running_mean: torch.Tensor = new_mean
            self.running_variance: torch.Tensor = new_variance
            self.normalization_steps: torch.Tensor = total_new_steps

    def copy_from(self, other_normalizer: "Normalizer") -> None:
        self.normalization_steps.data.copy_(other_normalizer.normalization_steps.data)
        self.running_mean.data.copy_(other_normalizer.running_mean.data)
        self.running_variance.copy_(other_normalizer.running_variance.data)


def conv_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a convolution layer.
    kernel_size, stride, padding and dilation correspond to the inputs of the
    torch.nn.Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    :param h_w: The height and width of the input.
    :param kernel_size: The size of the kernel of the convolution (can be an int or a
    tuple [width, height])
    :param stride: The stride of the convolution
    :param padding: The padding of the convolution
    :param dilation: The dilation of the convolution
    """
    from math import floor

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))
    h = floor(
        ((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


def pool_out_shape(h_w: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a max pooling layer.
    kernel_size corresponds to the inputs of the
    torch.nn.MaxPool2d layer (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    :param kernel_size: The size of the kernel of the convolution
    """
    height = (h_w[0] - kernel_size) // 2 + 1
    width = (h_w[1] - kernel_size) // 2 + 1
    return height, width


class VectorInput(nn.Module):
    def __init__(self, input_size: int, normalize: bool = False):
        super().__init__()
        self.normalizer: Optional[Normalizer] = None
        if normalize:
            self.normalizer = Normalizer(input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            inputs = self.normalizer(inputs)
        return inputs

    def copy_normalization(self, other_input: "VectorInput") -> None:
        if self.normalizer is not None and other_input.normalizer is not None:
            self.normalizer.copy_from(other_input.normalizer)

    def update_normalization(self, inputs: torch.Tensor) -> None:
        if self.normalizer is not None:
            self.normalizer.update(inputs)


class SimpleVariationnalInputBase(nn.Module):
    # mettre dans la base out une fonction "get_velocity" et "get_position"
    def __init__(self,input_size : int, latent_dim : int ,hidden_dim : int): 
        super(SimpleVariationnalInputBase, self).__init__() 
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            linear_layer(
                self.input_size,
                self.hidden_dim,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
            linear_layer(
                self.hidden_dim,
                self.hidden_dim//2,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
            linear_layer(
                self.hidden_dim//2,
                self.hidden_dim//4,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
        )

        self.mu_head = nn.Sequential(
            linear_layer(
                self.hidden_dim//4,
                self.latent_dim,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            )
        )
        self.logvar_head = nn.Sequential(
            linear_layer(
                self.hidden_dim//4,
                self.latent_dim,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            )
        )

    def forward(self,inputs : torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        encoding = self.encoder(inputs)
        mu = self.mu_head(encoding)
        logvar = self.logvar_head(encoding)
        latent_distribution = torch.distributions.normal.Normal(mu, torch.exp(0.5*logvar))
        z = latent_distribution.rsample()
        return z, mu, logvar


    def get_mu(self,inputs : torch.Tensor):
        encoding = self.encoder(inputs)
        return self.mu_head(encoding)
