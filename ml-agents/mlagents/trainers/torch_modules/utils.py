from typing import List, Optional, Tuple, Dict
from mlagents.torch_utils import torch, nn, default_device
from mlagents.trainers.torch_modules.layers import LinearEncoder, Initialization
import numpy as np

from mlagents.trainers.torch_modules.encoders import (
    VectorInput,
    SimpleVariationnalInputBase,
)
from mlagents.trainers.settings import TransferSettings, ScheduleType
from mlagents.trainers.exception import UnityTrainerException
from mlagents_envs.base_env import ObservationSpec, DimensionProperty


class ModelUtils:
    # Minimum supported side for each encoder type. If refactoring an encoder, please
    # adjust these also.

    @staticmethod
    def update_learning_rate(optim: torch.optim.Optimizer, lr: float) -> None:
        """
        Apply a learning rate to a torch optimizer.
        :param optim: Optimizer
        :param lr: Learning rate
        """
        for param_group in optim.param_groups:
            param_group["lr"] = lr

    class DecayedValue:
        def __init__(
            self,
            schedule: ScheduleType,
            initial_value: float,
            min_value: float,
            max_step: int,
        ):
            """
            Object that represnets value of a parameter that should be decayed, assuming it is a function of
            global_step.
            :param schedule: Type of learning rate schedule.
            :param initial_value: Initial value before decay.
            :param min_value: Decay value to this value by max_step.
            :param max_step: The final step count where the return value should equal min_value.
            :param global_step: The current step count.
            :return: The value.
            """
            self.schedule = schedule
            self.initial_value = initial_value
            self.min_value = min_value
            self.max_step = max_step

        def get_value(self, global_step: int) -> float:
            """
            Get the value at a given global step.
            :param global_step: Step count.
            :returns: Decayed value at this global step.
            """
            if self.schedule == ScheduleType.CONSTANT:
                return self.initial_value
            elif self.schedule == ScheduleType.LINEAR:
                return ModelUtils.polynomial_decay(
                    self.initial_value, self.min_value, self.max_step, global_step
                )
            else:
                raise UnityTrainerException(f"The schedule {self.schedule} is invalid.")

    @staticmethod
    def polynomial_decay(
        initial_value: float,
        min_value: float,
        max_step: int,
        global_step: int,
        power: float = 1.0,
    ) -> float:
        """
        Get a decayed value based on a polynomial schedule, with respect to the current global step.
        :param initial_value: Initial value before decay.
        :param min_value: Decay value to this value by max_step.
        :param max_step: The final step count where the return value should equal min_value.
        :param global_step: The current step count.
        :param power: Power of polynomial decay. 1.0 (default) is a linear decay.
        :return: The current decayed value.
        """
        global_step = min(global_step, max_step)
        decayed_value = (initial_value - min_value) * (
            1 - float(global_step) / max_step
        ) ** (power) + min_value
        return decayed_value



    @staticmethod
    def create_input_processors(
        observation_specs: List[ObservationSpec],
        h_size: int,
        transfer_settings : TransferSettings,
        normalize: bool = False,
        is_critic:bool = False
    ) -> Tuple[nn.ModuleList, List[int]]:
        """
        Creates visual and vector encoders, along with their normalizers.
        :param observation_specs: List of ObservationSpec that represent the observation dimensions.
        :param action_size: Number of additional un-normalized inputs to each vector encoder. Used for
            conditioning network on other values (e.g. actions for a Q function)
        :param h_size: Number of hidden units per layer excluding attention layers.
        :param attention_embedding_size: Number of hidden units per attention layer.
        :param vis_encode_type: Type of visual encoder to use.
        :param unnormalized_inputs: Vector inputs that should not be normalized, and added to the vector
            obs.
        :param normalize: Normalize all vector inputs.
        :return: Tuple of :
         - ModuleList of the encoders
         - A list of embedding sizes (0 if the input requires to be processed with a variable length
         observation encoder)
        """
        encoders: List[nn.Module] = []
        embedding_sizes: List[int] = []
        obs_spec = observation_specs[0]
        if transfer_settings.use_bases and not is_critic:
            # Load the model of the ouput base
            base_in = SimpleVariationnalInputBase(
                input_size = transfer_settings.state_dim,
                latent_dim = transfer_settings.latent_dim,
                hidden_dim = transfer_settings.hidden_units
            )
            base_in.load_state_dict(torch.load(transfer_settings.base_in_path, map_location=torch.device(default_device())),strict = True)
            print("BASE IN WEIGHTS LOADED")
            encoders.append(base_in) 
            obs_input_dim = transfer_settings.task_obs_dim+transfer_settings.latent_dim
            encoders.append(VectorInput(obs_input_dim, normalize))
            return (nn.ModuleList(encoders), obs_input_dim)
        else :
            encoders.append(VectorInput(obs_spec.shape[0], normalize))
            return (nn.ModuleList(encoders), obs_spec.shape[0])
        

    @staticmethod
    def list_to_tensor(ndarray_list: List[np.ndarray], dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
        """
        Converts a list of numpy arrays into a tensor. MUCH faster than
        calling as_tensor on the list directly.
        """
        return torch.as_tensor(np.asanyarray(ndarray_list), dtype=dtype)

    @staticmethod
    def list_to_tensor_list(
        ndarray_list: List[np.ndarray], dtype: Optional[torch.dtype] = torch.float32
    ) -> torch.Tensor:
        """
        Converts a list of numpy arrays into a list of tensors. MUCH faster than
        calling as_tensor on the list directly.
        """
        return [
            torch.as_tensor(np.asanyarray(_arr), dtype=dtype) for _arr in ndarray_list
        ]

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a Torch Tensor to a numpy array. If the Tensor is on the GPU, it will
        be brought to the CPU.
        """
        return tensor.detach().cpu().numpy()


    @staticmethod
    def trust_region_value_loss(
        values: Dict[str, torch.Tensor],
        old_values: Dict[str, torch.Tensor],
        returns: Dict[str, torch.Tensor],
        epsilon: float,
    ) -> torch.Tensor:
        """
        Evaluates value loss, clipping to stay within a trust region of old value estimates.
        Used for PPO and POCA.
        :param values: Value output of the current network.
        :param old_values: Value stored with experiences in buffer.
        :param returns: Computed returns.
        :param epsilon: Clipping value for value estimate.
        :param loss_mask: Mask for losses. Used with LSTM to ignore 0'ed out experiences.
        """
        value_losses = []
        for name, head in values.items():
            old_val_tensor = old_values[name]
            returns_tensor = returns[name]
            clipped_value_estimate = old_val_tensor + torch.clamp(
                head - old_val_tensor, -1 * epsilon, epsilon
            )
            v_opt_a = (returns_tensor - head) ** 2
            v_opt_b = (returns_tensor - clipped_value_estimate) ** 2
            value_loss = torch.max(v_opt_a, v_opt_b).mean()
            value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    @staticmethod
    def trust_region_policy_loss(
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Evaluate policy loss clipped to stay within a trust region. Used for PPO and POCA.
        :param advantages: Computed advantages.
        :param log_probs: Current policy probabilities
        :param old_log_probs: Past policy probabilities
        :param loss_masks: Mask for losses. Used with LSTM to ignore 0'ed out experiences.
        """
        advantage = advantages.unsqueeze(-1)
        r_theta = torch.exp(log_probs - old_log_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = torch.clamp(r_theta, 1.0 - epsilon, 1.0 + epsilon) * advantage
        policy_loss = -1 * torch.min(p_opt_a, p_opt_b).mean()
        return policy_loss
