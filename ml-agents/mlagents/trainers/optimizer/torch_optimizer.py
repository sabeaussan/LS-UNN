from typing import Dict, Optional, Tuple, List
from mlagents.torch_utils import torch
import numpy as np
from collections import defaultdict

from mlagents.trainers.buffer import AgentBuffer, AgentBufferField
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_modules.components.reward_providers import create_reward_provider

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    RewardSignalSettings,
    RewardSignalType,
)
from mlagents.trainers.torch_modules.utils import ModelUtils


class TorchOptimizer(Optimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__()
        self.policy = policy
        self.trainer_settings = trainer_settings
        self.update_dict: Dict[str, torch.Tensor] = {}
        self.value_heads: Dict[str, torch.Tensor] = {}
        self.m_size: int = 0
        self.global_step = torch.tensor(0)
        self.create_reward_signals(trainer_settings.reward_signals)


    @property
    def critic(self):
        raise NotImplementedError

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        pass

    def create_reward_signals(
        self, reward_signal_configs: Dict[RewardSignalType, RewardSignalSettings]
    ) -> None:
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        for reward_signal, settings in reward_signal_configs.items():
            # Name reward signals by string in case we have duplicates later
            self.reward_signals[reward_signal.value] = create_reward_provider(
                reward_signal, self.policy.behavior_spec, settings
            )

    

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Get value estimates and memories for a trajectory, in batch form.
        :param batch: An AgentBuffer that consists of a trajectory.
        :param next_obs: the next observation (after the trajectory). Used for boostrapping
            if this is not a termiinal trajectory.
        :param done: Set true if this is a terminal trajectory.
        :param agent_id: Agent ID of the agent that this trajectory belongs to.
        :returns: A Tuple of the Value Estimates as a Dict of [name, np.ndarray(trajectory_len)],
            the final value estimate as a Dict of [name, float], and optionally (if using memories)
            an AgentBufferField of initial critic memories to be used during update.
        """
        n_obs = len(self.policy.behavior_spec.observation_specs)


        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)]
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        # To prevent memory leak and improve performance, evaluate with no_grad.
        with torch.no_grad():
            value_estimates = self.critic.critic_pass(current_obs)

        next_value_estimate = self.critic.critic_pass(next_obs)

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)
            next_value_estimate[name] = ModelUtils.to_numpy(next_value_estimate[name])

        if done:
            for k in next_value_estimate:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimate[k] = 0.0
        return value_estimates, next_value_estimate
