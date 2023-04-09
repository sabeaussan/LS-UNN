from typing import List, Tuple, NamedTuple, Optional
from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_modules.distributions import (
    DistInstance,
    DiscreteDistInstance,
    GaussianDistribution,
    MultiCategoricalDistribution,
)
from mlagents.trainers.torch_modules.agent_action import AgentAction
from mlagents.trainers.torch_modules.action_log_probs import ActionLogProbs
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.settings import TransferSettings

EPSILON = 1e-7  # Small value to avoid divide by zero


class DistInstances(NamedTuple):
    """
    A NamedTuple with fields corresponding the the DistInstance objects
    output by continuous and discrete distributions, respectively. Discrete distributions
    output a list of DistInstance objects whereas continuous distributions output a single
    DistInstance object.
    """

    continuous: Optional[DistInstance]
    discrete: Optional[List[DiscreteDistInstance]]


class ActionModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        action_size: int,
        transfer_settings : TransferSettings,
        tanh_squash: bool = False,
        deterministic: bool = False,
    ):
        """
        A torch module that represents the action space of a policy. The ActionModel may contain
        a continuous distribution, a discrete distribution or both where construction depends on
        the action_spec.  The ActionModel uses the encoded input of the network body to parameterize
        these distributions. The forward method of this module outputs the action, log probs,
        and entropies given the encoding from the network body.
        :params hidden_size: Size of the input to the ActionModel.
        :params action_spec: The ActionSpec defining the action space dimensions and distributions.
        :params conditional_sigma: Whether or not the std of a Gaussian is conditioned on state.
        :params tanh_squash: Whether to squash the output of a Gaussian with the tanh function.
        :params deterministic: Whether to select actions deterministically in policy.
        """
        super().__init__()
        self.encoding_size = hidden_size
        self._continuous_distribution = None

        self._continuous_distribution = GaussianDistribution(
            self.encoding_size,
            action_size,
            tanh_squash=tanh_squash,
        )

        # During training, clipping is done in TorchPolicy, but we need to clip before ONNX
        # export as well.
        self._clip_action_on_export = not tanh_squash
        self._deterministic = deterministic

    def _sample_action(self, dists: DistInstances) -> AgentAction:
        """
        Samples actions from a DistInstances tuple
        :params dists: The DistInstances tuple
        :return: An AgentAction corresponding to the actions sampled from the DistInstances
        """

        continuous_action: Optional[torch.Tensor] = None
        # This checks None because mypy complains otherwise
        if self._deterministic:
            continuous_action = dists.continuous.deterministic_sample()
        else:
            continuous_action = dists.continuous.sample()
        return AgentAction(continuous_action)

    def _get_dists(self, inputs: torch.Tensor) -> DistInstances:
        """
        Creates a DistInstances tuple using the continuous and discrete distributions
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :return: A DistInstances tuple
        """
        continuous_dist: Optional[DistInstance] = None
        # This checks None because mypy complains otherwise
        continuous_dist = self._continuous_distribution(inputs)
        return DistInstances(continuous_dist, None)

    def _get_probs_and_entropy(self, actions: AgentAction, dists: DistInstances) -> Tuple[ActionLogProbs, torch.Tensor]:
        """
        Computes the log probabilites of the actions given distributions and entropies of
        the given distributions.
        :params actions: The AgentAction
        :params dists: The DistInstances tuple
        :return: An ActionLogProbs tuple and a torch tensor of the distribution entropies.
        """
        entropies_list: List[torch.Tensor] = []
        continuous_log_prob: Optional[torch.Tensor] = None

        # This checks None because mypy complains otherwise
        if dists.continuous is not None:
            continuous_log_prob = dists.continuous.log_prob(actions.continuous_tensor)
            entropies_list.append(dists.continuous.entropy())
        action_log_probs = ActionLogProbs(continuous_log_prob, None, None)
        entropies = torch.cat(entropies_list, dim=1)
        return action_log_probs, entropies

    def evaluate(self, inputs: torch.Tensor, actions: AgentAction) -> Tuple[ActionLogProbs, torch.Tensor]:
        """
        Given actions and encoding from the network body, gets the distributions and
        computes the log probabilites and entropies.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :params actions: The AgentAction
        :return: An ActionLogProbs tuple and a torch tensor of the distribution entropies.
        """
        dists = self._get_dists(inputs)
        log_probs, entropies = self._get_probs_and_entropy(actions, dists)
        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)
        return log_probs, entropy_sum


    def forward(self, inputs: torch.Tensor) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor]:
        """
        The forward method of this module. Outputs the action, log probs,
        and entropies given the encoding from the network body.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :return: Given the input, an AgentAction of the actions generated by the policy and the corresponding
        ActionLogProbs and entropies.
        """
        dists = self._get_dists(inputs)
        actions = self._sample_action(dists)
        log_probs, entropies = self._get_probs_and_entropy(actions, dists)
        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)
        return (actions, log_probs, entropy_sum)
