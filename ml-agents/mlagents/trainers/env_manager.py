from abc import ABC, abstractmethod

from typing import List, Dict, NamedTuple, Iterable, Tuple
from mlagents_envs.base_env import (
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    BehaviorName,
)
from mlagents_envs.side_channel.stats_side_channel import EnvironmentStats

from mlagents.trainers.policy import Policy
from mlagents.trainers.agent_processor import AgentManager, AgentManagerQueue
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.settings import TrainerSettings
from mlagents_envs.logging_util import get_logger

AllStepResult = Tuple[DecisionSteps, TerminalSteps] # enlever le behavior
AllGroupSpec = Dict[BehaviorName, BehaviorSpec]

logger = get_logger(__name__)


class EnvironmentStep(NamedTuple):
    current_all_step_result: AllStepResult
    worker_id: int
    action_info: ActionInfo
    environment_stats: EnvironmentStats


    @staticmethod
    def empty(worker_id: int) -> "EnvironmentStep":
        return EnvironmentStep({}, worker_id, {}, {})


class EnvManager(ABC):
    def __init__(self):
        self.policy: Policy = None
        self.agent_manager: AgentManager = None
        self.first_step_infos: List[EnvironmentStep] = []

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy
        self.agent_manager.policy = policy

    def set_agent_manager(self, manager: AgentManager) -> None:
        self.agent_manager = manager

    @abstractmethod
    def _step(self) -> List[EnvironmentStep]:
        pass

    @abstractmethod
    def _reset_env(self, config: Dict = None) -> List[EnvironmentStep]:
        pass

    def reset(self, config: Dict = None) -> int:
        if self.agent_manager is not None : 
            self.agent_manager.end_episode()
        # Save the first step infos, after the reset.
        # They will be processed on the first advance().
        self.first_step_infos = self._reset_env()
        return len(self.first_step_infos)

    @property
    @abstractmethod
    def training_behaviors(self) -> BehaviorSpec:
        pass

    @abstractmethod
    def close(self):
        pass

    def get_steps(self) -> List[EnvironmentStep]:
        """
        Updates the policies, steps the environments, and returns the step information from the environments.
        Calling code should pass the returned EnvironmentSteps to process_steps() after calling this.
        :return: The list of EnvironmentSteps
        """
        # If we had just reset, process the first EnvironmentSteps.
        # Note that we do it here instead of in reset() so that on the very first reset(),
        # we can create the needed AgentManagers before calling advance() and processing the EnvironmentSteps.
        if self.first_step_infos:
            self._process_step_infos(self.first_step_infos)
            self.first_step_infos = []

        # Get new policies if found. Always get the latest policy.
        _policy = None
        try:
            # We make sure to empty the policy queue before continuing to produce steps.
            # This halts the trainers until the policy queue is empty.
            while True:
                _policy = self.agent_manager.policy_queue.get_nowait()
        except AgentManagerQueue.Empty:
            if _policy is not None:
                self.set_policy(_policy)

        # Step the environments
        new_step_infos = self._step()
        return new_step_infos

    def process_steps(self, new_step_infos: List[EnvironmentStep]) -> int:
        # Add to AgentProcessor
        num_step_infos = self._process_step_infos(new_step_infos)
        return num_step_infos

    def _process_step_infos(self, step_infos: List[EnvironmentStep]) -> int:
        for step_info in step_infos: 
            decision_steps, terminal_steps = step_info.current_all_step_result
            self.agent_manager.add_experiences(
                decision_steps,
                terminal_steps,
                step_info.worker_id,
                step_info.action_info if step_info.action_info else ActionInfo.empty(),
            )

            self.agent_manager.record_environment_stats(step_info.environment_stats, step_info.worker_id)
        return len(step_infos)
