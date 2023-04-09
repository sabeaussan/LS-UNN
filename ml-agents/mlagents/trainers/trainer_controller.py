# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
from typing import Dict, Set, List
from collections import defaultdict

import numpy as np

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from mlagents_envs.timers import (
    hierarchical_timer,
    timed,
    merge_gauges,
)
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.agent_processor import AgentManager
from mlagents import torch_utils
from mlagents.torch_utils.globals import get_rank


class TrainerController:
    def __init__(
        self,
        trainer_factory: TrainerFactory,
        output_path: str,
        run_id: str,
        train: bool,
        training_seed: int,
    ):
        """
        :param output_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param param_manager: EnvironmentParameterManager object which stores information about all
        environment parameters.
        :param train: Whether to train model, or only run inference.
        :param training_seed: Seed to use for Numpy and Torch random number generation.
        """
        self.trainer: Trainer = None
        self.trainer_factory = trainer_factory
        self.output_path = output_path
        self.logger = get_logger(__name__)
        self.run_id = run_id
        self.train_model = train
        self.kill_trainers = False
        np.random.seed(training_seed)
        torch_utils.torch.manual_seed(training_seed)
        self.rank = get_rank()

    @timed
    def _save_models(self):
        """
        Saves current model to checkpoint folder.
        """
        if self.rank is not None and self.rank != 0:
            return

        self.trainer.save_model()
        self.logger.debug("Saved Model")

    @staticmethod
    def _create_output_path(output_path):
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except Exception:
            raise UnityEnvironmentException(
                f"The folder {output_path} containing the "
                "generated model could not be "
                "accessed. Please make sure the "
                "permissions are set correctly."
            )

    @timed
    def _reset_env(self, env_manager: EnvManager) -> None:
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        env_manager.reset()  # -> fait appel à la fonction reset de EnvManager -> restet_env de SubProc
        self._create_trainer_and_manager(env_manager)

    def _not_done_training(self) -> bool:
        return (self.trainer.should_still_train or not self.train_model) or self.trainer is None

    def _create_trainer_and_manager(self, env_manager: EnvManager) -> None:

        self.trainer = self.trainer_factory.generate()  # retourne un PPOTrainer
        # Pourquoi pas directement l'ajouter au trainer ?!!!
        policy = self.trainer.create_policy(env_manager.training_behaviors)
        self.trainer.add_policy(policy)

        agent_manager = AgentManager(
            policy,
            self.trainer.stats_reporter,
            self.trainer.parameters.time_horizon,
        )
        env_manager.set_agent_manager(agent_manager)
        env_manager.set_policy(policy)

        self.trainer.publish_policy_queue(agent_manager.policy_queue)
        self.trainer.subscribe_trajectory_queue(agent_manager.trajectory_queue)



    @timed
    def start_learning(self, env_manager: EnvManager) -> None:
        self._create_output_path(self.output_path)
        try:
            # Initial reset
            self._reset_env(env_manager)
            while self._not_done_training():
                n_steps = self.advance(env_manager)
        except (
            KeyboardInterrupt,
            UnityCommunicationException,
            UnityEnvironmentException,
            UnityCommunicatorStoppedException,
        ) as ex:
            self.logger.info(
                "Learning was interrupted. Please wait while the graph is generated."
            )
            if isinstance(ex, KeyboardInterrupt) or isinstance(
                ex, UnityCommunicatorStoppedException
            ):
                pass
            else:
                # If the environment failed, we want to make sure to raise
                # the exception so we exit the process with an return code of 1.
                raise ex
        finally:
            if self.train_model:
                self._save_models()

    def end_trainer_episodes(self) -> None:
        # Reward buffers reset takes place only for curriculum learning
        # else no reset.
        for trainer in self.trainers.values():
            trainer.end_episode()



    @timed
    def advance(self, env_manager: EnvManager) -> int:
        # Get steps
        with hierarchical_timer("env_step"):
            new_step_infos = env_manager.get_steps()
            num_steps = env_manager.process_steps(new_step_infos)

        with hierarchical_timer("trainer_advance"):
                self.trainer.advance()

        return num_steps
