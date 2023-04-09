import os.path
import warnings

import attr
import cattr
from typing import (
    Dict,
    Optional,
    List,
    Any,
    DefaultDict,
    Mapping,
    Tuple,
    Union,
    ClassVar,
)
from enum import Enum
import collections
import argparse
import abc
import numpy as np
import math
import copy

from mlagents.trainers.cli_utils import StoreConfigFile, DetectDefault, parser
from mlagents.trainers.cli_utils import load_config
from mlagents.trainers.exception import TrainerConfigError, TrainerConfigWarning

from mlagents_envs import logging_util
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

logger = logging_util.get_logger(__name__)


def check_and_structure(key: str, value: Any, class_type: type) -> Any:
    attr_fields_dict = attr.fields_dict(class_type)
    if key not in attr_fields_dict:
        raise TrainerConfigError(
            f"The option {key} was specified in your YAML file for {class_type.__name__}, but is invalid."
        )
    # Apply cattr structure to the values
    return cattr.structure(value, attr_fields_dict[key].type)


class TrainerType(Enum):
    PPO: str = "ppo"
    SAC: str = "sac"

    def to_settings(self) -> type:
        _mapping = {
            TrainerType.PPO: PPOSettings,
            TrainerType.SAC: SACSettings,
        }
        return _mapping[self]


def check_hyperparam_schedules(val: Dict, trainer_type: TrainerType) -> Dict:
    # Check if beta and epsilon are set. If not, set to match learning rate schedule.
    if trainer_type is TrainerType.PPO:
        if "beta_schedule" not in val.keys() and "learning_rate_schedule" in val.keys():
            val["beta_schedule"] = val["learning_rate_schedule"]
        if (
            "epsilon_schedule" not in val.keys()
            and "learning_rate_schedule" in val.keys()
        ):
            val["epsilon_schedule"] = val["learning_rate_schedule"]
    return val


def strict_to_cls(d: Mapping, t: type) -> Any:
    if not isinstance(d, Mapping):
        raise TrainerConfigError(f"Unsupported config {d} for {t.__name__}.")
    d_copy: Dict[str, Any] = {}
    d_copy.update(d)
    for key, val in d_copy.items():
        d_copy[key] = check_and_structure(key, val, t)
    return t(**d_copy)


def defaultdict_to_dict(d: DefaultDict) -> Dict:
    return {key: cattr.unstructure(val) for key, val in d.items()}


def deep_update_dict(d: Dict, update_d: Mapping) -> None:
    """
    Similar to dict.update(), but works for nested dicts of dicts as well.
    """
    for key, val in update_d.items():
        if key in d and isinstance(d[key], Mapping) and isinstance(val, Mapping):
            deep_update_dict(d[key], val)
        else:
            d[key] = val


class SerializationSettings:
    convert_to_onnx = True
    onnx_opset = 9


@attr.s(auto_attribs=True)
class ExportableSettings:
    def as_dict(self):
        return cattr.unstructure(self)


class ScheduleType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


@attr.s(auto_attribs=True)
class NetworkSettings:
    normalize: bool = False
    hidden_units: int = 128
    num_layers: int = 2
    deterministic: bool = parser.get_default("deterministic")



@attr.s(auto_attribs=True)
class HyperparamSettings:
    batch_size: int = 1024
    buffer_size: int = 10240
    learning_rate: float = 3.0e-4
    learning_rate_schedule: ScheduleType = ScheduleType.CONSTANT


@attr.s(auto_attribs=True)
class PPOSettings(HyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR



@attr.s(auto_attribs=True)
class SACSettings(HyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    init_entcoef: float = 1.0
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


@attr.s(auto_attribs=True)
class TransferSettings:
    use_bases : bool = False
    fine_tune : bool = False
    base_in_path: str = "dummy"
    base_out_path: str = "dummy" 
    task_obs_dim: int = attr.ib(default=0)
    state_dim: int =  attr.ib(default=0)
    latent_dim: int = attr.ib(default=0)
    hidden_units: int = 256

    @task_obs_dim.validator
    def validate_task_obs_dim(self, attribute, value):
        if value <= 0 and self.use_bases:
            raise ValueError(" task_obs_dim should be > 0")

    @state_dim.validator
    def validate_state_dim(self, attribute, value):
        if value <= 0 and self.use_bases:
            raise ValueError(" state_dim should be > 0")

    @latent_dim.validator
    def validate_latent_dim(self, attribute, value):
        if value <= 0 and self.use_bases:
            raise ValueError(" latent_dim should be > 0")

# INTRINSIC REWARD SIGNALS #############################################################
class RewardSignalType(Enum):
    EXTRINSIC: str = "extrinsic"

    def to_settings(self) -> type:
        _mapping = {
            RewardSignalType.EXTRINSIC: RewardSignalSettings,
        }
        return _mapping[self]


@attr.s(auto_attribs=True)
class RewardSignalSettings:
    gamma: float = 0.99
    strength: float = 1.0
    network_settings: NetworkSettings = attr.ib(factory=NetworkSettings)

    @staticmethod
    def structure(d: Mapping, t: type) -> Any:
        """
        Helper method to structure a Dict of RewardSignalSettings class. Meant to be registered with
        cattr.register_structure_hook() and called with cattr.structure(). This is needed to handle
        the special Enum selection of RewardSignalSettings classes.
        """
        if not isinstance(d, Mapping):
            raise TrainerConfigError(f"Unsupported reward signal configuration {d}.")
        d_final: Dict[RewardSignalType, RewardSignalSettings] = {}
        for key, val in d.items():
            enum_key = RewardSignalType(key)
            t = enum_key.to_settings()
            d_final[enum_key] = strict_to_cls(val, t)
            # Checks to see if user specifying deprecated encoding_size for RewardSignals.
            # If network_settings is not specified, this updates the default hidden_units
            # to the value of encoding size. If specified, this ignores encoding size and
            # uses network_settings values.
            if "encoding_size" in val:
                logger.warning(
                    "'encoding_size' was deprecated for RewardSignals. Please use network_settings."
                )
                # If network settings was not specified, use the encoding size. Otherwise, use hidden_units
                if "network_settings" not in val:
                    d_final[enum_key].network_settings.hidden_units = val[
                        "encoding_size"
                    ]
        return d_final



# TRAINERS #############################################################################
@attr.s(auto_attribs=True)
class TrainerSettings(ExportableSettings):
    default_override: ClassVar[Optional["TrainerSettings"]] = None
    trainer_type: TrainerType = TrainerType.PPO
    hyperparameters: HyperparamSettings = attr.ib()

    @hyperparameters.default
    def _set_default_hyperparameters(self):
        return self.trainer_type.to_settings()()

    network_settings: NetworkSettings = attr.ib(factory=NetworkSettings)
    transfer_settings: TransferSettings = attr.ib(factory=TransferSettings)
    reward_signals: Dict[RewardSignalType, RewardSignalSettings] = attr.ib(
        factory=lambda: {RewardSignalType.EXTRINSIC: RewardSignalSettings()}
    )
    init_path: Optional[str] = None
    keep_checkpoints: int = 5
    checkpoint_interval: int = 500000
    max_steps: int = 500000
    time_horizon: int = 64
    summary_freq: int = 50000
    cattr.register_structure_hook(
        Dict[RewardSignalType, RewardSignalSettings], RewardSignalSettings.structure
    )

    @staticmethod
    def dict_to_trainerdict(d: Dict, t: type) -> "TrainerSettings.DefaultTrainerDict":
        return TrainerSettings.DefaultTrainerDict(
            cattr.structure(d, Dict[str, TrainerSettings])
        )

    @staticmethod
    def structure(d: Mapping, t: type) -> Any:
        """
        Helper method to structure a TrainerSettings class. Meant to be registered with
        cattr.register_structure_hook() and called with cattr.structure().
        """

        if not isinstance(d, Mapping):
            raise TrainerConfigError(f"Unsupported config {d} for {t.__name__}.")

        d_copy: Dict[str, Any] = {}

        # Check if a default_settings was specified. If so, used those as the default
        # rather than an empty dict.
        if TrainerSettings.default_override is not None:
            d_copy.update(cattr.unstructure(TrainerSettings.default_override))

        deep_update_dict(d_copy, d)

        if "framework" in d_copy:
            logger.warning("Framework option was deprecated but was specified")
            d_copy.pop("framework", None)

        for key, val in d_copy.items():
            if attr.has(type(val)):
                # Don't convert already-converted attrs classes.
                continue
            if key == "hyperparameters":
                if "trainer_type" not in d_copy:
                    raise TrainerConfigError(
                        "Hyperparameters were specified but no trainer_type was given."
                    )
                else:
                    d_copy[key] = check_hyperparam_schedules(
                        val, d_copy["trainer_type"]
                    )
                    d_copy[key] = strict_to_cls(
                        d_copy[key], TrainerType(d_copy["trainer_type"]).to_settings()
                    )
            elif key == "max_steps":
                d_copy[key] = int(float(val))
                # In some legacy configs, max steps was specified as a float
            else:
                d_copy[key] = check_and_structure(key, val, t)
        return t(**d_copy)

    class DefaultTrainerDict(collections.defaultdict):
        def __init__(self, *args):
            # Depending on how this is called, args may have the defaultdict
            # callable at the start of the list or not. In particular, unpickling
            # will pass [TrainerSettings].
            if args and args[0] == TrainerSettings:
                super().__init__(*args)
            else:
                super().__init__(TrainerSettings, *args)
            self._config_specified = True

        def set_config_specified(self, require_config_specified: bool) -> None:
            self._config_specified = require_config_specified

        def __missing__(self, key: Any) -> "TrainerSettings":
            if TrainerSettings.default_override is not None:
                self[key] = copy.deepcopy(TrainerSettings.default_override)
            elif self._config_specified:
                raise TrainerConfigError(
                    f"The behavior name {key} has not been specified in the trainer configuration. "
                    f"Please add an entry in the configuration file for {key}, or set default_settings."
                )
            else:
                logger.warning(
                    f"Behavior name {key} does not match any behaviors specified "
                    f"in the trainer configuration file. A default configuration will be used."
                )
                self[key] = TrainerSettings()
            return self[key]


# COMMAND LINE #########################################################################
@attr.s(auto_attribs=True)
class CheckpointSettings:
    run_id: str = parser.get_default("run_id")
    initialize_from: Optional[str] = parser.get_default("initialize_from")
    load_model: bool = parser.get_default("load_model")
    resume: bool = parser.get_default("resume")
    force: bool = parser.get_default("force")
    train_model: bool = parser.get_default("train_model")
    inference: bool = parser.get_default("inference")
    results_dir: str = parser.get_default("results_dir")

    @property
    def write_path(self) -> str:
        return os.path.join(self.results_dir, self.run_id)

    @property
    def maybe_init_path(self) -> Optional[str]:
        return (
            os.path.join(self.results_dir, self.initialize_from)
            if self.initialize_from is not None
            else None
        )

    @property
    def run_logs_dir(self) -> str:
        return os.path.join(self.write_path, "run_logs")

    def prioritize_resume_init(self) -> None:
        """Prioritize explicit command line resume/init over conflicting yaml options.
        if both resume/init are set at one place use resume"""
        _non_default_args = DetectDefault.non_default_args
        if "resume" in _non_default_args:
            if self.initialize_from is not None:
                logger.warning(
                    f"Both 'resume' and 'initialize_from={self.initialize_from}' are set!"
                    f" Current run will be resumed ignoring initialization."
                )
                self.initialize_from = parser.get_default("initialize_from")
        elif "initialize_from" in _non_default_args:
            if self.resume:
                logger.warning(
                    f"Both 'resume' and 'initialize_from={self.initialize_from}' are set!"
                    f" {self.run_id} is initialized_from {self.initialize_from} and resume will be ignored."
                )
                self.resume = parser.get_default("resume")
        elif self.resume and self.initialize_from is not None:
            # no cli args but both are set in yaml file
            logger.warning(
                f"Both 'resume' and 'initialize_from={self.initialize_from}' are set in yaml file!"
                f" Current run will be resumed ignoring initialization."
            )
            self.initialize_from = parser.get_default("initialize_from")


@attr.s(auto_attribs=True)
class EnvironmentSettings:
    env_path: Optional[str] = parser.get_default("env_path")
    env_args: Optional[List[str]] = parser.get_default("env_args")
    base_port: int = parser.get_default("base_port")
    num_envs: int = attr.ib(default=parser.get_default("num_envs"))
    num_areas: int = attr.ib(default=parser.get_default("num_areas"))
    seed: int = parser.get_default("seed")
    max_lifetime_restarts: int = parser.get_default("max_lifetime_restarts")
    restarts_rate_limit_n: int = parser.get_default("restarts_rate_limit_n")
    restarts_rate_limit_period_s: int = parser.get_default(
        "restarts_rate_limit_period_s"
    )

    @num_envs.validator
    def validate_num_envs(self, attribute, value):
        if value > 1 and self.env_path is None:
            raise ValueError("num_envs must be 1 if env_path is not set.")

    @num_areas.validator
    def validate_num_area(self, attribute, value):
        if value <= 0:
            raise ValueError("num_areas must be set to a positive number >= 1.")


@attr.s(auto_attribs=True)
class EngineSettings:
    width: int = parser.get_default("width")
    height: int = parser.get_default("height")
    quality_level: int = parser.get_default("quality_level")
    time_scale: float = parser.get_default("time_scale")
    target_frame_rate: int = parser.get_default("target_frame_rate")
    capture_frame_rate: int = parser.get_default("capture_frame_rate")
    no_graphics: bool = parser.get_default("no_graphics")


@attr.s(auto_attribs=True)
class TorchSettings:
    device: Optional[str] = parser.get_default("device")


@attr.s(auto_attribs=True)
class RunOptions(ExportableSettings):

    default_settings: Optional[TrainerSettings] = None
    behaviors: TrainerSettings.DefaultTrainerDict = attr.ib(factory=TrainerSettings.DefaultTrainerDict)
    env_settings: EnvironmentSettings = attr.ib(factory=EnvironmentSettings)
    engine_settings: EngineSettings = attr.ib(factory=EngineSettings)
    checkpoint_settings: CheckpointSettings = attr.ib(factory=CheckpointSettings)
    torch_settings: TorchSettings = attr.ib(factory=TorchSettings)

    # These are options that are relevant to the run itself, and not the engine or environment.
    # They will be left here.
    debug: bool = parser.get_default("debug")

    # Convert to settings while making sure all fields are valid
    cattr.register_structure_hook(EnvironmentSettings, strict_to_cls)
    cattr.register_structure_hook(EngineSettings, strict_to_cls)
    cattr.register_structure_hook(CheckpointSettings, strict_to_cls)
    cattr.register_structure_hook(TrainerSettings, TrainerSettings.structure)
    cattr.register_structure_hook(
        TrainerSettings.DefaultTrainerDict, TrainerSettings.dict_to_trainerdict
    )
    cattr.register_unstructure_hook(collections.defaultdict, defaultdict_to_dict)

    @staticmethod
    def from_argparse(args: argparse.Namespace) -> "RunOptions":
        """
        Takes an argparse.Namespace as specified in `parse_command_line`, loads input configuration files
        from file paths, and converts to a RunOptions instance.
        :param args: collection of command-line parameters passed to mlagents-learn
        :return: RunOptions representing the passed in arguments, with trainer config, curriculum and sampler
          configs loaded from files.
        """
        argparse_args = vars(args)
        config_path = StoreConfigFile.trainer_config_path

        # Load YAML
        configured_dict: Dict[str, Any] = {
            "checkpoint_settings": {},
            "env_settings": {},
            "engine_settings": {},
            "torch_settings": {},
        }
        _require_all_behaviors = True
        if config_path is not None:
            configured_dict.update(load_config(config_path))
        else:
            # If we're not loading from a file, we don't require all behavior names to be specified.
            _require_all_behaviors = False

        # Use the YAML file values for all values not specified in the CLI.
        for key in configured_dict.keys():
            # Detect bad config options
            if key not in attr.fields_dict(RunOptions):
                raise TrainerConfigError(
                    "The option {} was specified in your YAML file, but is invalid.".format(key)
                )


        # Override with CLI args
        # Keep deprecated --load working, TODO: remove
        argparse_args["resume"] = argparse_args["resume"] or argparse_args["load_model"]

        for key, val in argparse_args.items():
            if key == "ckpt_name":
                if val is not None :
                    configured_dict["behaviors"]["Behavior"]["init_path"] = val
                    continue
            if key == "initialize_from":
                if val is not None :
                    configured_dict["behaviors"]["Behavior"]["transfer_settings"]["fine_tune"] = True
                else :
                    configured_dict["behaviors"]["Behavior"]["transfer_settings"]["fine_tune"] = False

            if key == "unn":
                if val :
                    configured_dict["behaviors"]["Behavior"]["transfer_settings"]["use_bases"] = True
                    continue
                    print()
                    print("================================")
                    print("=*=*= TRAINING A UNN AGENT =*=*=")
                    print("================================")
                    print()
                else :
                    configured_dict["behaviors"]["Behavior"]["transfer_settings"]["use_bases"] = False
                    continue
                    print()
                    print("========================================")
                    print("=*=*= TRAINING A REGULAR PPO AGENT =*=*=")
                    print("========================================")
                    print()
            if key in DetectDefault.non_default_args:
                if key in attr.fields_dict(CheckpointSettings):
                    configured_dict["checkpoint_settings"][key] = val
                elif key in attr.fields_dict(EnvironmentSettings):
                    configured_dict["env_settings"][key] = val
                elif key in attr.fields_dict(EngineSettings):
                    configured_dict["engine_settings"][key] = val
                elif key in attr.fields_dict(TorchSettings):
                    configured_dict["torch_settings"][key] = val
                else:  # Base options
                    configured_dict[key] = val
        final_runoptions = RunOptions.from_dict(configured_dict)
        final_runoptions.checkpoint_settings.prioritize_resume_init()
        # Need check to bypass type checking but keep structure on dict working
        if isinstance(final_runoptions.behaviors, TrainerSettings.DefaultTrainerDict):
            # configure whether or not we should require all behavior names to be found in the config YAML
            final_runoptions.behaviors.set_config_specified(_require_all_behaviors)

        _non_default_args = DetectDefault.non_default_args

        # Prioritize the deterministic mode from the cli for deterministic actions.
        if "deterministic" in _non_default_args:
            for behaviour in final_runoptions.behaviors.keys():
                final_runoptions.behaviors[
                    behaviour
                ].network_settings.deterministic = argparse_args["deterministic"]
        return final_runoptions

    @staticmethod
    def from_dict(options_dict: Dict[str, Any]) -> "RunOptions":
        # If a default settings was specified, set the TrainerSettings class override
        if (
            "default_settings" in options_dict.keys()
            and options_dict["default_settings"] is not None
        ):
            TrainerSettings.default_override = cattr.structure(
                options_dict["default_settings"], TrainerSettings
            )
        return cattr.structure(options_dict, RunOptions)
