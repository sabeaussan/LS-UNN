import numpy as np
import gym
from gym import spaces
import torch


from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs.base_env import ActionTuple,BaseEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment

# rajouter un booleen pour savoir quelle bases on utilise

# Tester avec regularizion term pour pénaliser l'UNN si s'éloigne de 1 et 1.5 de clipping

# remplacer dim task obs par task_obs = obs[self.action_size:]

# TODO : 
#
#

def make_unity_env(env_path,worker_id,no_graphics = True,time_scale = 20.0):
	channel = EngineConfigurationChannel()
	channel.set_configuration_parameters(time_scale = time_scale, width = 600, height = 600)
	EVAL_MODE = "false"
	unity_env = UnityEnvironment(file_name=env_path, seed=1, side_channels=[channel],worker_id = worker_id, no_graphics=no_graphics, additional_args = ["-evalMode",EVAL_MODE])
	env = UnityGymEnvironment(unity_env)
	return env

class UnityGymEnvironment(gym.Env):

	# a extend pour d'autres env plus sophistiqué

	alpha = 0.4

	def __init__(self,unity_env):
		super(UnityGymEnvironment, self).__init__()
		self._env = unity_env

		# step so that the environment contain behavior spec
		unity_env.step()
		behavior_spec = self._env.behavior_specs
		self.behavior_name = list(behavior_spec.keys())[0]
		self.action_size = behavior_spec[self.behavior_name].action_spec.continuous_size
		self.observation_shape = self._env.behavior_specs[self.behavior_name].observation_specs[0].shape
		# Define action and observation space
		self.action_space = spaces.Box(low=-1.0, high=1.0,shape=(self.action_size,), dtype=np.float32)   # <-------------- pas generale le +1 pour pince
		self.observation_space = spaces.Box(low=-1.0, high=1.0,shape=self.observation_shape, dtype=np.float32)





	def reset(self):
		self._env.reset()

		# retrieve the current step of the environment containing the observations
		decision_step,_ = self._env.get_steps()
		env_obs = decision_step.obs[0][0]
		return env_obs



	def step(self, action):
		try :
			# feed unn action to the env and step env
			action = np.array(action).reshape((1, self.action_size))
			action_tuple = ActionTuple()
			action_tuple.add_continuous(action)
			self._env.set_actions(action_tuple)
			self._env.step()

			# retrieve step type
			decision_step, terminal_step = self._env.get_steps()
			done = False

			if len(terminal_step) != 0:
				done = True
				obs = terminal_step.obs[0][0]
				rew = terminal_step.reward[0].item()
			else : 
				done = False
				obs = decision_step.obs[0][0]
				rew = decision_step.reward[0].item()
		except Exception as e:
			pass

		return obs,rew,done,{"step": decision_step}

	def render(self, mode='console'):
		# a voir plus tard 
		pass

	def close(self):
		self._env.close()
