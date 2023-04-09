from mlagents.trainers.torch_modules.encoders import (
    VectorInput,
    SimpleVariationnalInputBase,
)
from mlagents.trainers.torch_modules.layers import LinearEncoder
from mlagents.trainers.torch_modules.decoders import SimpleVariationnalOutputBase
from mlagents.trainers.torch_modules.action_model import ActionModel
from mlagents.trainers.settings import TransferSettings
from mlagents_envs.base_env import ActionSpec
from unity_env_gym import make_unity_env
import torch
import numpy as np
import glob
import re
import argparse
from mlagents.torch_utils import default_device

# add flag for fine tuned model

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_test_episode", type=int, default=100, help="number of episode of tests per checkpoints")
parser.add_argument("-f", "--freq_test_models", type=int, default=2, help="skip checkpoints every freq_test_models")
parser.add_argument("-src", "--src_robot", type=int, default=0, help="UNN model from : Braccio (0), Panda (1), UR10")
parser.add_argument("-tgt", "--tgt_robot", type=int, default=0, help="Test on : Braccio (0), Panda (1), UR10")
parser.add_argument("-t", "--task", type=int, default=0, help="PicknPlace (0), BallCatcher (1), PegInsertion (2)")
parser.add_argument("-g", "--graphic", action='store_true', help="Wether to activate visuals")
parser.add_argument("--bases_id", type=str, default="aligned_00015", help="bases id")
parser.add_argument("-ts", "--time-scale", type=int, default=20, help="Time scale of the simulation (1-20)")
args = parser.parse_args()

@torch.no_grad()
def test_agent_unn(nb_test_episodes):
    rewards_episode = []
    reward_history = []
    reward_tracking = []
    for episode in range(nb_test_episodes):
        obs = env.reset()
        done = False 
        while not done: 
            env.render()
            with torch.no_grad(): 
                robot_obs = obs[:STATE_DIM[robot_idx]]                                                                        
                task_obs = obs[STATE_DIM[robot_idx]:]
                latent_obs = base_in.get_mu(torch.FloatTensor(robot_obs).to(default_device()))
                unn_obs = torch.cat((latent_obs,torch.FloatTensor(task_obs).to(default_device())),dim = 0)
                norm_unn_obs = vector_input(unn_obs)
                encoding = unn_body(norm_unn_obs)
                latent_action,_,_ = unn_action_head(encoding)
                latent_action = (torch.clamp(latent_action.continuous_tensor, -3, 3) / 3 )*1.25
                robot_act = base_out.get_joints_velocity(latent_action[:LATENT_DIM].unsqueeze(0)).cpu()
                if use_gripper :
                    gripper_act = latent_action[-1].unsqueeze(0).cpu()
                    action = np.concatenate((robot_act[0],gripper_act),axis = 0)
                else :
                    action = robot_act
            next_obs, rew, done, _ = env.step(action)

            rewards_episode.append(rew)

            obs = next_obs 


        reward_history.append(np.sum(rewards_episode))
        rewards_episode = []
    return np.mean(reward_history)



def bubbleSort(arr):
    n = len(arr)
    swapped = False
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            return
    return arr

TASKS = ["picknplace", "ball_catcher", "peg_insertion"]
ROBOTS = ["Braccio", "Panda", "UR10"]
ENV = ["PicknPlace", "BallCatcher", "PegInsertion"]
STATE_DIM = [10,14,12]
TASK_DIM = [10, 13, 6]

robot_idx = args.tgt_robot
task_idx = args.task

if task_idx == 0 :
    use_gripper = True
else :
    use_gripper = False

UNN_HIDDEN_DIM = 225
LATENT_DIM = 6

PREFIX = "Behavior-*.pt"
SOURCE_ROBOT_NAME = ROBOTS[args.src_robot]
TARGET_ROBOT_NAME = ROBOTS[args.tgt_robot]
BASES_PATH_in = f"bases/models/Reacher/bases_dim_6/100k/{args.bases_id}/{TARGET_ROBOT_NAME}_input.pth"
BASES_PATH_out = f"bases/models/Reacher/bases_dim_6/100k/{args.bases_id}/{TARGET_ROBOT_NAME}_output.pth"
ENV_NAME = f"envs/{ENV[task_idx]}_{TARGET_ROBOT_NAME}_testing.x86_64"

base_in = SimpleVariationnalInputBase(
    input_size = STATE_DIM[robot_idx],
    latent_dim = LATENT_DIM,
    hidden_dim = 256
)
base_in.load_state_dict(torch.load(BASES_PATH_in, map_location=torch.device(default_device())), strict = True)

base_out = SimpleVariationnalOutputBase(
    output_size = STATE_DIM[robot_idx],
    latent_dim = LATENT_DIM,
    hidden_dim = 256
)
base_out.load_state_dict(torch.load(BASES_PATH_out, map_location=torch.device(default_device())),strict = True)

best_perfs = 0
best_perfs_step = 0

RES_DIR_PATH = f"results/{TASKS[task_idx]}/{TARGET_ROBOT_NAME.lower()}/unn_ft_{SOURCE_ROBOT_NAME.lower()}/checkpoints/"

files_name = glob.glob(RES_DIR_PATH+PREFIX)

if files_name == []:
    print(f"/!\ /!\ {RES_DIR_PATH} not found /!\ /!\ ")
    raise

steps = bubbleSort([int(re.search('(?<=Behavior-).*?(?=.pt)',step).group(0)) for step in files_name])


print("* Total number of steps : ",len(steps))

if task_idx == 0 :
    use_gripper = True
else :
    use_gripper = False

unn_obs_dim = LATENT_DIM + TASK_DIM[task_idx]

print()
print(f"============== TESTING {SOURCE_ROBOT_NAME} UNN fine tuned on {TARGET_ROBOT_NAME} robot ================")
print()


for step in steps[::args.freq_test_models] :

    ############## For every chkpt models #################
    print(f"-- step {step} --")
    vector_input = VectorInput(input_size = unn_obs_dim, normalize = True)
    vector_input.load_state_dict(torch.load(RES_DIR_PATH+f"vector_input-{step}.pth", map_location=torch.device(default_device())),strict = True)

    unn_body = LinearEncoder(
        input_size=unn_obs_dim,
        num_layers=3,
        hidden_size=UNN_HIDDEN_DIM,
    )
    unn_body.load_state_dict(torch.load(RES_DIR_PATH+f"body_endoder-{step}.pth", map_location=torch.device(default_device())),strict = True)

    
    action_size = LATENT_DIM + use_gripper    

    unn_action_head = ActionModel(
        UNN_HIDDEN_DIM,
        action_size,
        tanh_squash=False,
        deterministic=True,
        transfer_settings = None
    )
    unn_action_head.load_state_dict(torch.load(RES_DIR_PATH+f"action_model-{step}.pth", map_location=torch.device(default_device())),strict = True)

    env = make_unity_env(ENV_NAME, worker_id = 40, no_graphics = not args.graphic, time_scale = args.time_scale)

    
    ckpt_perf = test_agent_unn(args.num_test_episode)

    if ckpt_perf > best_perfs :
        best_perfs = ckpt_perf
        best_perfs_step = step

    env.close()
    print("perf : ",ckpt_perf)
    print()

    if ckpt_perf == 1.0 :
        # max perf
        break

print()
print("============== BEST MODEL ================")
print("> performance : ", best_perfs)
print("> ckpt step : ", best_perfs_step)
print("==========================================")
print()