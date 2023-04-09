from mlagents.trainers.torch_modules.encoders import (
    VectorInput,
    SimpleVariationnalInputBase,
)
from mlagents.trainers.torch_modules.layers import LinearEncoder
from mlagents.trainers.torch_modules.decoders import SimpleVariationnalOutputBase
from mlagents.trainers.torch_modules.action_model import ActionModel
from mlagents.trainers.settings import TransferSettings
from unity_env_gym import make_unity_env
import torch
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
import os
from mlagents.torch_utils import default_device

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_test_episode", type=int, default=100, help="number of episode of tests per checkpoints")
parser.add_argument("-f", "--freq_test_models", type=int, default=2, help="skip checkpoints every freq_test_models")
parser.add_argument("-r", "--robot", type=int, default=0, help="Braccio (0), Panda (1), UR10 (2)")
parser.add_argument("-t", "--task", type=int, default=0, help="PicknPlace (0), BallCatcher (1), PegInsertion (2)")
parser.add_argument("--bases_id", type=str, default="aligned_00015", help="bases id")
args = parser.parse_args()

rcParams['figure.figsize'] = 20, 15
num_test_episode = args.num_test_episode
freq_test_models = args.freq_test_models
freq_chkpt = 0.05 #1e6 steps
linspace = freq_test_models * freq_chkpt

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
                robot_obs = obs[:transfer_settings.state_dim]                                                                        
                task_obs = obs[transfer_settings.state_dim:]
                latent_obs = base_in.get_mu(torch.FloatTensor(robot_obs).to(default_device()))
                unn_obs = torch.cat((latent_obs,torch.FloatTensor(task_obs).to(default_device())),dim = 0)
                norm_unn_obs = vector_input(unn_obs)
                encoding = unn_body(norm_unn_obs)
                latent_action,_,_ = unn_action_head(encoding)
                latent_action = (torch.clamp(latent_action.continuous_tensor, -3, 3) / 3 )*1.25
                robot_act = base_out.get_joints_velocity(latent_action[:transfer_settings.latent_dim].unsqueeze(0)).cpu()
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
LATENT_DIM = 6

robot_idx = args.robot
task_idx = args.task

SOURCE_ROBOT_NAME = ROBOTS[robot_idx]


UNN_HIDDEN_DIM = 225
PREFIX = "Behavior-*.pt"

label = {}
for name in ROBOTS:
    label[name] = name
    if name == SOURCE_ROBOT_NAME :
        label[name] += " (training)"
    else :
        label[name] += " (testing)"

if task_idx == 0 :
    use_gripper = True
else :
    use_gripper = False

# First with test reward, then with train reward
perfs_dict = {}
RES_DIR_PATH = f"results/{TASKS[task_idx]}/{SOURCE_ROBOT_NAME.lower()}/unn/checkpoints/"

for i, target_robot in enumerate(ROBOTS):
    print(f"========== TESTING overfitting from {SOURCE_ROBOT_NAME} UNN WITH {target_robot} robot ==========")

    BASES_PATH_in = f"bases/models/Reacher/bases_dim_6/100k/{args.bases_id}/{target_robot}_input.pth"
    BASES_PATH_out = f"bases/models/Reacher/bases_dim_6/100k/{args.bases_id}/{target_robot}_output.pth"

    base_in = SimpleVariationnalInputBase(
        input_size = STATE_DIM[i],
        latent_dim = LATENT_DIM,
        hidden_dim = 256
    )
    base_in.load_state_dict(torch.load(BASES_PATH_in, map_location=torch.device(default_device())), strict = True)

    base_out = SimpleVariationnalOutputBase(
        output_size = STATE_DIM[i],
        latent_dim = LATENT_DIM,
        hidden_dim = 256
    )
    base_out.load_state_dict(torch.load(BASES_PATH_out, map_location=torch.device(default_device())),strict = True)

    ENV_NAME = f"envs/{ENV[task_idx]}_{target_robot}_testing.x86_64"                                                 # <------- A changer pour les autres taches
    

    use_bases = True
    
    files_name = glob.glob(RES_DIR_PATH+PREFIX)
    if files_name == []:
        print(f"/_!_\ /_!_\ {RES_DIR_PATH} not found /_!_\ /_!_\ ")
        continue

    steps = bubbleSort([int(re.search('(?<=Behavior-).*?(?=.pt)',step).group(0)) for step in files_name])

    print("* Total number of steps : ",len(steps))


    transfer_settings = TransferSettings(
        use_bases = use_bases,
        base_in_path = BASES_PATH_in,
        base_out_path = BASES_PATH_out, 
        task_obs_dim = TASK_DIM[task_idx],                                                                                      # <------- A changer pour les autres taches
        state_dim = STATE_DIM[i],
        latent_dim = 6,
        hidden_units = 256
    )

    unn_obs_dim = transfer_settings.latent_dim + transfer_settings.task_obs_dim



    # Inner test loop
    # Test perf for each chkpt
    perfs_dict[target_robot] = []

    for step in steps[::freq_test_models] :
        print(f"-- step {step} --")
        vector_input = VectorInput(input_size = unn_obs_dim, normalize = True)
        vector_input.load_state_dict(torch.load(RES_DIR_PATH+f"vector_input-{step}.pth", map_location=torch.device(default_device())),strict = True)

        unn_body = LinearEncoder(
            input_size=unn_obs_dim,
            num_layers=3,
            hidden_size=UNN_HIDDEN_DIM,
        )
        unn_body.load_state_dict(torch.load(RES_DIR_PATH+f"body_endoder-{step}.pth", map_location=torch.device(default_device())),strict = True)

        action_size = transfer_settings.latent_dim+use_gripper                                                      # <------- A changer pour les autres taches

        unn_action_head = ActionModel(
            UNN_HIDDEN_DIM,
            action_size,
            tanh_squash=False,
            deterministic=True,
            transfer_settings = None
        )
        unn_action_head.load_state_dict(torch.load(RES_DIR_PATH+f"action_model-{step}.pth", map_location=torch.device(default_device())),strict = True)


        env = make_unity_env(ENV_NAME, worker_id = 48, no_graphics = True, time_scale = 20.0)

        step_perf = test_agent_unn(num_test_episode)

        perfs_dict[target_robot].append(step_perf)


        print("perf : ",step_perf)
        print()
        env.close()



x_perfs = np.arange(0,(len(perfs_dict["Braccio"])+1)*linspace,linspace)
# add a perf of 0 for the 0-step
plt.plot(x_perfs,np.concatenate(([0],perfs_dict["Braccio"]), axis = 0), '--r', label = label["Braccio"])
plt.plot(x_perfs,np.concatenate(([0],perfs_dict["Panda"]), axis = 0), '--b', label = label["Panda"])
plt.plot(x_perfs,np.concatenate(([0],perfs_dict["UR10"]), axis = 0), '--g', label = label["UR10"])

plt.xlabel('Number of training steps (1e6)', fontsize=20)
plt.ylabel("Performance", fontsize=20)
plt.legend(loc="lower right")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
