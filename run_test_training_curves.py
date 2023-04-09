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
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from mlagents.torch_utils import default_device

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_test_episode", type=int, default=100, help="number of episode of tests per checkpoints")
parser.add_argument("-f", "--freq_test_models", type=int, default=2, help="skip checkpoints every freq_test_models")
parser.add_argument("-r", "--robot", type=int, default=0, help="Braccio (0), Panda (1), UR10")
parser.add_argument("--bases_id", type=str, default="aligned_00015", help="bases id")
parser.add_argument("-t", "--task", type=int, default=0, help="PicknPlace (0), BallCatcher (1), PegInsertion (2)")
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
                encoding = body(norm_unn_obs)
                latent_action,_,_ = action_head(encoding)
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
    return np.mean(reward_history), np.std(reward_history)



@torch.no_grad()
def test_agent_rl(nb_test_episodes):
    rewards_episode = []
    reward_history = []
    for episode in range(nb_test_episodes):
        obs = env.reset()
        done = False 
        while not done: 
            env.render()
            with torch.no_grad(): 
                robot_obs = obs[:transfer_settings.state_dim]                                                                        
                task_obs = obs[transfer_settings.state_dim:]
                norm_obs = vector_input(torch.FloatTensor(obs).to(default_device()))
                encoding = body(norm_obs)
                action,_,_ = action_head(encoding)
                action = (torch.clamp(action.continuous_tensor, -3, 3) / 3 ).squeeze(0)
            next_obs, rew, done, _ = env.step(action.cpu())

            rewards_episode.append(rew)

            obs = next_obs 

        reward_history.append(np.sum(rewards_episode))
        rewards_episode = []

    return np.mean(reward_history), np.std(reward_history)

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
MODES = ["unn", "rl", "unn_ft_braccio", "unn_ft_panda", "unn_ft_ur10"]

robot_idx = args.robot
task_idx = args.task

if task_idx == 0 :
    use_gripper = True
else :
    use_gripper = False


SOURCE_ROBOT_NAME = ROBOTS[robot_idx]

BASES_PATH_in = f"bases/models/Reacher/bases_dim_6/100k/{args.bases_id}/{SOURCE_ROBOT_NAME}_input.pth"
BASES_PATH_out = f"bases/models/Reacher/bases_dim_6/100k/{args.bases_id}/{SOURCE_ROBOT_NAME}_output.pth"
AGENT_HIDDEN_DIM = 225
PREFIX = "Behavior-*.pt"
SAVE_PERFS_DIR = f"./results/{TASKS[task_idx]}/{SOURCE_ROBOT_NAME.lower()}/perfs/"

base_in = SimpleVariationnalInputBase(
    input_size = STATE_DIM[robot_idx],
    latent_dim = 6,
    hidden_dim = 256
)
base_in.load_state_dict(torch.load(BASES_PATH_in, map_location=torch.device(default_device())), strict = True)

base_out = SimpleVariationnalOutputBase(
    output_size = STATE_DIM[robot_idx],
    latent_dim = 6,
    hidden_dim = 256
)
base_out.load_state_dict(torch.load(BASES_PATH_out, map_location=torch.device(default_device())),strict = True)



# First we test performance, then we test average cumulative reward
perfs_dict = {}

# Outter loop for recording either test performance or avg rwd
for i in range(2):


    # we test average cumulative reward on the training env
    if i == 1:
        print()
        print("-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*")
        print("-*-*-*-* TEST AVERAGE REWARD -*-*-*-*-*")
        print("-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*")
        print()
        ENV_NAME = f"envs/{ENV[task_idx]}_{SOURCE_ROBOT_NAME}_training.x86_64"
        max_rwd = -100000
        std_dict = {}
    # We test performance on the test environment
    else :
        print()
        print("-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*")
        print("-*-*-*-*-* TEST PERFORMANCE -*-*-*-*-*-")
        print("-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*")
        print()
        ENV_NAME = f"envs/{ENV[task_idx]}_{SOURCE_ROBOT_NAME}_testing.x86_64"

    # Middle loop for testing every possible model type (unn, vanilla rl, finetuned models)
    for mode in MODES:

        print(f"========== TESTING {SOURCE_ROBOT_NAME} WITH {mode} MODEL ==========")

        RES_DIR_PATH = f"results/{TASKS[task_idx]}/{SOURCE_ROBOT_NAME.lower()}/{mode}/checkpoints/"

        use_bases = True
        if mode == "rl":
            use_bases = False

        # Retrieve every models from the checkpoint folder for tests
        files_name = glob.glob(RES_DIR_PATH+PREFIX)
        if files_name == []:
            print(f"/!\ /!\ {RES_DIR_PATH} not found /!\ /!\ ")
            continue

        # Glob does not return file names in chronological order so need to sort
        # Use regexp for retrieving step number in the checkpoint name
        steps = bubbleSort([int(re.search('(?<=Behavior-).*?(?=.pt)',step).group(0)) for step in files_name])

        print("* Total number of checkpoint : ",len(steps))


        transfer_settings = TransferSettings(
            use_bases = use_bases,
            base_in_path = BASES_PATH_in,
            base_out_path = BASES_PATH_out, 
            task_obs_dim = TASK_DIM[task_idx],
            state_dim = STATE_DIM[robot_idx],
            latent_dim = 6,
            hidden_units = 256
        )

        if use_bases :
            unn_obs_dim = transfer_settings.latent_dim + transfer_settings.task_obs_dim
        else : 
            unn_obs_dim = transfer_settings.state_dim + transfer_settings.task_obs_dim


        
        perfs_dict[mode+"_"+str(i)] = []
        if i == 1:
            std_dict[mode+"_"+str(i)] = []

        # Inner loop to test chkpt (one every two)
        for step in steps[::freq_test_models] :
            print(f"-- step {step} --")

            # ======= LOAD MODEL =======
            vector_input = VectorInput(input_size = unn_obs_dim, normalize = True)
            vector_input.load_state_dict(torch.load(RES_DIR_PATH+f"vector_input-{step}.pth", map_location=torch.device(default_device())),strict = True)

            body = LinearEncoder(
                input_size=unn_obs_dim,
                num_layers=3,
                hidden_size=AGENT_HIDDEN_DIM,
            )
            body.load_state_dict(torch.load(RES_DIR_PATH+f"body_endoder-{step}.pth", map_location=torch.device(default_device())),strict = True)


            if use_bases :
                action_size = transfer_settings.latent_dim+use_gripper
            else :
                action_size = transfer_settings.state_dim//2+use_gripper

            action_head = ActionModel(
                AGENT_HIDDEN_DIM,
                action_size,
                tanh_squash=False,
                deterministic=True,
                transfer_settings = transfer_settings
            )
            action_head.load_state_dict(torch.load(RES_DIR_PATH+f"action_model-{step}.pth", map_location=torch.device(default_device())),strict = True)
            # ===========================

            # Load unity environment
            env = make_unity_env(ENV_NAME, worker_id = 30, no_graphics = True, time_scale = 20.0)

            # test for num_test_episode
            if use_bases : 
                step_perf, step_std = test_agent_unn(num_test_episode)
            else :
                step_perf, step_std = test_agent_rl(num_test_episode)

            perfs_dict[mode+"_"+str(i)].append(step_perf)

            # If recording avg rwd, we add the std
            # Save max rwd for normalization
            if i ==1:
                std_dict[mode+"_"+str(i)].append(step_std)
                if step_perf > max_rwd :
                    max_rwd = step_perf

            print("perf : ",step_perf)
            print()
            env.close()
        # os makdir
        if not os.path.exists(SAVE_PERFS_DIR):
            os.makedirs(SAVE_PERFS_DIR)
        np.savetxt(SAVE_PERFS_DIR+mode+"_"+str(i)+".txt", np.array(perfs_dict[mode+"_"+str(i)]))


x_perfs = np.arange(0,(len(perfs_dict["unn_0"])+1)*linspace,linspace)

# add a perf of 0 for the 0-step
plt.plot(x_perfs,np.concatenate(([0],perfs_dict["rl_0"]), axis = 0), '--r', label = "Performance PPO")
plt.plot(x_perfs,np.concatenate(([0],perfs_dict["unn_0"]), axis = 0), '--b', label = "Performance UNN")

# start at 1 because we don't have chckpt for 0 step
plt.plot(x_perfs[1:],np.array(perfs_dict["rl_1"])/max_rwd, '-r', label = "Average reward PPO")
plt.fill_between(x_perfs[1:], (np.array(perfs_dict["rl_1"])-np.array(std_dict["rl_1"]))/max_rwd, (np.array(perfs_dict["rl_1"])+np.array(std_dict["rl_1"]))/max_rwd, alpha=0.35, edgecolor='r', facecolor='r')
plt.plot(x_perfs[1:],np.array(perfs_dict["unn_1"])/max_rwd,'-b', label = "Average reward UNN")
plt.fill_between(x_perfs[1:], (np.array(perfs_dict["unn_1"])-np.array(std_dict["unn_1"]))/max_rwd, (np.array(perfs_dict["unn_1"])+np.array(std_dict["unn_1"]))/max_rwd, alpha=0.35, edgecolor='b', facecolor='b')

if "unn_ft_braccio_0" in perfs_dict and "unn_ft_braccio_1" in perfs_dict :
    plt.plot(x_perfs[:len(perfs_dict["unn_ft_braccio_0"])],perfs_dict["unn_ft_braccio_0"], '--g', label = "UNN fine tuned from Braccio")
    plt.plot(x_perfs[:len(perfs_dict["unn_ft_braccio_1"])],np.array(perfs_dict["unn_ft_braccio_1"])/max_rwd,'-g', label = "UNN fine tuned from Braccio")
    plt.fill_between(x_perfs[:len(perfs_dict["unn_ft_braccio_1"])], (np.array(perfs_dict["unn_ft_braccio_1"])-np.array(std_dict["unn_ft_braccio_1"]))/max_rwd, (np.array(perfs_dict["unn_ft_braccio_1"])+np.array(std_dict["unn_ft_braccio_1"]))/max_rwd, alpha=0.35, edgecolor='g', facecolor='g')

if "unn_ft_panda_0" in perfs_dict and "unn_ft_panda_1" in perfs_dict :
    plt.plot(x_perfs[:len(perfs_dict["unn_ft_panda_0"])],perfs_dict["unn_ft_panda_0"], '--k', label = "UNN fine tuned from Panda")
    plt.plot(x_perfs[:len(perfs_dict["unn_ft_panda_1"])],np.array(perfs_dict["unn_ft_panda_1"])/max_rwd,'-k', label = "UNN fine tuned from Panda")
    plt.fill_between(x_perfs[:len(perfs_dict["unn_ft_panda_1"])], (np.array(perfs_dict["unn_ft_panda_1"])-np.array(std_dict["unn_ft_panda_1"]))/max_rwd, (np.array(perfs_dict["unn_ft_panda_1"])+np.array(std_dict["unn_ft_panda_1"]))/max_rwd, alpha=0.35, edgecolor='k', facecolor='k')

if "unn_ft_ur10_0" in perfs_dict and "unn_ft_ur10_1" in perfs_dict :
    plt.plot(x_perfs[:len(perfs_dict["unn_ft_ur10_0"])],perfs_dict["unn_ft_ur10_0"], '--y', label = "UNN fine tuned from UR10")
    plt.plot(x_perfs[:len(perfs_dict["unn_ft_ur10_1"])],np.array(perfs_dict["unn_ft_ur10_1"])/max_rwd,'-y', label = "UNN fine tuned from UR10")
    plt.fill_between(x_perfs[:len(perfs_dict["unn_ft_ur10_1"])], (np.array(perfs_dict["unn_ft_ur10_1"])-np.array(std_dict["unn_ft_ur10_1"]))/max_rwd, (np.array(perfs_dict["unn_ft_ur10_1"])+np.array(std_dict["unn_ft_ur10_1"]))/max_rwd, alpha=0.35, edgecolor='y', facecolor='y')



plt.xlabel('Number of training steps (1e6)', fontsize=20)
plt.ylabel("Performance / Normalized average cumulative rewards", fontsize=20)
plt.title(f"Performance and learning curves for the {SOURCE_ROBOT_NAME} robot on {TASKS[task_idx]} task", fontsize=18)
plt.legend(loc="lower right", fontsize=20)
plt.savefig(SAVE_PERFS_DIR+'perfs.png', bbox_inches='tight', dpi=199)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
