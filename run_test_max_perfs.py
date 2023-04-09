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
import argparse
import shutil
from mlagents.torch_utils import default_device

def pretty_print_perfs(dict_):
   for target_robot, dico in dict_.items():
        print()
        print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        print(f"* > target robot : {target_robot}")
        print("--------------------------------")
        print(f"* > source robots : perfs")
        for source_robot, perf in dico.items():
            print(f"*       {source_robot} : {perf['best_perf']}")
        print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        print()

def copy_best_checkpoints(step, path, target_robot):
    ckpt_model = path+"Behavior-"+str(step)+".pt"
    dst_folder = path+"../Behavior/"+f"checkpoint_{target_robot}.pt"
    print(ckpt_model)
    shutil.copy(ckpt_model, dst_folder)

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_test_episode", type=int, default=100, help="number of episode of tests per checkpoints")
parser.add_argument("-f", "--freq_test_models", type=int, default=2, help="skip checkpoints every freq_test_models")
parser.add_argument("-m", "--mode", type=int, default=0, help="test zero-shot models (0) or fine-tuned models (1)")
parser.add_argument("-t", "--task", type=int, default=0, help="PicknPlace (0), BallCatcher (1), PegInsertion (2)")
parser.add_argument("--bases_id", type=str, default="aligned_00015", help="bases id")
args = parser.parse_args()

num_test_episode = args.num_test_episode
freq_test_models = args.freq_test_models

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
ENV = ["PicknPlace", "BallCatcher", "PegInsertion"]
ROBOTS = ["Braccio", "Panda", "UR10"]
STATE_DIM = [10,14,12]
TASK_DIM = [10, 13, 6]
task_idx = args.task

if task_idx == 0 :
    use_gripper = True
else :
    use_gripper = False


UNN_HIDDEN_DIM = 225
LATENT_DIM = 6

PREFIX = "Behavior-*.pt"

best_perfs_dict = {}


# For each possbile target robot
for target_state_dim, target_robot in zip(STATE_DIM,ROBOTS):

    ############## For every target robot #################
    best_perfs_dict[target_robot] = {}
    ENV_NAME = f"envs/{ENV[task_idx]}_{target_robot}_testing.x86_64"
    BASES_PATH_in = f"bases/models/Reacher/bases_dim_6/100k/{args.bases_id}/{target_robot}_input.pth"
    BASES_PATH_out = f"bases/models/Reacher/bases_dim_6/100k/{args.bases_id}/{target_robot}_output.pth"
    transfer_settings = TransferSettings(
        use_bases = True,
        base_in_path = BASES_PATH_in,
        base_out_path = BASES_PATH_out, 
        task_obs_dim = TASK_DIM[task_idx],
        state_dim = target_state_dim,
        latent_dim = 6,
        hidden_units = 256
    )
    base_in = SimpleVariationnalInputBase(
        input_size = transfer_settings.state_dim,
        latent_dim = transfer_settings.latent_dim,
        hidden_dim = transfer_settings.hidden_units
    )
    base_in.load_state_dict(torch.load(transfer_settings.base_in_path, map_location=torch.device(default_device())), strict = True)

    base_out = SimpleVariationnalOutputBase(
            output_size = transfer_settings.state_dim,
            latent_dim = transfer_settings.latent_dim,
            hidden_dim = transfer_settings.hidden_units
        )
    base_out.load_state_dict(torch.load(transfer_settings.base_out_path, map_location=torch.device(default_device())),strict = True)

    for src_robot in ROBOTS:


        ############## For every source UNN models #################
        if args.mode == 0:
            print(f"========== TESTING {target_robot} WITH {src_robot} UNN ==========")
        else :
            print(f"========== TESTING {target_robot} WITH {src_robot} UNN FINE TUNED ==========")


        best_perfs_dict[target_robot][src_robot] = {}
        best_perfs_dict[target_robot][src_robot]["best_perf"] = 0

        if args.mode == 0 :
            RES_DIR_PATH = f"results/{TASKS[task_idx]}/{src_robot.lower()}/unn/checkpoints/"
        else :
            RES_DIR_PATH = f"results/{TASKS[task_idx]}/{target_robot.lower()}/unn_ft_{src_robot.lower()}/checkpoints/"

        # Retrieve every models from the checkpoint folder for tests
        files_name = glob.glob(RES_DIR_PATH+PREFIX)
        if files_name == []:
            print(f"/!\ /!\ {RES_DIR_PATH} not found /!\ /!\ ")
            continue

        # Glob does not return file names in chronological order so need to sort
        # Use regexp for retrieving step number in the checkpoint name
        steps = bubbleSort([int(re.search('(?<=Behavior-).*?(?=.pt)',step).group(0)) for step in files_name])


        print("* Total number of checkpoints : ",len(steps))
        
        

        unn_obs_dim = transfer_settings.latent_dim + transfer_settings.task_obs_dim
        

        for step in steps[::freq_test_models] :

            ############## For every chkpt models #################
            print(f"-- step {step} --")

            # ======= LOAD MODEL =======
            vector_input = VectorInput(input_size = unn_obs_dim, normalize = True)
            vector_input.load_state_dict(torch.load(RES_DIR_PATH+f"vector_input-{step}.pth", map_location=torch.device(default_device())),strict = True)

            unn_body = LinearEncoder(
                input_size=unn_obs_dim,
                num_layers=3,
                hidden_size=UNN_HIDDEN_DIM,
            )
            unn_body.load_state_dict(torch.load(RES_DIR_PATH+f"body_endoder-{step}.pth", map_location=torch.device(default_device())),strict = True)

            
            action_size = transfer_settings.latent_dim+use_gripper

            unn_action_head = ActionModel(
                UNN_HIDDEN_DIM,
                action_size,
                tanh_squash=False,
                deterministic=True,
                transfer_settings = transfer_settings
            )
            unn_action_head.load_state_dict(torch.load(RES_DIR_PATH+f"action_model-{step}.pth", map_location=torch.device(default_device())),strict = True)
            # ===========================

            
            # Load unity environment
            env = make_unity_env(ENV_NAME, worker_id = 40, no_graphics = True, time_scale = 20.0)

            # test for num_test_episode
            ckpt_perf = test_agent_unn(num_test_episode)

            if ckpt_perf >= best_perfs_dict[target_robot][src_robot]["best_perf"] :
                best_perfs_dict[target_robot][src_robot]["best_perf"] = ckpt_perf
                best_perfs_dict[target_robot][src_robot]["step_perf"] = step

            env.close()
            print("perf : ",ckpt_perf)
            print()

            if ckpt_perf == 1.0 :
                # max perf
                break

        if args.mode == 0 :
            # copy best ckpt model in the Behavior folder for further fine tuning
            copy_best_checkpoints(best_perfs_dict[target_robot][src_robot]["step_perf"],RES_DIR_PATH, target_robot)

            
if args.mode == 0 :
    print()
    print("=====================================================")
    print("=============== ZERO SHOT PERFORMANCE ===============")
    print("=====================================================")
    print()
else :
    print()
    print("======================================================")
    print("=============== FINE TUNED PERFORMANCE ===============")
    print("======================================================")
    print()
pretty_print_perfs(best_perfs_dict)
print(best_perfs_dict)