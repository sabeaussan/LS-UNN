# Latent Space Universal Notice Network


This repository is accompanying our paper "Towards Zero-Shot Cross-Agent Transfer Learning via Aligned Latent-Space Task-Solving".

<p align="center">
  <img src="/ressources/LS-UNN_pipeline.png" width=70% />
</p>
<p align="center">
  
</p>



## Install

### Requirements
  - Python >= 3.7

Inside your favorite python virtual environment run the following bash command to pip install all the required python package :
```bash
./scripts/install_commands.sh
```
Then final package "gy_unity" can be installed with 
```bash
cd gym_unity
pip3 install -e ./gym_unity
```
If everything is set correctly, you should be good to go ! 

The test files are written in python and accept a list of arguments to specify what to evaluate. Every arguments for the python scripts are documented. Simply run
```bash
python "name_script".py -h
```
for information about how to chose the command line arguments.

## Reproduce results from saved models

You can download the models used to produce the results presented in the paper using the following link : [?](https://drive.google.com/drive/folders/1oHurrIOmDSvkXpp44jjq2Jgw2JFxygrX?usp=sharing).
The UNN models for all the considered robots and tasks are located inside the "published_results" archive. It contains all the checkpoints obtained during training. To run the tests with these models you need to first extract it at the root ot this repo and rename it to "results". At the same google drive location you will also find the models of the bases used throughout our experiments. To use these, you need to extract the files inside the bases folder. It sould contains the models under the "aligned_00015" subfolder.

### Testing for Zero-shot performance and Fine-tuned performance
It will test the checkpoints of each robots to find the one giving the best transfer results for the considered task. Checkpoints were saved every 50000 training steps. You can specify test frequency with the -f argument. It will test one every f checkpoints (default one every 2 checkpoints).
Run the following python script
```bash
python3 run_test_max_perfs.py -n 1000 -f 2 -m MODE_ID -t TASK_ID
```
and complete the argument with the relevant values. Use "-h" for informations about the accepted arguments.

### Plotting learning curves regular PPO agents vs UNN agents
Test the checkpoints of UNN agents and PPO agents using the reward and performance metric and plot the training curves.
You need to run the following general command :
```bash
python3 run_test_training_curves.py -n 1000 -f 2 -r ROBOT_ID -t TASK_ID
```
Once again, you can look at the documention outputted by "-h" for how to properly complete this command.

### Testing for overfitting
To test for overfitting, run 
```bash
python3 test_overfitting.py -n 1000 -f 2 -r ROBOT_ID -t TASK_ID
```
## Reproduce results from scratch

To regenerate newly trained UNN models, you can either use the pre-trained bases "aligned_00015" or retrain the bases using the provided trajectories. For the latter, execute the following command:

```bash
./scripts/train_bases.sh RUN-ID
```
where RUN-ID is the name you want to give to the generated bases. It will run bases training for the Panda/Braccio pair and then the Braccio/UR10 pair. Not specifying any RUN-ID will create the models under "trained_bases" by default. If you want to use your newly trained bases, you will need to modify the config yaml files (in the config folder at the root) for the 3 robots to specify the bases name to replace the default "aligned_00015" ("base_in_path" and "base_outpath" attributes). Ex :

### Training the models
You can train all the models (PPOs and UNNs) for a given task with the given script :
```bash
./scripts/train_agents_{task_name}.sh
```
where task_name is either : "picknplace", "ball_catcher" or "peg_insertion". By default it wil generate results in "results/{task_name}. It will use PPO implementation of ml-agents for training. Hyper-parameters are specified in yaml files inside the config folder at the root of the repository. Please check https://github.com/Unity-Technologies/ml-agents/tree/develop for additionnal information about the ml-agents config files.

### Fine-tuning the UNN models
Before fine-tuning the generated UNN models, it is necessary to run 
```bash
python3 run_test_max_perfs.py -n 1000 -f 2 -m MODE_ID -t TASK_ID
```
It will automatically populate the folder containing the best checkpoints for further training on the target robot. Once checkpoint files are available, you can run 
```bash
./scripts/fine_tune_unn_agents_{task_name}.sh
```
where task_name is either : "picknplace", "ball_catcher" or "peg_insertion". By default it wil generate results in "results/{task_name}.

Once the all the models are fully trained, you can proceed to the same tests as the previous section. If you don't use the pre-trained bases, you will need to specify the name using the additionnal "--bases_id" arguments for each test commands.

## Tests with visuals
You can also visualize tests or simply try a single transfer performance with either 
```bash
python3 test_zero_shot.py -n 1000 -f 2 -src SOURCE_ROBOT_ID -tgt TARGET_ROBOT_ID -t TASK_ID -ts TIME_SCALE -g
```
Once again, you can look at the documention outputted by "-h" for how to properly complete this command.
