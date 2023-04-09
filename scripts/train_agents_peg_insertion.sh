#!/bin/bash
# train each unn agents and rl agents
mlagents-learn config/peg_insertion/peg_insertion_braccio.yaml --run-id=peg_insertion/braccio/unn --env=envs/PegInsertion_Braccio_training --force --unn
mlagents-learn config/peg_insertion/peg_insertion_panda.yaml --run-id=peg_insertion/panda/unn --env=envs/PegInsertion_Panda_training --force --unn
mlagents-learn config/peg_insertion/peg_insertion_ur10.yaml --run-id=peg_insertion/ur10/unn --env=envs/PegInsertion_UR10_training --force --unn
mlagents-learn config/peg_insertion/peg_insertion_braccio.yaml --run-id=peg_insertion/braccio/rl --env=envs/PegInsertion_Braccio_training --force
mlagents-learn config/peg_insertion/peg_insertion_panda.yaml --run-id=peg_insertion/panda/rl --env=envs/PegInsertion_Panda_training --force
mlagents-learn config/peg_insertion/peg_insertion_ur10.yaml --run-id=peg_insertion/ur10/rl --env=envs/PegInsertion_UR10_training --force
