#!/bin/bash
# fine tune unn agents on braccio
mlagents-learn config/peg_insertion/peg_insertion_braccio.yaml --run-id=peg_insertion/braccio/unn_ft_panda --env=envs/PegInsertion_Braccio_training --force --unn --initialize-from=peg_insertion/panda/unn --ckpt-name=checkpoint_Braccio.pt
mlagents-learn config/peg_insertion/peg_insertion_braccio.yaml --run-id=peg_insertion/braccio/unn_ft_ur10 --env=envs/PegInsertion_Braccio_training --force --unn --initialize-from=peg_insertion/ur10/unn --ckpt-name=checkpoint_Braccio.pt
# fine tune unn agents on panda
mlagents-learn config/peg_insertion/peg_insertion_panda.yaml --run-id=peg_insertion/panda/unn_ft_braccio --env=envs/PegInsertion_Panda_training --force --unn --initialize-from=peg_insertion/braccio/unn --ckpt-name=checkpoint_Panda.pt
mlagents-learn config/peg_insertion/peg_insertion_panda.yaml --run-id=peg_insertion/panda/unn_ft_ur10 --env=envs/PegInsertion_Panda_training --force --unn --initialize-from=peg_insertion/ur10/unn --ckpt-name=checkpoint_Panda.pt
# fine tune unn agents on ur10
mlagents-learn config/peg_insertion/peg_insertion_ur10.yaml --run-id=peg_insertion/ur10/unn_ft_braccio --env=envs/PegInsertion_UR10_training --force --unn --initialize-from=peg_insertion/braccio/unn --ckpt-name=checkpoint_UR10.pt
mlagents-learn config/peg_insertion/peg_insertion_ur10.yaml --run-id=peg_insertion/ur10/unn_ft_ur10 --env=envs/PegInsertion_UR10_training --force --unn --initialize-from=peg_insertion/panda/unn --ckpt-name=checkpoint_UR10.pt
