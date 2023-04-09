#!/bin/bash
# fine tune unn agents on braccio
mlagents-learn config/picknplace/picknplace_braccio.yaml --run-id=picknplace/braccio/unn_ft_panda --env=envs/PicknPlace_Braccio_training --force --unn --initialize-from=picknplace/panda/unn --ckpt-name=checkpoint_Braccio.pt
mlagents-learn config/picknplace/picknplace_braccio.yaml --run-id=picknplace/braccio/unn_ft_ur10 --env=envs/PicknPlace_Braccio_training --force --unn --initialize-from=picknplace/ur10/unn --ckpt-name=checkpoint_Braccio.pt
# fine tune unn agents on panda
mlagents-learn config/picknplace/picknplace_panda.yaml --run-id=picknplace/panda/unn_ft_braccio --env=envs/PicknPlace_Panda_training --force --unn --initialize-from=picknplace/braccio/unn --ckpt-name=checkpoint_Panda.pt
mlagents-learn config/picknplace/picknplace_panda.yaml --run-id=picknplace/panda/unn_ft_ur10 --env=envs/PicknPlace_Panda_training --force --unn --initialize-from=picknplace/ur10/unn --ckpt-name=checkpoint_Panda.pt
# fine tune unn agents on ur10
mlagents-learn config/picknplace/picknplace_ur10.yaml --run-id=picknplace/ur10/unn_ft_braccio --env=envs/PicknPlace_UR10_training --force --unn --initialize-from=picknplace/braccio/unn --ckpt-name=checkpoint_UR10.pt
mlagents-learn config/picknplace/picknplace_ur10.yaml --run-id=picknplace/ur10/unn_ft_ur10 --env=envs/PicknPlace_UR10_training --force --unn --initialize-from=picknplace/panda/unn --ckpt-name=checkpoint_UR10.pt
