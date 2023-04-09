#!/bin/bash
# fine tune unn agents on braccio
mlagents-learn config/ball_catcher/ball_catcher_braccio.yaml --run-id=ball_catcher/braccio/unn_ft_panda --env=envs/BallCatcher_Braccio_training --force --unn --initialize-from=ball_catcher/panda/unn --ckpt-name=checkpoint_Braccio.pt
mlagents-learn config/picknplace/ball_catcher_braccio.yaml --run-id=ball_catcher/braccio/unn_ft_ur10 --env=envs/BallCatcher_Braccio_training --force --unn --initialize-from=ball_catcher/ur10/unn --ckpt-name=checkpoint_Braccio.pt
# fine tune unn agents on panda
mlagents-learn config/picknplace/ball_catcher_panda.yaml --run-id=ball_catcher/panda/unn_ft_braccio --env=envs/BallCatcher_Panda_training --force --unn --initialize-from=ball_catcher/braccio/unn --ckpt-name=checkpoint_Panda.pt
mlagents-learn config/picknplace/ball_catcher_panda.yaml --run-id=ball_catcher/panda/unn_ft_ur10 --env=envs/BallCatcher_Panda_training --force --unn --initialize-from=ball_catcher/ur10/unn --ckpt-name=checkpoint_Panda.pt
# fine tune unn agents on ur10
mlagents-learn config/picknplace/ball_catcher_ur10.yaml --run-id=ball_catcher/ur10/unn_ft_braccio --env=envs/BallCatcher_UR10_training --force --unn --initialize-from=ball_catcher/braccio/unn --ckpt-name=checkpoint_UR10.pt
mlagents-learn config/picknplace/ball_catcher_ur10.yaml --run-id=ball_catcher/ur10/unn_ft_ur10 --env=envs/BallCatcher_UR10_training --force --unn --initialize-from=ball_catcher/panda/unn --ckpt-name=checkpoint_UR10.pt
