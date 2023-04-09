#!/bin/bash
# train each unn agents and rl agents
mlagents-learn config/ball_catcher/ball_catcher_braccio.yaml --run-id=ball_catcher/braccio/unn --env=envs/BallCatcher_Braccio_training --force --unn
mlagents-learn config/ball_catcher/ball_catcher_panda.yaml --run-id=ball_catcher/panda/unn --env=envs/BallCatcher_Panda_training --force --unn
mlagents-learn config/ball_catcher/ball_catcher_ur10.yaml --run-id=ball_catcher/ur10/unn --env=envs/BallCatcher_UR10_training --force --unn
mlagents-learn config/ball_catcher/ball_catcher_braccio.yaml --run-id=ball_catcher/braccio/rl --env=envs/BallCatcher_Braccio_training_rl --force
mlagents-learn config/ball_catcher/ball_catcher_panda.yaml --run-id=ball_catcher/panda/rl --env=envs/BallCatcher_Panda_training_rl --force
mlagents-learn config/ball_catcher/ball_catcher_ur10.yaml --run-id=ball_catcher/ur10/rl --env=envs/BallCatcher_UR10_training_rl --force
