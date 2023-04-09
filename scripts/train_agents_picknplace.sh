#!/bin/bash
# train each unn agents and rl agents
mlagents-learn config/picknplace/picknplace_braccio.yaml --run-id=picknplace/braccio/unn --env=envs/PicknPlace_Braccio_training --force --unn
mlagents-learn config/picknplace/picknplace_panda.yaml --run-id=picknplace/panda/unn --env=envs/PicknPlace_Panda_training --force --unn
mlagents-learn config/picknplace/picknplace_ur10.yaml --run-id=picknplace/ur10/unn --env=envs/PicknPlace_UR10_training --force --unn
mlagents-learn config/picknplace/picknplace_braccio.yaml --run-id=picknplace/braccio/rl --env=envs/PicknPlace_Braccio_training --force
mlagents-learn config/picknplace/picknplace_panda.yaml --run-id=picknplace/panda/rl --env=envs/PicknPlace_Panda_training --force
mlagents-learn config/picknplace/picknplace_ur10.yaml --run-id=picknplace/ur10/rl --env=envs/PicknPlace_UR10_training --force
