#!/bin/bash
pip3 install torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
pip3 install protobuf==3.19.6
pip3 install numpy==1.22.3
pip3 install matplotlib==3.7.1
pip3 install -U scikit-learn==1.2.2
