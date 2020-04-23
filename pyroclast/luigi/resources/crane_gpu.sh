#!/bin/bash

#module load python/3.7
#pip install --user -r requirements.txt
# Currently doesn't work as intended due to fs issues
# pip install --user -e /home/[group]/[user]/universe/
#export PATH=$PATH:${HOME}/.local/bin

module load anaconda
source activate tf2

export LUIGI_CONFIG_PARSER=toml
export LUIGI_CONFIG_PATH=$1
python3 ${@:2}
