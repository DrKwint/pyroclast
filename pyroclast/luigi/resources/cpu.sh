#!/bin/bash

# Set the path to include the standard folders (in case they are not already there)
export PATH="$PATH:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/util/opt/bin"

if [ -e /util/opt/lmod/lmod/init/profile ]; then
    . /util/opt/lmod/lmod/init/profile
    export -f module
    GROUPMODPATH=`echo ${HOME} | sed "s/\/${USER}$/\/shared\/modulefiles/g"`
    if [ -d $GROUPMODPATH ]
    then
        MODULEPATH=`/util/opt/lmod/lmod/libexec/addto  --append MODULEPATH $GROUPMODPATH`
        export MODULEPATH
    fi
    MODULEPATH=`/util/opt/lmod/lmod/libexec/addto  --append MODULEPATH /util/opt/hcc-modules/Common`
    export LMOD_AVAIL_STYLE="system:<en_grouped>"
    module load compiler/gcc/8.2
    module load boost/1.69
    module list
fi

module load python/3.7
pip install --user -r requirements.txt
# Currently doesn't work as intended due to fs issues
# pip install --user -e /home/[group]/[user]/universe/
export PATH=$PATH:/home/[group]/[user]/.local/bin

export LUIGI_CONFIG_PARSER=toml
export LUIGI_CONFIG_PATH=$1
python3 ./universe/sched.py ${@:2}
