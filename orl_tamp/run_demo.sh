#!/bin/sh

name=$1

# if [ $name=="edgepush" ]; then
#     echo "Running rearrange demo"
#     cd opt_tamp_combined/solver/
#     python run_reuse.py $name
#     exit 0
# fi

cd opt_tamp_combined/solver/
python run_reuse.py $name
