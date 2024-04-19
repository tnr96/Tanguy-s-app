cd /Users/tmr96/Documents/Automatic
conda init
conda activate moose
/Users/tmr96/projects/my_files/solid_mechanics -i to_run/sigma_11.i & \
/Users/tmr96/projects/my_files/solid_mechanics -i to_run/sigma_21.i & \
/Users/tmr96/projects/my_files/solid_mechanics -i to_run/sigma_22.i & \
/Users/tmr96/projects/my_files/solid_mechanics -i to_run/epsilon_11.i & \
/Users/tmr96/projects/my_files/solid_mechanics -i to_run/epsilon_21.i & \
/Users/tmr96/projects/my_files/solid_mechanics -i to_run/epsilon_22.i
