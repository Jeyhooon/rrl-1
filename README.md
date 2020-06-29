### Code for Robot Reinforcement Learning Praktikum in SS2020.

Clone the repository and then install requirements: `pip install -e ".[dev]"`

The following environment variables need to be set:

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ruben/.mujoco/mujoco200/bin`

`export MJLIB_PATH=/home/ruben/.mujoco/mujoco200/bin/libmujoco200.so`

Furthermore, SimulationFramework has to be installed locally according to the directions at: 
https://git.informatik.kit.edu/i53/SimulationFramework

To run the code execute the following command:

`python main.py --logdir {your_log_dir_here} --task dmc_reach_site_vision`