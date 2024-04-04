# ORL-TAMP

Optimistic Rienforcement Learning Task and Motion Planning (ORL-TAMP) is a framework intergrating RL policy into the TAMP pipelines.   

<img src="pics/structure.png" height="200">
<!-- <img src="images/continuous_tamp.png" height="100">&emsp;<img src="images/motion.png" height="100"> -->

## Video
The method introduction and experiments:

[![Watch the video](https://github.com/Gaoyuan-Liu/Non-prehensile-Augmented-TAMP/blob/main/pics/youtube.png)](https://youtu.be/mlLTIFM01ig)

## Installation 
   
   The current version is tested on Ubuntu 20.04
   
   Dependencies:
   
   * [MoveIt](https://moveit.ros.org/) (ROS Noetic)

   * [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/tree/master)

   We are currently trying to remove the dependency of MoveIt due to its inflexibility and ROS specificity.

   Build PDDL solver:
   ```
   orl_tamp$ ./downward/build.py
   ```

   Compile IK solver:
   ```
   orl_tamp$ cd utils/pybullet_tools/ikfast/franka_panda/
   franka_panda$ python setup.py
   ```



## Run
   
   1. Download the RL policy models: [Retrieve](https://drive.google.com/file/d/1UGd9uoGRnoQsUGBsJQmJ6i1QxkTuBz9B/view?usp=drive_link) and [EdgePush](https://drive.google.com/file/d/1tdIOrf1GFvP4PCmKRepSF5rJe3CE-rUU/view?usp=drive_link), and save policies in the 'policies' folder. 

   2. Run MoveIt (following the [tutorial](https://ros-planning.github.io/moveit_tutorials/))

   3. Run demos:
      * Retrieve: `orl_tamp$ ./run_demo.sh retrieve`
      * EdgePush: `orl_tamp$ ./run_demo.sh edgepush`
      * Rearange: `orl_tamp$ ./run_demo.sh rearrange`

## Train 
   
   This section we give instructions about how to train your own skills. 

   We recomand using [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) to standardized the policy trainning. 