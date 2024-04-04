# ORL-TAMP

Optimistic Rienforcement Learning Task and Motion Planning (ORL-TAMP) is a framework intergrating RL policy into the TAMP pipelines.   

## Video
The method introduction and experiments:

[![Watch the video](https://github.com/Gaoyuan-Liu/Non-prehensile-Augmented-TAMP/blob/main/pics/youtube.png)](https://youtu.be/mlLTIFM01ig)

## Installation 
   
   The current version is tested on Ubuntu 20.04
   
   Dependencies:
   
   * [MoveIt](https://moveit.ros.org/) (ROS Noetic)
   * [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/tree/master)

## Run

   Download the RL policy models: [Retrieve](https://drive.google.com/file/d/1UGd9uoGRnoQsUGBsJQmJ6i1QxkTuBz9B/view?usp=drive_link)

   Run the 'Rearrange' demo:

   ```
   ./run_demo_rearrange.sh
   ```