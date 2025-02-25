# ROS2 Foxy Obstacle Avoidance with RL Algorithms

This project implements obstacle avoidance for a robotic system using three reinforcement learning (RL) algorithms within a ROS2 Foxy environment. The simulation is conducted in Gazebo, utilizing Velodyne LiDAR for obstacle detection.

## Features
- **Reinforcement Learning Algorithms:**
  - **TD3** (Twin Delayed Deep Deterministic Policy Gradient)
  - **TD3-CLN** (TD3 with Curriculum Learning and Noise)
  - **SAC** (Soft Actor-Critic)
- **Simulation Environment:** Gazebo
- **Sensor Integration:** Velodyne LiDAR (Velodyne ROS package)
- **Obstacle Avoidance:** The trained policy enables the robot to navigate around obstacles efficiently.

## Installation

### Prerequisites
Ensure you have the following installed:
-Ubuntu 20.04
- ROS2 Foxy
- Gazebo (compatible version with ROS2 Foxy)
- Velodyne ROS package (you can try this:https://github.com/ros-drivers/velodyne)
- Python 3 and required dependencies
- `gym`, `stable-baselines3`, and `torch` for reinforcement learning
- tensorboard 

### Setup
Clone the repository and install dependencies:
```bash
cd ~/ros2_ws/src
git clone https://github.com/sHaises/robot_navigation_with_td3_sac_itd3cln.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

## Usage

### 1. Launch Gazebo with the Environment
```bash
ros2 launch <your_package> gazebo.launch.py
```



### . Run the RL-Based Navigation
To start the obstacle avoidance using a specific algorithm for training and testing:

#### TD3:
```bash
ros2 launch td3 training_simulation.launch.py 
```

#### TD3-CLN:
```bash
ros2 launch td3 train_it3.launch.py ^C
```

#### SAC:
```bash
ros2 launch td3 SAC.launch.py

```



## Simulation Visualization
- Use Gazebo to visualize the robot navigating obstacles.
- ROS2 Rviz can be used to view LiDAR data:


## Contributing
Feel free to submit issues or pull requests for improvements.
