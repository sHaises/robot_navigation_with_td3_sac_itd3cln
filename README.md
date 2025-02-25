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
- ROS2 Foxy
- Gazebo (compatible version with ROS2 Foxy)
- Velodyne ROS package (`velodyne_driver`, `velodyne_laserscan`, etc.)
- Python 3 and required dependencies
- `gym`, `stable-baselines3`, and `torch` for reinforcement learning

### Setup
Clone the repository and install dependencies:
```bash
cd ~/ros2_ws/src
git clone <repository_url>
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

### 2. Start Velodyne Sensor
```bash
ros2 launch velodyne_driver velodyne_driver_node.launch.py
```

### 3. Run the RL-Based Navigation
To start the obstacle avoidance using a specific algorithm:

#### TD3:
```bash
ros2 run <your_package> obstacle_avoidance_td3
```

#### TD3-CLN:
```bash
ros2 run <your_package> obstacle_avoidance_td3_cln
```

#### SAC:
```bash
ros2 run <your_package> obstacle_avoidance_sac
```

## Training the Model
To train a new model using a specific algorithm, run:
```bash
python3 train.py --algorithm TD3
python3 train.py --algorithm TD3-CLN
python3 train.py --algorithm SAC
```

## Simulation Visualization
- Use Gazebo to visualize the robot navigating obstacles.
- ROS2 Rviz can be used to view LiDAR data:
```bash
ros2 launch velodyne_laserscan velodyne_laserscan_node.launch.py
```

## Contributing
Feel free to submit issues or pull requests for improvements.
