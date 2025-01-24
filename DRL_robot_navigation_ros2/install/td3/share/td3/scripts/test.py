#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import threading

import math
import random
from torch.optim import Adam
import point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.2

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu

last_odom = None
environment_dim = 20
velodyne_data = np.ones(environment_dim) * 10

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        """
        Initializează ReplayBuffer cu o dimensiune maximă dată.

        Args:
        - max_size (int): Dimensiunea maximă a bufferului.
        """
        self.storage = []    # Lista pentru stocarea experiențelor
        self.max_size = max_size   # Dimensiunea maximă a bufferului
        self.ptr = 0         # Pointerul pentru următoarea locație disponibilă în buffer

    def add(self, state, action, reward, next_state, done):
        """
        Adaugă o experiență nouă în buffer.

        Args:
        - state: Starea curentă.\
        - action: Acțiunea luată în starea curentă.
        - reward: Răsplata obținută după efectuarea acțiunii.
        - next_state: Starea următoare.
        - done: Flag care indică dacă starea următoare este finală (True) sau nu (False).
        """
        data = (state, action, reward, next_state, done)   # Creează o tuplă cu datele experienței noi

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data   # Dacă bufferul este plin, înlocuiește o experiență veche cu una nouă
            self.ptr = (self.ptr + 1) % self.max_size  # Actualizează pointerul circular
        else:
            self.storage.append(data)   # Altfel, adaugă experiența la sfârșitul listei

    def sample(self, batch_size):
        """
        Returnează un eșantion de dimensiune batch_size din buffer.

        Args:
        - batch_size (int): Dimensiunea eșantionului (numărul de experiențe de extras).

        Returns:
        - state: Array cu stările eșantionate.
        - action: Array cu acțiunile eșantionate.
        - reward: Array cu recompensele eșantionate.
        - next_state: Array cu stările următoare eșantionate.
        - done: Array cu flagurile 'done' eșantionate.
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)   # Generează indici aleatorii pentru eșantionare
        state, action, reward, next_state, done = [], [], [], [], []

        for i in ind:
            s, a, r, s_, d = self.storage[i]   # Extrage o experiență din buffer la indicele i
            state.append(np.array(s, copy=False))       # Adaugă starea la listă
            action.append(np.array(a, copy=False))     # Adaugă acțiunea la listă
            reward.append(np.array(r, copy=False))     # Adaugă răsplata la listă
            next_state.append(np.array(s_, copy=False))   # Adaugă starea următoare la listă
            done.append(np.array(d, copy=False))       # Adaugă flagul 'done' la listă

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Definim straturile rețelei neuronale pentru actor
        self.layer_1 = nn.Linear(state_dim, 256)  # Stratul 1: state_dim -> 256
        self.layer_2 = nn.Linear(256, 256)       # Stratul 2: 256 -> 256
        self.mean_layer = nn.Linear(256, action_dim)   # Stratul pentru mean: 256 -> action_dim
        self.log_std_layer = nn.Linear(256, action_dim)  # Stratul pentru log_std: 256 -> action_dim
        self.max_action = max_action  # Valoarea maximă pe care o poate lua o acțiune

    def forward(self, state):
        # Implementăm funcția forward a rețelei neuronale
        x = F.relu(self.layer_1(state))   # Aplicăm funcția de activare ReLU pe primul strat
        x = F.relu(self.layer_2(x))       # Aplicăm funcția de activare ReLU pe al doilea strat
        mean = self.mean_layer(x)         # Obținem mean-ul acțiunii
        log_std = self.log_std_layer(x)   # Obținem log_std-ul acțiunii
        log_std = torch.clamp(log_std, min=-20, max=2)  # Asigurăm că log_std-ul este într-un interval rezonabil
        return mean, log_std

    def get_action(self, state, deterministic=False):
        # Funcție pentru a obține acțiunea, poate fi deterministă sau nu
        mean, log_std = self.forward(state)  # Obținem mean-ul și log_std-ul acțiunii
        std = log_std.exp()                 # Obținem deviația standard pe baza log_std-ului
        normal = Normal(mean, std)          # Distribuția normală cu mean și std
        if deterministic:
            action = mean                  # Acțiunea deterministă este mean-ul
        else:
            action = normal.rsample()      # Acțiunea stochastică este o mostră din distribuția normală
        action = torch.tanh(action) * self.max_action  # Aplicăm funcția tanh pentru a aduce acțiunea în intervalul [-max_action, max_action]
        return action

    def evaluate(self, state, epsilon=1e-6):
        # Funcție pentru evaluarea acțiunii, folosită în antrenare
        mean, log_std = self.forward(state)   # Obținem mean-ul și log_std-ul acțiunii
        std = log_std.exp()                  # Obținem deviația standard pe baza log_std-ului
        normal = Normal(mean, std)           # Distribuția normală cu mean și std
        z = normal.rsample()                # Obținem o mostră din distribuția normală
        action = torch.tanh(z) * self.max_action  # Aplicăm funcția tanh pentru a aduce acțiunea în intervalul [-max_action, max_action]

        # Calculăm log-probabilitatea acțiunii generate
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)  # Sumăm log-probabilitățile
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Definim straturile rețelei neuronale pentru critic
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)  # Stratul 1: state_dim + action_dim -> 256
        self.layer_2 = nn.Linear(256, 256)                     # Stratul 2: 256 -> 256
        self.layer_3 = nn.Linear(256, 1)                       # Stratul 3 (ieșire): 256 -> 1

        self.layer_4 = nn.Linear(state_dim + action_dim, 256)  # Al doilea critic (pentru Q2)
        self.layer_5 = nn.Linear(256, 256)
        self.layer_6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)  # Concatenăm starea și acțiunea
        q1 = F.relu(self.layer_1(sa))      # Aplicăm funcția de activare ReLU pe primul strat
        q1 = F.relu(self.layer_2(q1))      # Aplicăm funcția de activare ReLU pe al doilea strat
        q1 = self.layer_3(q1)              # Obținem ieșirea Q1

        q2 = F.relu(self.layer_4(sa))      # Aplicăm funcția de activare ReLU pe primul strat al criticului 2
        q2 = F.relu(self.layer_5(q2))      # Aplicăm funcția de activare ReLU pe al doilea strat al criticului 2
        q2 = self.layer_6(q2)              # Obținem ieșirea Q2
        return q1, q2

class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        # Definim straturile rețelei neuronale pentru value
        self.layer_1 = nn.Linear(state_dim, 256)  # Stratul 1: state_dim -> 256
        self.layer_2 = nn.Linear(256, 256)        # Stratul 2: 256 -> 256
        self.layer_3 = nn.Linear(256, 1)          # Stratul 3 (ieșire): 256 -> 1

    def forward(self, state):
        v = F.relu(self.layer_1(state))   # Aplicăm funcția de activare ReLU pe primul strat
        v = F.relu(self.layer_2(v))       # Aplicăm funcția de activare ReLU pe al doilea strat
        v = self.layer_3(v)               # Obținem ieșirea value
        return v

class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-4)  # Optimizator pentru actor

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # Setăm parametrii critic-target identici cu cei ai criticului
        self.critic_optimizer = Adam(self.critic.parameters(), lr=3e-4)  # Optimizator pentru critic

        # Initialize the Value network
        self.value = Value(state_dim).to(device)
        self.value_target = Value(state_dim).to(device)
        self.value_target.load_state_dict(self.value.state_dict())  # Setăm parametrii value-target identici cu cei ai value
        self.value_optimizer = Adam(self.value.parameters(), lr=3e-4)  # Optimizator pentru value

        self.max_action = max_action  # Valoarea maximă pe care o poate avea o acțiune
        self.writer = SummaryWriter(log_dir="./DRL_robot_navigation_ros2/src/td3/scripts/runs")  # Obiect pentru scrierea în TensorBoard
        self.iter_count = 0  # Contor pentru numărul de iterații de antrenament

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  # Transformăm starea într-un tensor și îl mutăm pe GPU (dacă este specificat)
        return self.actor.get_action(state, deterministic).cpu().data.numpy().flatten()  # Obținem acțiunea, o mutăm pe CPU și o convertim într-un șir de numpy

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, alpha=0.2):
        av_Q = 0  # Variabilă pentru media valorilor Q
        max_Q = -float('inf')  # Variabilă pentru valoarea maximă a Q
        av_loss = 0  # Variabilă pentru media pierderilor

        for it in range(iterations):
            # Samplează un lot din replay buffer
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).to(device)

            # Evaluează acțiunea următoare și log-probabilitatea asociată folosind actorul
            next_action, next_log_prob = self.actor.evaluate(next_state)

            # Calculează Q-urile țintă folosind criticul-target pentru perechea următoare stare-acțiune
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_v = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_v = reward + (1 - done) * discount * target_v.detach()

            # Calculează Q-urile curente folosind criticul pentru perechea curentă stare-acțiune
            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_v) + F.mse_loss(current_q2, target_v)

            # Efectuează descendentul gradientului pentru critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Calculează valoarea previzionată folosind rețeaua Value
            predicted_v = self.value(state)
            target_v_value = torch.min(current_q1, current_q2) - alpha * next_log_prob.detach()
            value_loss = F.mse_loss(predicted_v, target_v_value.detach())

            # Efectuează descendentul gradientului pentru rețeaua Value
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Evaluează acțiunea curentă și log-probabilitatea asociată folosind actorul
            action, log_prob = self.actor.evaluate(state)

            # Calculează pierderea actorului
            q1, q2 = self.critic(state, action)
            actor_loss = (alpha * log_prob - torch.min(q1, q2)).mean()

            # Efectuează descendentul gradientului pentru actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Actualizează parametrii rețelelor target folosind actualizarea moale (soft update)
            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Scrie pierderea criticului, a value-ului și a actorului în TensorBoard
            env.get_logger().info(f"writing new results for a tensorboard")
            env.get_logger().info(f"loss, Av.Q, Max.Q, iterations : {av_loss / iterations}, {av_Q / iterations}, {max_Q}, {self.iter_count}")
            self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
            self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
            self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

            self.iter_count += 1

    def save(self, filename, directory):
        # Salvăm parametrii actorului, criticului și value-ului în fișiere separate
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.value.state_dict(), f"{directory}/{filename}_value.pth")

    def load(self, filename, directory):
        # Încărcăm parametrii actorului, criticului și value-ului din fișiere separate
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.value.load_state_dict(torch.load(f"{directory}/{filename}_value.pth"))


def evaluate(network, epoch, eval_episodes=10):
    #initializarea avg_reward unde stocam recompensa medie i col unde se va stoca numarul de coliziunii
    avg_reward = 0.0
    col = 0
    #parcurgem toate episoadele de evaluare
    for _ in range(eval_episodes):
        env.get_logger().info(f"evaluating episode {_}")
        count = 0
        #reseteam mediul de antrenare
        state = env.reset()
        done = False
        #atata timp cat done este false si nu sau facut m,ai mult de 500 de pasi 
        
        while not done and count < 501:
            #determinam actiunea folosind reteaua nuronala network.get_action()
            action = network.get_action(np.array(state))
            env.get_logger().info(f"action : {action}")
            # adaugă 1 la valoarea acțiunii pentru a asigura că aceasta este întotdeauna pozitivă și apoi se împarte la 2 pentru a scala valoarea în intervalul [0, 1]
            a_in = [(action[0] + 1) / 2, action[1]]
            #agentul efectueeaza a_in
            state, reward, done, _ = env.step(a_in)
            #crestem avg_rewadul cu rewardul pe care il primeste agentul 
            avg_reward += reward
            count += 1
            #daca rewardul e mai mic de -90 cretem numarul de coliziunii 
            if reward < -90:
                col += 1
    #calculam recompensa medie si numarul mediu de coliziunii            
    avg_reward /= eval_episodes

    avg_col = col / eval_episodes
    env.get_logger().info("..............................................")
    env.get_logger().info(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    env.get_logger().info("..............................................")
    return avg_reward

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')
        self.environment_dim = 20
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

    # Perform an action and read a new state
    def step(self, action):
        global velodyne_data
        target = False
        
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(velodyne_data)
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y
        quaternion = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service("/gazebo/reset_world")
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")

        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = float(abs(action[0]))
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5.0
        marker2.pose.position.y = 0.0
        marker2.pose.position.z = 0.0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = float(abs(action[1]))
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5.0
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0.0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            env.get_logger().info("Collision is detected!")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            env.get_logger().info("reward 100")
            return 100.0
        elif collision:
            env.get_logger().info("reward -100")
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class Velodyne_subscriber(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            10)
        self.subscription

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

if __name__ == '__main__':

    rclpy.init(args=None)

    seed = 0  # Random seed number
    eval_freq = 2e3  # After how many steps to perform the evaluation
    max_ep = 500  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        500000  # Number of steps over which the initial exploration noise will decay over
    )
    expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 40  # Size of the mini-batch
    discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
    tau = 0.005  # Soft target update variable (should be close to 0)
    policy_noise = 0.2  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e6  # Maximum size of the buffer
    file_name = "SAC_velodyne"  # name of the file to store the policy
    save_model = True  # Weather to save the model or not
    load_model = False  # Weather to load a stored model
    random_near_obstacle = True  # To take random actions near obstacles or not

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Create the training environment
    environment_dim = 20
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    # Create the network
    network = SAC(state_dim, action_dim, max_action)
    # Create a replay buffer
    
    if load_model:
        try:
            print("Will load existing model.")
            network.load(file_name, "./pytorch_models")
        except:
            print("Could not load the stored model parameters, initializing training with random parameters")

    # Create evaluation data store
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1
    alpha=0.2
    # Instanțierea ReplayBuffer
    replay_buffer = ReplayBuffer()

# Adăugarea tranzițiilor în ReplayBuffer
    
    count_rand_actions = 0
    random_action = []

    env = GazeboEnv()
    odom_subscriber = Odom_subscriber()
    velodyne_subscriber = Velodyne_subscriber()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)
    try:
        while rclpy.ok():
            if timestep < max_timesteps:
            # On termination of episode
                if done:
                    env.get_logger().info(f"Done. timestep : {timestep}")
                    if timestep != 0:
                        env.get_logger().info(f"train")
                        network.train(
                            replay_buffer,
                            episode_timesteps,
                            batch_size,
                            discount,
                            tau,
                            alpha
                        )

                    if timesteps_since_eval >= eval_freq:
                        env.get_logger().info("Validating")
                        timesteps_since_eval %= eval_freq
                        evaluations.append(
                            evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
                        )

                        network.save(file_name, directory="./DRL_robot_navigation_ros2/src/td3/scripts/pytorch_models")
                        np.save("./DRL_robot_navigation_ros2/src/td3/scripts/results/%s" % (file_name), evaluations)
                        epoch += 1

                    state = env.reset()
                    done = False

                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

                # add some exploration noise
                if expl_noise > expl_min:
                    expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

                action = network.get_action(np.array(state))
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                     -max_action, max_action
                )

                # If the robot is facing an obstacle, randomly force it to take a consistent random action.
                # This is done to increase exploration in situations near obstacles.
                # Training can also be performed without it
                if random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(state[4:-8]) < 0.6
                        and count_rand_actions < 1
                    ):
                        count_rand_actions = np.random.randint(8, 15)
                        random_action = np.random.uniform(-1, 1, 2)

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action = random_action
                        action[0] = -1

                # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
                a_in = [(action[0] + 1) / 2, action[1]]
                next_state, reward, done, target = env.step(a_in)
                done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
                done = 1 if episode_timesteps + 1 == max_ep else int(done)
                episode_reward += reward

                # Save the tuple in replay buffer
                replay_buffer.add(state, action, reward, next_state, done_bool)

                # Update the counters
                state = next_state
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()