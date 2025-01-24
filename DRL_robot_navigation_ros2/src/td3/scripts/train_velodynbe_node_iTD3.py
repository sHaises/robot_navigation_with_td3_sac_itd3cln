#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import threading

import math
import random

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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        # Definim primul strat liniar care primește dimensiunea stării și produce 800 de neuroni
        self.layer_1 = nn.Linear(state_dim, 800)
        
        # Definim al doilea strat liniar care primește 800 de neuroni și produce 600 de neuroni
        self.layer_2 = nn.Linear(800, 600)
        
        # Definim al treilea strat liniar care primește 600 de neuroni și produce dimensiunea acțiunii
        self.layer_3 = nn.Linear(600, action_dim)
        
        # Funcția de activare Tanh pentru a scala ieșirea acțiunii între -1 și 1
        self.tanh = nn.Tanh()

    def forward(self, s):
        # Aplicăm funcția de activare ReLU pe ieșirea primului strat
        s = F.relu(self.layer_1(s))
        
        # Aplicăm funcția de activare ReLU pe ieșirea celui de-al doilea strat
        s = F.relu(self.layer_2(s))
        
        # Aplicăm funcția de activare Tanh pe ieșirea celui de-al treilea strat pentru a obține acțiunea
        a = self.tanh(self.layer_3(s))
        
        # Returnăm acțiunea
        return a

# Definim clasa Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Primul set de straturi pentru primul critic
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)  # Stratul pentru stări
        self.layer_2_a = nn.Linear(action_dim, 600)  # Stratul pentru acțiuni
        self.layer_3 = nn.Linear(600, 1)  # Stratul final care produce valoarea Q

        # Al doilea set de straturi pentru al doilea critic
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)  # Stratul pentru stări
        self.layer_5_a = nn.Linear(action_dim, 600)  # Stratul pentru acțiuni
        self.layer_6 = nn.Linear(600, 1)  # Stratul final care produce valoarea Q

    def forward(self, s, a):
        # Primul critic
        s1 = F.relu(self.layer_1(s))  # Aplicăm ReLU pe ieșirea primului strat
        self.layer_2_s(s1)  # Aplicăm stratul pentru stări (doar pentru actualizarea ponderilor)
        self.layer_2_a(a)  # Aplicăm stratul pentru acțiuni (doar pentru actualizarea ponderilor)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())  # Calculăm produsul matricial pentru stări
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())  # Calculăm produsul matricial pentru acțiuni
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)  # Aplicăm ReLU pe suma ieșirilor + bias
        q1 = self.layer_3(s1)  # Calculăm valoarea Q finală

        # Al doilea critic
        s2 = F.relu(self.layer_4(s))  # Aplicăm ReLU pe ieșirea primului strat
        self.layer_5_s(s2)  # Aplicăm stratul pentru stări (doar pentru actualizarea ponderilor)
        self.layer_5_a(a)  # Aplicăm stratul pentru acțiuni (doar pentru actualizarea ponderilor)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())  # Calculăm produsul matricial pentru stări
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())  # Calculăm produsul matricial pentru acțiuni
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)  # Aplicăm ReLU pe suma ieșirilor + bias
        q2 = self.layer_6(s2)  # Calculăm valoarea Q finală

        # Returnăm valorile Q calculate de cei doi critici
        return q1, q2

# Definim clasa CLN (Collision Likelihood Network)
class CLN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CLN, self).__init__()

        # Definim primul strat liniar care primește atât starea, cât și acțiunea și produce 400 de neuroni
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        
        # Definim al doilea strat liniar care primește 400 de neuroni și produce 300 de neuroni
        self.layer_2 = nn.Linear(400, 300)
        
        # Definim al treilea strat liniar care produce probabilitatea de coliziune
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, s, a):
        # Concatenăm starea și acțiunea într-un singur tensor
        x = torch.cat([s, a], 1)
        
        # Aplicăm funcția de activare ReLU pe ieșirea primului strat
        x = F.relu(self.layer_1(x))
        
        # Aplicăm funcția de activare ReLU pe ieșirea celui de-al doilea strat
        x = F.relu(self.layer_2(x))
        
        # Calculăm probabilitatea de coliziune
        collision_prob = self.layer_3(x)
        
        # Returnăm probabilitatea de coliziune
        return collision_prob


# iTD3-CLN network
class itd3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # Setăm parametrii actor-target identici cu cei ai actorului
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())  # Optimizator pentru actor

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # Setăm parametrii critic-target identici cu cei ai criticului
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())  # Optimizator pentru critic

        # Initialize the Collision Learning Network (CLN)
        self.cln = CLN(state_dim, action_dim).to(device)
        self.cln_target = CLN(state_dim, action_dim).to(device)
        self.cln_target.load_state_dict(self.cln.state_dict())  # Setăm parametrii cln-target identici cu cei ai cln
        self.cln_optimizer = torch.optim.Adam(self.cln.parameters())  # Optimizator pentru CLN

        self.max_action = max_action  # Valoarea maximă pe care o poate avea o acțiune
        self.writer = SummaryWriter(log_dir="./DRL_robot_navigation_ros2/src/td3/scripts/runs")  # Obiect pentru scrierea în TensorBoard
        self.iter_count = 0  # Contor pentru numărul de iterații de antrenament

    def get_action(self, state):
        # Funcție pentru a obține acțiunea de la actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)  # Transformăm starea într-un tensor și îl mutăm pe GPU (dacă este specificat)
        return self.actor(state).cpu().data.numpy().flatten()  # Obținem acțiunea, o mutăm pe CPU și o convertim într-un șir de numpy

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0  # Variabilă pentru media valorilor Q
        max_Q = -float('inf')  # Variabilă pentru valoarea maximă a Q
        av_loss = 0  # Variabilă pentru media pierderilor
        for it in range(iterations):
            # Amestecăm un lot din buffer-ul de redare
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)  # Starea din lot, pe GPU
            next_state = torch.Tensor(batch_next_states).to(device)  # Următoarea stare din lot, pe GPU
            action = torch.Tensor(batch_actions).to(device)  # Acțiunea din lot, pe GPU
            reward = torch.Tensor(batch_rewards).to(device)  # Răsplata din lot, pe GPU
            done = torch.Tensor(batch_dones).to(device)  # Flagul de finalizare din lot, pe GPU

            # Obținem acțiunea estimată pentru următoarea stare folosind actorul-target
            next_action = self.actor_target(next_state)

            # Adăugăm zgomot la acțiune
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)  # Generăm zgomot normal
            noise = noise.clamp(-noise_clip, noise_clip)  # Limităm zgomotul
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)  # Adăugăm zgomotul la acțiune și o limităm la valoarea maximă a acțiunii

            # Calculăm valorile Q din rețeaua critică-target pentru perechea următoare stare-acțiune
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Selectăm valoarea minimă a Q dintre cele două calculate
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)  # Adăugăm valoarea medie a Q la variabila av_Q
            max_Q = max(max_Q, torch.max(target_Q))  # Actualizăm valoarea maximă a Q

            # Calculăm valoarea Q finală folosind parametrii rețelei target cu ecuația Bellman
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Obținem valorile Q ale rețelelor de bază cu parametrii curenți
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculăm pierderea între valoarea Q curentă și valoarea Q țintă
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Efectuăm descendentul gradientului pentru critic
            self.critic_optimizer.zero_grad()  # Resetăm gradienții
            loss.backward()  # Efectuăm înapoierea
            self.critic_optimizer.step()  # Aplicăm pasul gradientului

            # Actualizăm CLN (Collision Learning Network)
            collision_prob = self.cln(state, action)  # Obținem probabilitatea de coliziune
            target_collision_prob = self.cln_target(next_state, next_action).detach()  # Obținem probabilitatea de coliziune țintă
            cln_loss = F.mse_loss(collision_prob, target_collision_prob)  # Calculăm pierderea pentru CLN

            self.cln_optimizer.zero_grad()  # Resetăm gradienții pentru CLN
            cln_loss.backward()  # Efectuăm înapoierea pentru CLN
            self.cln_optimizer.step()  # Aplicăm pasul gradientului pentru CLN

            if it % policy_freq == 0:
                # Maximizăm valoarea ieșirii actorului efectuând gradientul descendent pe valorile negative ale Q
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()  # Resetăm gradienții pentru actor
                actor_grad.backward()  # Efectuăm înapoierea pentru actor
                self.actor_optimizer.step()  # Aplicăm pasul gradientului pentru actor

                # Actualizăm parametrii rețelei actor-target folosind o actualizare moale
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                # Actualizăm parametrii rețelei critic-target folosind o actualizare moale
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                # Actualizăm parametrii rețelei CLN-target folosind o actualizare moale
                for param, target_param in zip(
                    self.cln.parameters(), self.cln_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss  # Adăugăm pierderea la av_loss

        self.iter_count += 1  # Incrementăm numărul de iterații

        # Scriem noile valori în TensorBoard
        env.get_logger().info(f"writing new results for a tensorboard")
        env.get_logger().info(f"loss, Av.Q, Max.Q, iterations : {av_loss / iterations}, {av_Q / iterations}, {max_Q}, {self.iter_count}")
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        # Salvăm parametrii actorului, criticului și CLN în fișiere separate
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
        torch.save(self.cln.state_dict(), "%s/%s_cln.pth" % (directory, filename))

    def load(self, filename, directory):
        # Încărcăm parametrii actorului, criticului și CLN din fișiere separate
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
        self.cln.load_state_dict(
            torch.load("%s/%s_cln.pth" % (directory, filename))
        )


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
    file_name = "itd3"  # name of the file to store the policy
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
    network = itd3(state_dim, action_dim, max_action)
    # Create a replay buffer
    replay_buffer = ReplayBuffer(buffer_size, seed)
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
                        policy_noise,
                        noise_clip,
                        policy_freq,
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
                replay_buffer.add(state, action, reward, done_bool, next_state)

                # Update the counters
                state = next_state
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()