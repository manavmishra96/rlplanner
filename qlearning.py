# Q Learning - Frozen lake

import numpy as np
import gym
import random
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from frozenlake import FrozenLakeEnv
from boxworldgen import WorldSetup
from params import parameters

DIRECTIONS = {
    0: "here",
    1: "e",
    2: "ne",
    3: "n",
    4: "nw",
    5: "w",
    6: "sw",
    7: "s",
    8: "se"
}

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# env = gym.make("FrozenLake-v1", desc=desc)
P = parameters()
box = WorldSetup()
desc = box.get_map_random()

env = FrozenLakeEnv(desc=desc, slippery=P.slippery)

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))


# Hyper-parameters
total_episodes = 30000
learning_rate = 0.7
max_steps = 99
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

train_rewards = []
epsilon_list = []

for episode in tqdm(range(total_episodes)):
    desc = box.get_map_random()

    env = FrozenLakeEnv(desc=desc, slippery=P.slippery)
    state = env.reset()

    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0,1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        
        qtable[state, action] += learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward
        state = new_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    train_rewards.append(total_rewards)
    epsilon_list.append(epsilon)

print("Score over time: " + str(sum(train_rewards)/total_episodes))
# SCORES_LST.append(sum(train_rewards)/total_episodes)
action_lst = [np.argmax(i) for i in qtable]
ACTION_MAP = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
Q_action = [ACTION_MAP[a] for a in action_lst]
# print("Q* action = \n", Q_action)
print("\n Training over!")

file = 'models/qtable' + str(int(P.slippery * 100)) + '.npy'
np.save(file, qtable)