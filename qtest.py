import numpy as np
import matplotlib.pyplot as plt

from frozenlake import FrozenLakeEnv
from boxworldgen import WorldSetup

qtable = np.load('models/qtable33.npy')
max_steps = 99

box = WorldSetup()
desc = box.get_map()

env = FrozenLakeEnv(desc=desc, slip_ratio=box.slip_ratio)

env.reset()
test_rewards = []

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    # list_action = []
    print("\n********************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        action = np.argmax(qtable[state, :])
        # list_action.append(action)
        new_state, reward, done, info = env.step(action)

        if done:
            env.render()
            print("We reached our Goal 🏆")
            print("Number of steps", step)
            break
        total_rewards += reward
        state = new_state
    test_rewards.append(total_rewards)

# print("Score over time: " + str(sum(test_rewards)/total_episodes))
print("\n Testing over!")
env.close()