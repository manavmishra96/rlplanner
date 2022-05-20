import numpy as np
import matplotlib.pyplot as plt

from frozenlake import FrozenLakeEnv
from boxworldgen import WorldSetup

# open the plan file and extract data
filename = "plan.txt"

PLAN = []

f = open(filename,'r')
    
while True:
    line = f.readline()
    if line.startswith('step'):
        PLAN.append(line.split()[2:])
        break

while True:
    line = f.readline()
    if line.startswith('time'):
        break
    else:
        PLAN.append(line.split()[1:])

PLAN = list(filter(None, PLAN))

# execute the plan
box = WorldSetup()
qtable = np.load('models/qtable33.npy')


for plan_step in PLAN:
    print(plan_step)

    if plan_step[0] == 'MOVE':
      start_box = '(' + plan_step[1] + ', ' + plan_step[2] + ')'
      end_box = '(' + plan_step[3] + ', ' + plan_step[4] + ')'

      desc = box.get_map(start_box, end_box)

      env = FrozenLakeEnv(desc=desc, slip_ratio=0.33)
      state = env.reset()
      step = 0
      done = False
      total_rewards = 0

      for step in range(99):
          action = np.argmax(qtable[state, :])
          # list_action.append(action)
          new_state, reward, done, info = env.step(action)

          if done:
              env.render()
              print("We reached our Goal 🏆")
              print("Number of steps " + str(step) + "\n")
              break
          total_rewards += reward
          state = new_state
    
    if plan_step[0] == 'PICKUP-KEY':
      env.render()
      print("Picked key! 🔑")

    if plan_step[0] == 'PICKUP-GEM':
      env.render()
      print("Gem achieved! 💎")

    if plan_step[0] == 'UNLOCK-BOX':
      env.render()
      print("Box unlocked! 🎁 \n")