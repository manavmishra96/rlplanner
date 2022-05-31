import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from frozenlake import FrozenLakeEnv
from boxworldgen import WorldSetup
from params import parameters


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

 
dir = 'images/'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

# execute the plan
count = 0
box = WorldSetup()
P = parameters()
desc = box.get_map_random()

BoxWorld = box.world
CopyBoxWorld = BoxWorld.copy()

img = BoxWorld.astype(np.uint8)
plt.imshow(img, vmin=0, vmax=255, interpolation='none')
plt.axis('off')
plt.savefig('images/img' + str(count) + '.png')
count += 1

qtable = np.load('models/qtable0.npy')


for plan_step in PLAN:
    print(plan_step)

    # MOVE motion
    if plan_step[0] == 'MOVE':
      start_box = '(' + plan_step[1] + ', ' + plan_step[2] + ')'
      end_box = '(' + plan_step[3] + ', ' + plan_step[4] + ')'

      desc = box.get_map(start_box, end_box)
    
      env = FrozenLakeEnv(desc=desc, slippery=P.slippery)
      state = env.reset()
      step = 0
      done = False
      total_rewards = 0

      for step in range(99):
          action = np.argmax(qtable[state, :])
          # list_action.append(action)
          new_state, reward, done, info = env.step(action)

          cur_row, cur_col, _ = env.decoder(state)
          new_row, new_col, _ = env.decoder(new_state)

          BoxWorld[new_row + 1, new_col + 1] = P.agent_color
          BoxWorld[cur_row + 1, cur_col + 1] = P.grid_color

          img = BoxWorld.astype(np.uint8)
          plt.imshow(img, vmin=0, vmax=255, interpolation='none')
          plt.axis('off')
          plt.savefig('images/img' + str(count) + '.png')
          count += 1

          if done:
              # env.render()
              print("We reached our Goal 🏆")
              print("Number of steps " + str(step) + "\n")
              break
          total_rewards += reward
          state = new_state

      if not done:
        print('Goal not reached!')

    

    # PICKUP-KEY motion
    if plan_step[0] == 'PICKUP-KEY':
      # env.desc = box.go_left()
      env.step(0, det=True)
      # env.render()
      print("Picked key! 🔑")

      cur_row, cur_col, _ = env.decoder(env.s)
      BoxWorld[0, 0] = CopyBoxWorld[cur_row + 1, cur_col + 1]
      img = BoxWorld.astype(np.uint8)
      plt.imshow(img, vmin=0, vmax=255, interpolation='none')
      plt.axis('off')
      plt.savefig('images/img' + str(count) + '.png')
      count += 1


    # PICKUP-GEM motion
    if plan_step[0] == 'PICKUP-GEM':
      # env.desc = box.go_left()
      # env.reset()
      env.step(0, det=True)
      # env.render()
      print("Gem achieved! 💎")

      cur_row, cur_col, _ = env.decoder(env.s)
      BoxWorld[0, 0] = P.goal_color
      img = BoxWorld.astype(np.uint8)
      plt.imshow(img, vmin=0, vmax=255, interpolation='none')
      plt.axis('off')
      plt.savefig('images/img' + str(count) + '.png')
      count += 1
      

    # UNLOCK-BOX motion
    if plan_step[0] == 'UNLOCK-BOX':
      # env.render()
      print("Box unlocked! 🎁 \n")

      cur_row, cur_col, _ = env.decoder(env.s)
      BoxWorld[cur_row + 1, cur_col + 1] = P.grid_color
      BoxWorld[cur_row + 1, cur_col] = P.agent_color
      img = BoxWorld.astype(np.uint8)
      plt.imshow(img, vmin=0, vmax=255, interpolation='none')
      plt.axis('off')
      plt.savefig('images/img' + str(count) + '.png')
      count += 1

print("Total images: ", count)
# Execution completed!

image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()