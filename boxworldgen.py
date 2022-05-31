import numpy as np
import random
# import pygame
from params import parameters

P = parameters()

class WorldSetup():
    def __init__(self):
       pass
        
    def get_map_random(self):
        # BoxWorld Environment Generation
        self.world = self.world_gen(n=6, goal_length=2, num_distractor=1, distractor_length=1, seed=10)

        desc = np.full((self.world.shape[0], self.world.shape[1]), 'F')
        desc = np.where(self.world[:,:,0] != 220, 'H', desc)

        desc = desc[1:-1, 1:-1]
        desc = np.asarray(desc, dtype="c")

        Hrow, Hcol = np.where(desc == b"H")
        Hidx = len(Hrow)

        # Set start position S and goal position G
        Sidx, Gidx = np.random.choice(range(Hidx), 2, replace=False)

        srow, scol = Hrow[Sidx], Hcol[Sidx]
        grow, gcol = Hrow[Gidx], Hcol[Gidx]

        desc[srow, scol] = b"S"
        desc[grow, gcol] = b"G"

        self.desc = desc
        return desc


    def get_map(self, start, goal):
        # BoxWorld Environment Generation
        self.world = self.world_gen(n=P.boxsize, goal_length=P.goal_length, num_distractor=P.num_distractor, 
                                    distractor_length=P.num_distractor, seed=P.seed)

        desc = np.full((self.world.shape[0], self.world.shape[1]), 'F')
        desc = np.where(self.world[:,:,0] != 220, 'H', desc)

        desc = desc[1:-1, 1:-1]
        desc = np.asarray(desc, dtype="c")

        srow, scol = P.BOXES_LST[start][0]
        grow, gcol = P.BOXES_LST[goal][0]

        if P.BOXES_LST[start][1] == 'box':
            srow = srow; scol = scol - 1

        desc[srow, scol] = b"S"
        desc[grow, gcol] = b"G"

        self.desc = desc
        return desc


    def sampling_pairs(self, num_pair, n=12):
        possibilities = set(range(1, n*(n-1)))
        keys = []
        locks = []
        for k in range(num_pair):
            key = random.sample(possibilities, 1)[0]
            key_x, key_y = key//(n-1), key%(n-1)
            lock_x, lock_y = key_x, key_y + 1
            to_remove = [key_x * (n-1) + key_y] +\
                        [key_x * (n-1) + i + key_y for i in range(1, min(2, n - 2 - key_y) + 1)] +\
                        [key_x * (n-1) - i + key_y for i in range(1, min(2, key_y) + 1)]

            possibilities -= set(to_remove)
            keys.append([key_x, key_y])
            locks.append([lock_x, lock_y])
        agent_pos = random.sample(possibilities, 1)
        possibilities -= set(agent_pos)
        first_key = random.sample(possibilities, 1)

        agent_pos = np.array([agent_pos[0]//(n-1), agent_pos[0]%(n-1)])
        first_key = first_key[0]//(n-1), first_key[0]%(n-1)
        return keys, locks, first_key, agent_pos


    def world_gen(self, n=12, goal_length=3, num_distractor=2, distractor_length=2, seed=None):
        """
        generate BoxWorld
        """
        if seed is not None:
            random.seed(seed)

        world_dic = {} # dic keys are lock positions, value is 0 if distractor, else 1.
        world = np.ones((n, n, 3)) * 220
        goal_colors = random.sample(range(P.num_colors), goal_length - 1)
        distractor_possible_colors = [color for color in range(P.num_colors) if color not in goal_colors]
        distractor_colors = [random.sample(distractor_possible_colors, distractor_length) for k in range(num_distractor)]
        distractor_roots = random.choices(range(goal_length - 1), k=num_distractor)
        keys, locks, first_key, agent_pos = self.sampling_pairs(goal_length - 1 + distractor_length * num_distractor, n)


        # first, create the goal path
        for i in range(1, goal_length):
            if i == goal_length - 1:
                color = P.goal_color  # final key is white
            else:
                color = P.colors[goal_colors[i]]
            # print("place a key with color {} on position {}".format(color, keys[i-1]))
            # print("place a lock with color {} on {})".format(colors[goal_colors[i-1]], locks[i-1]))
            world[keys[i-1][0], keys[i-1][1]] = np.array(color)
            world[locks[i-1][0], locks[i-1][1]] = np.array(P.colors[goal_colors[i-1]])
            world_dic[tuple(locks[i-1] + np.array([1, 1]))] = 1

        # keys[0] is an orphand key so skip it
        world[first_key[0], first_key[1]] = np.array(P.colors[goal_colors[0]])
        # world_dic[first_key[0]+1, first_key[1]+2] = 1
        # print("place the first key with color {} on position {}".format(goal_colors[0], first_key))

        # place distractors
        for i, (distractor_color, root) in enumerate(zip(distractor_colors, distractor_roots)):
            key_distractor = keys[goal_length-1 + i*distractor_length: goal_length-1 + (i+1)*distractor_length]
            color_lock = P.colors[goal_colors[root]]
            color_key = P.colors[distractor_color[0]]
            world[key_distractor[0][0], key_distractor[0][1] + 1] = np.array(color_lock)
            world[key_distractor[0][0], key_distractor[0][1]] = np.array(color_key)
            world_dic[key_distractor[0][0] + 1, key_distractor[0][1] + 2] = 0
            for k, key in enumerate(key_distractor[1:]):
                color_lock = P.colors[distractor_color[k]]
                color_key = P.colors[distractor_color[k-1]]
                world[key[0], key[1]] = np.array(color_key)
                world[key[0], key[1]+1] = np.array(color_lock)
                world_dic[key[0] + 1, key[1] + 2] = 0

        # place an agent
        world[agent_pos[0], agent_pos[1]] = np.array(P.agent_color)
        agent_pos += np.array([1, 1])

        # add black wall
        wall_0 = np.zeros((1, n, 3))
        wall_1 = np.zeros((n+2, 1, 3))
        world = np.concatenate((wall_0, world, wall_0), axis=0)
        world = np.concatenate((wall_1, world, wall_1), axis=1)

        return world

    def update_color(self, world, previous_agent_loc, new_agent_loc):
            world[previous_agent_loc[0], previous_agent_loc[1]] = P.grid_color
            world[new_agent_loc[0], new_agent_loc[1]] = P.agent_color

    def is_empty(self, room):
        return np.array_equal(room, self.grid_color) or np.array_equal(room, P.agent_color)