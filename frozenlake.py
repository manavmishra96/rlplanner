from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import pygame
from pygame.constants import SRCALPHA
import numpy as np
from tqdm import tqdm

import gym
from gym import Env, spaces, utils


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

# Modified frozen lake environment

class FrozenLakeEnv(gym.Env):
    def __init__(self, desc=None, is_slippery=True, slip_ratio = 0.33):
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.ndir = ndir = 9
        self.reward_range = (-1, 0)

        if is_slippery:
            self.probable_slip = slip_ratio

        nA = 4
        nS = nrow * ncol * ndir

        self.goalX, self.goalY = np.where(desc == b'G')
        self.goalX, self.goalY = self.goalX[0], self.goalY[0]

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col, dir):
            return nrow * ncol * dir + row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, dir, action):
            newrow, newcol = inc(row, col, action)
            newdir = self.bearing_m2(self.goalX - newrow, self.goalY - newcol)

            oldstate = to_s(row, col, dir)
            newstate = to_s(newrow, newcol, newdir)
            newletter = desc[newrow, newcol]
            if newletter == b"H":
                newstate = oldstate
            done = bytes(newletter) in b"G"
            reward = 0 if (newletter == b"G") else -1
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                for dir in range(ndir):
                    s = to_s(row, col, dir)
                    for a in range(4):
                        li = self.P[s][a]
                        letter = desc[row, col]
                        if letter in b"G":
                            li.append((1.0, s, 0, True))
                        else:
                            if is_slippery:
                                li.append((self.probable_slip, *update_probability_matrix(row, col, dir, a)))
                                for b in [(a - 1) % 4, (a + 1) % 4]:
                                    li.append(
                                        (((1.0 - self.probable_slip)/2.0, *update_probability_matrix(row, col, dir, b)))
                                    )
                            else:
                                li.append((1.0, *update_probability_matrix(row, col, dir, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        # print(self.P)

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def step(self, a, det=False):
        transitions = self.P[self.s][a]
        if det:
            i = self.categorical_sample([1.0, 0, 0])
        else:
            i = self.categorical_sample([t[0] for t in transitions])
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

    def categorical_sample(self, prob_n):
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np.random.random()).argmax()

    def reset(self, return_info=False):
        idx = self.categorical_sample(self.initial_state_distrib)

        residue = int(idx/self.ncol)
        row = residue % self.nrow
        col = idx % self.ncol

        theta = self.bearing_m2(self.goalX - row, self.goalY - col) 
        self.s = self.nrow * self.nrow * theta  + idx
        self.lastaction = None

        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}

    def bearing_m1(self, dx, dy):
        if dx == 0 and dy == 0:
            return 0
        else:
            theta = np.arctan2(dy, dx) * 180 / np.pi
            if -22.5 < theta <= 22.5:
                return 1
            elif 22.5 < theta <= 67.5:
                return 2
            elif 67.5 < theta <= 112.5:
                return 3
            elif 112.5 < theta <= 157.5:
                return 4
            elif 157.5 < theta and theta <= -157.5:
                return 5
            elif -157.5 < theta <= -112.5:
                return 6 
            elif -112.5 < theta <= -67.5:
                return 7
            elif -67.5 < theta <= -22.5:
                return 8

    def bearing_m2(self, dy, dx): # dx = delta column and dy = delta row
        if dx == 0 and dy == 0:
            return 0
        elif dx > 0 and dy == 0:
            return 1
        elif dx > 0 and dy < 0:
            return 2
        elif dx == 0 and dy < 0:
            return 3
        elif dx < 0 and dy < 0:
            return 4
        elif dx < 0 and dy == 0:
            return 5
        elif dx < 0 and dy > 0:
            return 6
        elif dx == 0 and dy > 0:
            return 7
        elif dx > 0 and dy > 0:
            return 8


    def encoder(self, row, col):
        bearing = self.bearing_m2(self.goalX - row, self.goalY - col)
        return self.nrow * self.ncol * bearing + row * self.ncol + col

    def decoder(self, state): 
        col = state % self.ncol
        residue = int(state / self.ncol) 

        row = residue % self.nrow
        bearing = int(residue / self.nrow) 
        return row, col, DIRECTIONS[bearing]

    def render(self, mode="ansi"):
        desc = self.desc.tolist()
        if mode == "ansi":
            return self._render_text(desc)
        else:
            return self._render_gui(desc, mode)

    def _render_gui(self, desc, mode):
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Frozen Lake")
            if mode == "human":
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.image.load(file_name)
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.image.load(file_name)
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.image.load(file_name)
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.image.load(file_name)
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.image.load(file_name)
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [pygame.image.load(f_name) for f_name in elfs]

        board = pygame.Surface(self.window_size, flags=SRCALPHA)
        cell_width = self.window_size[0] // self.ncol
        cell_height = self.window_size[1] // self.nrow
        smaller_cell_scale = 0.6
        small_cell_w = smaller_cell_scale * cell_width
        small_cell_h = smaller_cell_scale * cell_height

        # prepare images
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]
        elf_scale = min(
            small_cell_w / elf_img.get_width(),
            small_cell_h / elf_img.get_height(),
        )
        elf_dims = (
            elf_img.get_width() * elf_scale,
            elf_img.get_height() * elf_scale,
        )
        elf_img = pygame.transform.scale(elf_img, elf_dims)
        hole_img = pygame.transform.scale(self.hole_img, (cell_width, cell_height))
        cracked_hole_img = pygame.transform.scale(
            self.cracked_hole_img, (cell_width, cell_height)
        )
        ice_img = pygame.transform.scale(self.ice_img, (cell_width, cell_height))
        goal_img = pygame.transform.scale(self.goal_img, (cell_width, cell_height))
        start_img = pygame.transform.scale(self.start_img, (small_cell_w, small_cell_h))

        for y in range(self.nrow):
            for x in range(self.ncol):
                rect = (x * cell_width, y * cell_height, cell_width, cell_height)
                if desc[y][x] == b"H":
                    self.window_surface.blit(hole_img, (rect[0], rect[1]))
                elif desc[y][x] == b"G":
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))
                    goal_rect = self._center_small_rect(rect, goal_img.get_size())
                    self.window_surface.blit(goal_img, goal_rect)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))
                    stool_rect = self._center_small_rect(rect, start_img.get_size())
                    self.window_surface.blit(start_img, stool_rect)
                else:
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))

                pygame.draw.rect(board, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (
            bot_col * cell_width,
            bot_row * cell_height,
            cell_width,
            cell_height,
        )
        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(cracked_hole_img, (cell_rect[0], cell_rect[1]))
        else:
            elf_rect = self._center_small_rect(cell_rect, elf_img.get_size())
            self.window_surface.blit(elf_img, elf_rect)

        self.window_surface.blit(board, board.get_rect())
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self, desc):
        outfile = StringIO()

        # row, col = self.s // self.ncol, self.s % self.ncol
        col = self.s % self.ncol
        residue = int(self.s / self.ncol) 
        row = residue % self.nrow

        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            print(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            print("\n")
        print("\n".join("".join(line) for line in desc) + "\n")
        with closing(outfile):
            return outfile.getvalue()