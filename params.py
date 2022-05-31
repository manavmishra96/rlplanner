# Class definition of all the constant parameters used in the code

class parameters():
    def __init__(self):
        self.slippery = 0.0
        self.BOXES_LST = {
                    '(X0, Y0)' : [[2,0], 'key'],
                    '(X1, Y1)' : [[4,2], 'key'],
                    '(X2, Y2)' : [[3,5], 'box'],
                    '(X3, Y3)' : [[0,2], 'box']
                    }

        self.colors = {1: [230, 190, 255], 2: [170, 255, 195], 3: [255, 250, 200],
                       4: [255, 216, 177], 5: [250, 190, 190], 6: [240, 50, 230], 7: [145, 30, 180], 8: [67, 99, 216],
                       9: [66, 212, 244], 10: [60, 180, 75], 11: [191, 239, 69], 12: [255, 255, 25], 13: [245, 130, 49],
                       14: [230, 25, 75], 15: [128, 0, 0], 16: [154, 99, 36], 17: [128, 128, 0], 18: [70, 153, 144],
                       0: [0, 0, 117]}

        self.num_colors = len(self.colors)
        self.agent_color = [128, 128, 128]
        self.goal_color = [255, 255, 255]
        self.grid_color = [220, 220, 220]
        self.wall_color = [0, 0, 0]

        self.boxsize = 6
        self.goal_length = 2
        self.num_distractor = 1
        self.distractor_length = 1
        self.seed = 10