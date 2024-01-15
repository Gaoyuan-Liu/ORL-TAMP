from __future__ import print_function

import copy
import pybullet as p
import random
import time, os, sys
from itertools import islice, count

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose




##################################################

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../state_discriminator/')

from utils import save_list, load_list



##################################################

def get_loose_goal_gen(problem, collisions=True, **kwargs):
    # goals = load_list('./../state_discriminator/goals')
    def gen(body):
        while True:
            # goal = random.choice(goals)
            x = np.random.uniform(low=-0.0, high=1)
            y = np.random.uniform(low=-1, high=1)
            x = 0.5
            y = 0.2
            body_pose = ((x, y, 0.035), (0.0, 0.0, 0.0, 1.0))
            ap = Pose(body, body_pose)
            yield (ap,)
    return gen

##################################################


class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(32,64)
    self.fc3 = nn.Linear(64,1)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x

def get_hook_initial_test(problem, collisions=True):
    # model = Net(input_shape=3)
    # name = 'state_discriminator_cube'
    # model.load_state_dict(torch.load('./%s.pth' % (name), map_location='cpu'))

    def test(b, p):
        # # p.assign()
        # input = np.array([p.value[0][0], p.value[0][1], 0])
        # input = torch.tensor(input,dtype=torch.float32)
        # model.eval()
        # output = model(input).detach().numpy().round()

        # if output == 1:
        #     return True
        # else:
        #     return False
        return True
    return test

        

##################################################



def main():
    gen = get_loose_goal_gen(None, collisions=True)
    for i in range(5):
        # gen(None)
        print(next(gen(None))[0].value)








if __name__ == '__main__':
    main()

