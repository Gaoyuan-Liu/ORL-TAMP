from __future__ import print_function

import copy
import pybullet as p
import random
import time, os, sys
from itertools import islice, count
import pandas as pd

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../../pddlstream/')

from examples.pybullet.utils.pybullet_tools.panda_primitives import Pose
from examples.pybullet.utils.pybullet_tools.utils import pairwise_collision




##################################################

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../state_discriminator/')




##################################################

def get_loose_goal_gen(problem, collisions=True, **kwargs):
    data = pd.read_csv('./../state_discriminator/data/data_dish.csv')
    success = data[data['success'] == 1]
    obstacles = list(set(problem.fixed).difference(problem.marks)) if collisions else []

    def gen(body):
        obstacles = list(set(problem.fixed).difference(problem.marks+[body])) if collisions else []
        while True:
            sample = success.sample() #random.choice(goals)
            # body_pose = ((sample['x_g'].to_numpy()[0], sample['y_g'].to_numpy()[0], 0.31), (0.0, 0.0, 0.0, 1.0))
            body_pose = ((0.4, 0.25, 0.17), (0.0, 0.0, 0.0, 1.0))
            ap = Pose(body, body_pose)
            ap.assign()
            if not any(pairwise_collision(body, o) for o in obstacles):
                yield (ap,)
            # yield (ap,)
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

def get_push_initial_test(problem, collisions=True):
    model = Net(input_shape=4)
    name = 'state_discriminator_cube'
    model.load_state_dict(torch.load('./%s.pth' % (name), map_location='cpu'))

    def test(b, p, lg):
        p.assign()
        input = np.array([p.value[0][0], p.value[0][1], lg.value[0][0], lg.value[0][1]], dtype=np.float32)
        input = torch.tensor(input,dtype=torch.float32)
        model.eval()
        output = model(input).detach().numpy().round()
        if output == 1:
            return True
        else:
            return True
        # return True
    return test

##################################################

def get_same_surface_test(problem, collisions=True):
    def test(o, p, gl): # Maybe now only define as if they have the same height
        # if abs(p.value[0][2] - gl.value[0][2]) >= 0.01:
        #     return False
        if p.support != gl.support:
            return False
        return True
    return test



def main():
    gen = get_loose_goal_gen(None, collisions=True)
    for i in range(5):
        # gen(None)
        print(next(gen(None))[0].value)



if __name__ == '__main__':
    main()

