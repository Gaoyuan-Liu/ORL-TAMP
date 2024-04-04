import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


import copy
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from utils import save_list, load_list
import sys, os


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../utils/')


from env import HookingEnv 
from scenario import Scenario

from pybullet_tools.transformations import quaternion_from_euler
import pybullet as p

import cv2



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

class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]

  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length


################################################################

SHOWING_WHAT = 'cube'# 'hook' 

def main():

  env = HookingEnv()
  scn = Scenario()
  scn.reset()



  # ----------------

  # Model

  state = env.reset_whole_workspace()
  state = state[0].reshape(-1)




  # print(f'\n state_shape = {state.shape}\n')

  epochs = 400
  # Model , Optimizer, Loss
  model = Net(input_shape=state.shape[0])
  name = 'state_discriminator_' + SHOWING_WHAT
  model.load_state_dict(torch.load('./%s.pth' % (name), map_location='cpu'))


  # ----------------
  # Camera

  viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[1, -1, 1.6],
    cameraTargetPosition=[0.5, 0, 0],
    cameraUpVector=[0, 0, 1])
  
  projectionMatrix = p.computeProjectionMatrixFOV(
                fov= 65,#math.atan(0.5) * (180/math.pi) * 2 #
                aspect= 1.0,#1.0,
                nearVal= 0.1,
                farVal=5)
  
  width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                                              width=1000, 
                                              height=1000,
                                              viewMatrix=viewMatrix,
                                              projectionMatrix=projectionMatrix, 
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)





      

  
  for i in range(epochs):
      
    state = env.reset_whole_workspace(render=False)
    state = state[0].reshape(-1)
    # state[0] = [0.9, 0.0, 0.0]
    # p.resetBasePositionAndOrientation(env.objects[0], [state[0][0], state[0][1], 0.025], [0,0,0,1])

    
    inputs = state.reshape(-1)
    inputs = torch.tensor(inputs,dtype=torch.float32)
    model.eval()
    output = model(inputs).detach().numpy().round()

    

    print(output)




    # ----------------
    
    if SHOWING_WHAT == 'hook':
      if output == 1:
        position = [state[-1][0], state[-1][1], 0.02]
        print(state[-1][2])
        orn = quaternion_from_euler(0,0,state[-1][2])
        print(orn)
        pose = (position, orn)
        scn.add_hook_mark(pose, color='green')
      else:
        position = [state[-1][0], state[-1][1], 0.02]
        print(state[-1][2])
        orn = quaternion_from_euler(0,0,state[-1][2])
        print(orn)
        pose = (position, orn)
        scn.add_hook_mark(pose, color='red')

    # ----------------

    elif SHOWING_WHAT == 'cube':
      if output == 1:
        position = [state[0], state[1], 0.02]
        print(state[2])
        orn = quaternion_from_euler(0,0,state[2])
        print(orn)
        pose = (position, orn)
        scn.add_cube_mark(pose, color='green')
      else:
        position = [state[0], state[1], 0.02]
        print(state[2])
        orn = quaternion_from_euler(0,0,state[2])
        print(orn)
        pose = (position, orn)
        scn.add_cube_mark(pose, color='red')
      
      print('success')
    
  
      
      

  input('press enter to take photo.')
  width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                                              width=1000, 
                                              height=1000,
                                              viewMatrix=viewMatrix,
                                              projectionMatrix=projectionMatrix, 
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
  cv2.cv2.imwrite('./image.jpg', cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR))

  # image = p.getCameraImage(cam_width, cam_height, cam_view_matrix, cam_projection_matrix)[2][:, :, :3]
  # cv2.cv2.imwrite('./image.png', image)

  input('press enter to exit.')

        



    
    


    # save_list(losses, 'losses')
    # save_list(accur, 'accur')









    

if __name__ == '__main__':
  main()