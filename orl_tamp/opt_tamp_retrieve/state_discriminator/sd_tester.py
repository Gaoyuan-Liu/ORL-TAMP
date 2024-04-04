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

import matplotlib.pyplot as plt


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../utils/')


from env import HookingEnv 
from scn import Scenario

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


def main():

  env = HookingEnv()
  # scn = Scenario()
  # scn.reset()



  # ----------------

  # Model

  state = env.reset()
  state = state[:2].reshape(-1)

  epochs = 400
  # Model , Optimizer, Loss
  model = Net(input_shape=state.shape[0])
  name = 'state_discriminator'
  model.load_state_dict(torch.load('./%s.pth' % (name), map_location='cpu'))


  # for i in range(epochs):
      
  #   state = env.reset()
  #   state = state[:2].reshape(-1)
  #   # state[0] = [0.9, 0.0, 0.0]
  #   # p.resetBasePositionAndOrientation(env.objects[0], [state[0][0], state[0][1], 0.025], [0,0,0,1])

    
  #   inputs = state.reshape(-1)
  #   inputs = torch.tensor(inputs,dtype=torch.float32)
  #   model.eval()
  #   output = model(inputs).detach().numpy().round()

  #   print(output)

  #   # ----------------
  #   if output == 1:
  #     position = [state[0], state[1], 0.02]
  #     orn = [0,0,0,1]#quaternion_from_euler(0,0,state[2])
  #     pose = (position, orn)
  #     scn.add_cup_mark(pose, color='green')
  #   else:
  #     position = [state[0], state[1], 0.02]
  #     orn = [0,0,0,1]#quaternion_from_euler(0,0,state[2])
  #     print(orn)
  #     pose = (position, orn)
  #     scn.add_cup_mark(pose, color='red')
      
  #     print('success')
    
  
      
      
  # ################################################################
  # # Camera
  # input('press enter to take photo.')
  # viewMatrix = p.computeViewMatrix(
  # cameraEyePosition=[1, -1, 1.6],
  # cameraTargetPosition=[0.5, 0, 0],
  # cameraUpVector=[0, 0, 1])
  
  # projectionMatrix = p.computeProjectionMatrixFOV(
  #               fov= 65,#math.atan(0.5) * (180/math.pi) * 2 #
  #               aspect= 1.0,#1.0,
  #               nearVal= 0.1,
  #               farVal=5)
  # width, height, rgbImg, depthImg, segImg = p.getCameraImage(
  #                                             width=1000, 
  #                                             height=1000,
  #                                             viewMatrix=viewMatrix,
  #                                             projectionMatrix=projectionMatrix, 
  #                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
  # cv2.cv2.imwrite('./image.jpg', cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR))

  # # image = p.getCameraImage(cam_width, cam_height, cam_view_matrix, cam_projection_matrix)[2][:, :, :3]
  # # cv2.cv2.imwrite('./image.png', image)

  # input('press enter to exit.')

        

  ################################################################

  # Heatmap

  input('press enter to draw heatmap')
  # generate 2 2d grids for the x & y bounds
  # x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(-1, 1, 100))
  # print(x)
  x = np.linspace(0,1,100)
  y = np.linspace(-1,1,100)
  print(x)
  z = np.zeros((len(x), len(y)))

  for i in range(len(x)):
    for j in range(len(y)):
      inputs = torch.tensor(np.array([x[i], y[j]]),dtype=torch.float32)
      model.eval()
      output = model(inputs).detach().numpy() #.round()
      print(output)
      z[i][j] = output

  z_min, z_max = 0, 1#-np.abs(z).max(), np.abs(z).max()    

  # x and y are bounds, so z should be the value *inside* those bounds.
  # Therefore, remove the last value from the z array.

  fig, ax = plt.subplots()

  # c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max, shading='auto') #
  c = ax.matshow(z, cmap='RdYlGn')
  ax.grid(False)
  # ax.axis('off')

  # 
  # ax.axis([x.min(), x.max(), y.min(), y.max()])
  cbar = fig.colorbar(c, ax=ax)
  cbar.ax.set_yticks([0, 0.5, 1])
  cbar.ax.tick_params(axis='y', which='major', grid_alpha=0, length=0, labelsize=30, colors='grey')

  # Ticks
  ax.set_xticks([0, 50, 100])
  ax.set_yticks([0, 50, 100])
  ax.set_xticklabels([-1, 0, 1])
  ax.set_yticklabels([0, 0.5, 1])
  ax.tick_params(axis='y', which='major', grid_alpha=0, length=0, labelsize=30, colors='grey')
  ax.tick_params(axis='x', which='major', grid_alpha=0, length=0, labelsize=30, colors='grey')

  # Labels
  # ax.set_ylabel('y (m)', fontsize = 20)
  # ax.set_xlabel('x (m)', fontsize = 20)
  plt.tight_layout()
  plt.savefig('./heatmap.pdf', bbox_inches="tight")

  plt.show()

  ################################################################

if __name__ == '__main__':
  main()