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
  scn = Scenario()
  scn.reset()



  # ----------------

  # Data
  df = pd.read_csv('data.csv')
  data = np.array(df, dtype=np.float32)

  print(data.shape)



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





      

  
  for i in range(data.shape[0]):
      

    state = data[i][:2]

    if data[i][-1] == 1:
      position = [state[0], state[1], 0.02]
      orn = [0,0,0,1]#quaternion_from_euler(0,0,state[2])
      pose = (position, orn)
      scn.add_cup_mark(pose, color='green')
    else:
      position = [state[0], state[1], 0.02]
      orn = [0,0,0,1]#quaternion_from_euler(0,0,state[2])
      print(orn)
      pose = (position, orn)
      scn.add_cup_mark(pose, color='red')
      
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