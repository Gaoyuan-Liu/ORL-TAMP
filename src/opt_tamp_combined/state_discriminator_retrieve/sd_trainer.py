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
from utils import save_list

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


def main():

    # Import data
    df = pd.read_csv('data_cube.csv')
    df = df.drop('x_h', axis=1)
    df = df.drop('y_h', axis=1)
    df = df.drop('yaw_h', axis=1)

    # print(df.head())

    # Data preprocessing
    df_train = df.sample(frac=0.7, random_state=0)
    df_valid = df.drop(df_train.index)

    X_train = np.array(df_train.drop('success', axis=1), dtype=np.float32)
    X_valid = np.array(df_valid.drop('success', axis=1), dtype=np.float32)
    y_train = np.array(df_train['success'], dtype=np.float32)
    y_valid = np.array(df_valid['success'], dtype=np.float32)

    trainset = dataset(X_train,y_train)

    # DataLoader
    trainloader = DataLoader(trainset, batch_size=20,shuffle=False)
    X_valid = torch.tensor(X_valid,dtype=torch.float32)
    y_valid = torch.tensor(y_valid,dtype=torch.float32)

    # Model

    # Training
    # hyper parameters
    learning_rate = 0.01
    epochs = 10000
    # Model , Optimizer, Loss
    model = Net(input_shape=X_train.shape[1])
    predict_model = Net(input_shape=X_train.shape[1])

    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    loss_fn = nn.BCELoss()


    losses = []
    accur = []
    for i in range(epochs):
        for j,(x_train,y_train) in enumerate(trainloader):

            #calculate output
            output = model(x_train)

            #calculate loss
            loss = loss_fn(output,y_train.reshape(-1,1))


            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #accuracy
            for predict_param, param in zip(predict_model.parameters(), model.parameters()):
                predict_param.data.copy_(param.data)
                predicted = predict_model(X_valid).detach() #model(torch.tensor(x,dtype=torch.float32)).detach()
                acc = (predicted.reshape(-1).numpy().round() == y_valid.numpy()).mean()

        if i%50 == 0:
            losses.append(loss.detach())
            accur.append(acc)
            print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))

    name = 'state_discriminator_cube'
    torch.save(model.state_dict(), './%s.pth' % name)


    save_list(losses, 'losses')
    save_list(accur, 'accur')









    

if __name__ == '__main__':
    main()