import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os


import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


import copy
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
from torch.nn import functional as F


import torchvision
import torchvision.transforms as transforms

from sklearn.preprocessing import StandardScaler

# ----------------------------
# Build NN model
class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(32 * 12 * 16, 64)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(64, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self,x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.pool2(x)
    x = x.view(-1, 32 * 12 * 16)
    x = self.fc1(x)
    x = self.relu3(x)
    x = self.fc2(x)
    x = self.sigmoid(x)

    return x


# ----------------------------
# Build dataset
# class Dataset(Dataset):
#   def __init__(self, X, y):
#     self.X = X
#     self.y = y

#   def __len__(self):
#     return len(self.X)

#   def __getitem__(self, idx):
#     return self.X[idx], self.y[idx]




def main():
  # Data
  loaded_data = np.load('./data/data_image.npz')
  X = loaded_data['features']
  y = loaded_data['labels']

  print(X.shape)
  print(y.shape)
  input()
  

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

  X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

  # Create a DataLoader for training and testing sets
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
  batch_size = 10
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
  # # Initialize the CNN model, loss function, and optimizer
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  net = CNN().to(device)
  net_predict = CNN().to(device) # For accu
  criterion = nn.BCELoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)

  # Train the network
  num_epochs = 500
  net.to(device)

  output_data = pd.DataFrame(columns=['epochs', 'loss', 'accu'])

  for epoch in range(num_epochs):

    # Loss
    net.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels.view(-1, 1))
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # ----------------------------
    # Accu
    net_predict.eval()
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for predict_param, param in zip(net_predict.parameters(), net.parameters()):
          predict_param.data.copy_(param.data)
      for inputs, labels in train_loader:
          outputs = net_predict(inputs)#.detach()
          predicted = (outputs > 0.5).float() #model(torch.tensor(x,dtype=torch.float32)).detach()
          total += labels.size(0)


          print(f'predicted: {predicted.reshape(-1)}')
          print(f'labels: {labels}')
          print('\n')
          correct += (predicted.reshape(-1) == labels).sum().item()
          print(f'total: {total}')
          print(f'correct: {correct}')
    
    accu = correct / total
    print(f"Epoch {epoch + 1}, Accu: {accu}")

    # Save loss and accu
    output_data.loc[len(output_data.index)] = [epoch, loss.detach().numpy(), accu]
    file_path = os.path.dirname(os.path.realpath(__file__))
    output_data.to_csv(file_path+'/training_data/training_data.csv', index=False)

  # Save model (net)
  name = 'state_discriminator_conv'
  torch.save(net.state_dict(), './%s.pth' % name)




if __name__ == '__main__':
    main()