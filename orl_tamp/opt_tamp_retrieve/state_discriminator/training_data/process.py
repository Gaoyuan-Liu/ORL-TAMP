import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Locate local working directory on where this file is
outdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(outdir)

processed_data = pd.df = pd.DataFrame(columns=['loss_up', 'loss_low', 'loss_center', 'accu_up', 'accu_low', 'accu_center'])

N_run = 3
# ----------------
# Import raw data
# ----------------
data = []
length = np.inf
for i in range(N_run):
    data.append(pd.read_csv('training_data_'+str(i+1)+'.csv'))
    length = min(length, len(data[i]))

# ----------------
# Cut
# ----------------
for i in range(N_run):
    data[i] = data[i].iloc[:length]

# ----------------
# Abstract
# ----------------
loss_up = []
loss_down = []
loss_center = []
accu_up = []
accu_down = []
accu_center = []

for i in range(length):
    loss_row = []
    accu_row = []
    for j in range(N_run):
        loss_row.append(data[j]['loss'][i])
        accu_row.append(data[j]['accu'][i])

    loss_up.append(max(loss_row))
    loss_down.append(min(loss_row))
    loss_center.append(np.mean(loss_row))
    accu_up.append(max(accu_row))
    accu_down.append(min(accu_row))
    accu_center.append(np.mean(accu_row))


# ----------------
# Save
# ----------------
processed_data['loss_up'] = loss_up
processed_data['loss_low'] = loss_down
processed_data['loss_center'] = loss_center
processed_data['accu_up'] = accu_up
processed_data['accu_low'] = accu_down
processed_data['accu_center'] = accu_center

processed_data.to_csv('processed_data.csv', index=False)

    
