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

processed_data = pd.df = pd.DataFrame(columns=['length_up', 'length_low', 'length_center', 'return_up', 'return_low', 'return_center'])

N_run = 3
# ----------------
# Import raw data
# ----------------
data = []
length = np.inf
for i in range(N_run):
    data.append(pd.read_csv('training_data_sac_'+str(i+1)+'.csv'))
    length = min(length, len(data[i]))

# ----------------
# Cut
# ----------------
for i in range(N_run):
    data[i] = data[i].iloc[:length]

# ----------------
# Abstract
# ----------------
length_up = []
length_down = []
length_center = []
return_up = []
return_down = []
return_center = []

for i in range(length):
    length_row = []
    return_row = []
    for j in range(N_run):
        length_row.append(data[j]['episode_length'][i])
        return_row.append(data[j]['episode_return'][i])

    length_up.append(max(length_row))
    length_down.append(min(length_row))
    length_center.append(np.mean(length_row))
    return_up.append(max(return_row))
    return_down.append(min(return_row))
    return_center.append(np.mean(return_row))


# ----------------
# Save
# ----------------
processed_data['length_up'] = length_up
processed_data['length_low'] = length_down
processed_data['length_center'] = length_center
processed_data['return_up'] = return_up
processed_data['return_low'] = return_down
processed_data['return_center'] = return_center

processed_data.to_csv('processed_data.csv', index=False)

    
