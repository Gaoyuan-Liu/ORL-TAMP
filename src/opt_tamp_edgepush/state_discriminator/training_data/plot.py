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


def moving_average_filter(dataframe, window_size):
    # Create a rolling window object
    rolling_window = dataframe.rolling(window=window_size)
    
    # Calculate the mean for each window
    moving_average = rolling_window.mean()
    
    return moving_average

def normalize_dataframe(df):
    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized

# ----------------
# Import raw data
# ----------------
data = pd.read_csv('processed_data.csv')
loss_center = data['loss_center']
loss_up = data['loss_up']
loss_low = data['loss_low']
accu_center = data['accu_center']
accu_up = data['accu_up']
accu_low = data['accu_low']

print("\033[1;35m {}" .format(f'\n No. Data = {len(loss_center)}'))

# --------------
# Define colors
# --------------
RebeccaPurple = (102/255, 51/255, 153/255)
OrangeRed = (255/255, 69/255, 0)
YellowGreen = (154/255, 205/255, 50/255)

# -------------------
# Global definiation
# -------------------
color_1 = 'teal'
color_2 = 'green'
linewidth_1 = 2
linewidth_2 = 2
fontsize = 32
titlename = "Bin Scenario"
window = 10 #200
# figure(figsize=(8, 6), dpi=80)


# ---------------
# Episode Length
# ---------------


filtered_loss_center = moving_average_filter(loss_center, window)
filtered_loss_up = moving_average_filter(loss_up, window)
filtered_loss_low = moving_average_filter(loss_low, window)

ax = filtered_loss_center.reset_index().plot(x='index', y='loss_center', kind='line',
                        linewidth=linewidth_1,
                        fontsize=fontsize,
                        xticks=np.arange(0, (len(filtered_loss_center)//100)*100 + 400, step=100),
                        color=color_1)

plt.fill_between(filtered_loss_center.index, filtered_loss_up, filtered_loss_low, alpha=0.2, color=color_1, edgecolor='none')

plt.xlabel('# Epoch', fontsize=fontsize)
plt.ylabel('Loss', fontsize=fontsize)
ax.get_legend().remove()
plt.grid(color = 'lightgrey', linestyle = '--', linewidth = 1)
plt.savefig("loss_edgepushing.pdf", bbox_inches="tight")
plt.show()


# ---------------
# Episode Return
# ---------------
filtered_accu_center = moving_average_filter(accu_center, window)
filtered_accu_up = moving_average_filter(accu_up, window)
filtered_accu_low = moving_average_filter(accu_low, window)
normalized_accu_center = normalize_dataframe(filtered_accu_center)
normalized_accu_up = normalize_dataframe(filtered_accu_up)
normalized_accu_low = normalize_dataframe(filtered_accu_low)

ax = normalized_accu_center.reset_index().plot(x='index', y='accu_center', kind='line',
                        linewidth=linewidth_1,
                        fontsize=fontsize,
                        xticks=np.arange(0, (len(normalized_accu_center)//100)*100 + 100, step=100),
                        color=color_1)

plt.fill_between(normalized_accu_center.index, normalized_accu_up, normalized_accu_low, alpha=0.2, color=color_1, edgecolor='none')

plt.xlabel('# Epoch', fontsize=fontsize)
plt.ylabel('Accuracy', fontsize=fontsize)
ax.get_legend().remove()
# plt.legend(['Without safety layer', 'With safety layer'], bbox_to_anchor=(1, 1.0),fontsize=fontsize, borderaxespad=0)

# plt.suptitle(titlename, fontsize=fontsize)
plt.grid(color = 'lightgrey', linestyle = '--', linewidth = 1)
plt.savefig("accu_edgepushing.pdf", bbox_inches="tight")
plt.show()


