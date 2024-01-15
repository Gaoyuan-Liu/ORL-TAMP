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
data = pd.read_csv('training_data.csv')
data_episode_length = data['episode_length']
data_episode_return = data['episodic_return']
print("\033[1;35m {}" .format(f'\n No. Data = {len(data_episode_length)}'))

# --------------
# Define colors
# --------------
RebeccaPurple = (102/255, 51/255, 153/255)
OrangeRed = (255/255, 69/255, 0)
YellowGreen = (154/255, 205/255, 50/255)

# -------------------
# Global definiation
# -------------------
color_1 = 'blue'
color_2 = 'green'
linewidth_1 = 2
linewidth_2 = 2
fontsize = 32
titlename = "Bin Scenario"
window = 200
# figure(figsize=(8, 6), dpi=80)


# ---------------
# Episode Length
# ---------------


filtered_data = moving_average_filter(data_episode_length, window)

ax = filtered_data.reset_index().plot(x='index', y='episode_length', kind='line',
                        linewidth=linewidth_1,
                        fontsize=fontsize,
                        xticks=np.arange(0, (len(filtered_data)//100)*100 + 400, step=400),
                        color=color_1)


plt.xlabel('# Episode', fontsize=fontsize)
plt.ylabel('Episode Length', fontsize=fontsize)
ax.get_legend().remove()
plt.savefig("learning_el.pdf", bbox_inches="tight")
plt.show()


# ---------------
# Episode Return
# ---------------
filtered_data = moving_average_filter(data_episode_return, window)
normalized_data = normalize_dataframe(filtered_data)

ax = normalized_data.reset_index().plot(x='index', y='episodic_return', kind='line',
                        linewidth=linewidth_1,
                        fontsize=fontsize,
                        xticks=np.arange(0, (len(filtered_data)//100)*100 + 100, step=400),
                        color=color_1)

plt.xlabel('# Episode', fontsize=fontsize)
plt.ylabel('Episode Return', fontsize=fontsize)
ax.get_legend().remove()
# plt.legend(['Without safety layer', 'With safety layer'], bbox_to_anchor=(1, 1.0),fontsize=fontsize, borderaxespad=0)

# plt.suptitle(titlename, fontsize=fontsize)
plt.savefig("learning_er.pdf", bbox_inches="tight")
plt.show()


