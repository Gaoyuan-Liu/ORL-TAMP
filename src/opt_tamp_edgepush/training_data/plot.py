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

def normalize_dataframe(df_1, df_2):
    df = pd.concat([df_1, df_2], ignore_index=True)
    df_normalized = (df_1 - df_1.min()) / (df.max() - df_1.min())
    return df_normalized

# ----------------
# Import raw data
# ----------------
# SAC
data_sac = pd.read_csv('./sac/processed_data.csv')
length_center_sac = data_sac['length_center']
length_up_sac = data_sac['length_up']
length_low_sac = data_sac['length_low']
return_center_sac = data_sac['return_center']
return_up_sac = data_sac['return_up']
return_low_sac = data_sac['return_low']

#PPO
data_ppo = pd.read_csv('./ppo/processed_data.csv')
length_center_ppo = data_ppo['length_center']
length_up_ppo = data_ppo['length_up']
length_low_ppo = data_ppo['length_low']
return_center_ppo = data_ppo['return_center']
return_up_ppo = data_ppo['return_up']
return_low_ppo = data_ppo['return_low']


# print("\033[1;35m {}" .format(f'\n No. Data = {len(length_center_sac)}'))


# ----------------
# Cut
# ----------------
length = min(len(length_center_sac), len(length_center_ppo))
length_center_sac = length_center_sac.iloc[:length]
length_up_sac = length_up_sac.iloc[:length]
length_low_sac = length_low_sac.iloc[:length]
return_center_sac = return_center_sac.iloc[:length]
return_up_sac = return_up_sac.iloc[:length]
return_low_sac = return_low_sac.iloc[:length]

length_center_ppo = length_center_ppo.iloc[:length]
length_up_ppo = length_up_ppo.iloc[:length]
length_low_ppo = length_low_ppo.iloc[:length]
return_center_ppo = return_center_ppo.iloc[:length]
return_up_ppo = return_up_ppo.iloc[:length]
return_low_ppo = return_low_ppo.iloc[:length]



# --------------
# Define colors
# --------------
RebeccaPurple = (102/255, 51/255, 153/255)
OrangeRed = (255/255, 69/255, 0)
YellowGreen = (154/255, 205/255, 50/255)

# -------------------
# Global definiation
# -------------------
color_1 = 'darkgreen'
color_2 = 'darkorange'
linewidth_1 = 2
linewidth_2 = 2
fontsize = 24
titlename = "Bin Scenario"
window = 200
# figure(figsize=(8, 6), dpi=80)


# ---------------
# Episode Length
# ---------------


# filtered_length_center = moving_average_filter(length_center, window)
# filtered_length_up = moving_average_filter(length_up, window)
# filtered_length_low = moving_average_filter(length_low, window)

# ax = filtered_length_center.reset_index().plot(x='index', y='length_center', kind='line',
#                         linewidth=linewidth_1,
#                         fontsize=fontsize,
#                         xticks=np.arange(0, (len(filtered_length_center)//100)*100 + 400, step=400),
#                         color=color_1)

# plt.fill_between(filtered_length_center.index, filtered_length_up, filtered_length_low, alpha=0.2, color=color_1, edgecolor='none')

# plt.xlabel('# Episode', fontsize=fontsize)
# plt.ylabel('Episode Length', fontsize=fontsize)
# ax.get_legend().remove()
# plt.grid(color = 'green', linestyle = '--', linewidth = 1)
# plt.savefig("el_retrieve.pdf", bbox_inches="tight")
# plt.show()


# ---------------
# Episode Return
# ---------------
# SAC
return_center_sac = normalize_dataframe(moving_average_filter(return_center_sac, window), moving_average_filter(return_center_ppo, window))
return_up_sac = normalize_dataframe(moving_average_filter(return_up_sac, window), moving_average_filter(return_up_ppo, window))
return_low_sac = normalize_dataframe(moving_average_filter(return_low_sac, window), moving_average_filter(return_low_ppo, window))

# PPO
return_center_ppo = normalize_dataframe(moving_average_filter(return_center_ppo, window), moving_average_filter(return_center_sac, window))
return_up_ppo = normalize_dataframe(moving_average_filter(return_up_ppo, window), moving_average_filter(return_up_sac, window))
return_low_ppo = normalize_dataframe(moving_average_filter(return_low_ppo, window), moving_average_filter(return_low_sac, window))




# ax = normalized_return_center.reset_index().plot(x='index', y='return_center', kind='line',
#                         linewidth=linewidth_1,
#                         fontsize=fontsize,
#                         xticks=np.arange(0, (len(normalized_return_center)//100)*100 + 100, step=400),
#                         color=color_1)

plt.plot(return_center_sac.index, return_center_sac, linewidth=linewidth_1, color=color_1)
plt.plot(return_center_ppo.index, return_center_ppo, linewidth=linewidth_1, color=color_2)

plt.legend(['SAC', 'PPO'], loc='upper left', bbox_to_anchor=(-0, 1),fontsize=fontsize, borderaxespad=0)
plt.fill_between(return_center_sac.index, return_up_sac, return_low_sac, alpha=0.2, color=color_1, edgecolor='none')
plt.fill_between(return_center_ppo.index, return_up_ppo, return_low_ppo, alpha=0.2, color=color_2, edgecolor='none')

plt.xlabel('# Episode', fontsize=fontsize)
plt.ylabel('Episode Return', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# ax.get_legend().remove()

plt.grid(color = 'lightgrey', linestyle = '--', linewidth = 1)
# plt.suptitle(titlename, fontsize=fontsize)
plt.savefig("er_edgepush.pdf", bbox_inches="tight")
plt.show()


