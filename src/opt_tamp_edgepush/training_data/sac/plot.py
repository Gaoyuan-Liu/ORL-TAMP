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
length_center = data['length_center']
length_up = data['length_up']
length_low = data['length_low']
return_center = data['return_center']
return_up = data['return_up']
return_low = data['return_low']

print("\033[1;35m {}" .format(f'\n No. Data = {len(length_center)}'))

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
color_2 = 'green'
linewidth_1 = 2
linewidth_2 = 2
fontsize = 32
titlename = "Bin Scenario"
window = 200
step = 5000#int(len(data)/4)

# figure(figsize=(8, 6), dpi=80)


# ---------------
# Episode Length
# ---------------


filtered_length_center = moving_average_filter(length_center, window)
filtered_length_up = moving_average_filter(length_up, window)
filtered_length_low = moving_average_filter(length_low, window)

ax = filtered_length_center.reset_index().plot(x='index', y='length_center', kind='line',
                        linewidth=linewidth_1,
                        fontsize=fontsize,
                        xticks=np.arange(0, (len(filtered_length_center)//100)*100 + 400, step=step),
                        color=color_1)

plt.fill_between(filtered_length_center.index, filtered_length_up, filtered_length_low, alpha=0.2, color=color_1, edgecolor='none')

plt.xlabel('# Episode', fontsize=fontsize)
plt.ylabel('Episode Length', fontsize=fontsize)
ax.get_legend().remove()
plt.grid(color = 'green', linestyle = '--', linewidth = 1)
plt.savefig("el_edgepush.pdf", bbox_inches="tight")
plt.show()


# ---------------
# Episode Return
# ---------------
filtered_return_center = moving_average_filter(return_center, window)
filtered_return_up = moving_average_filter(return_up, window)
filtered_return_low = moving_average_filter(return_low, window)
normalized_return_center = normalize_dataframe(filtered_return_center)
normalized_return_up = normalize_dataframe(filtered_return_up)
normalized_return_low = normalize_dataframe(filtered_return_low)

ax = normalized_return_center.reset_index().plot(x='index', y='return_center', kind='line',
                        linewidth=linewidth_1,
                        fontsize=fontsize,
                        xticks=np.arange(0, (len(normalized_return_center)//100)*100 + 100, step=step),
                        color=color_1)

plt.fill_between(normalized_return_center.index, normalized_return_up, normalized_return_low, alpha=0.2, color=color_1, edgecolor='none')

plt.xlabel('# Episode', fontsize=fontsize)
plt.ylabel('Episode Return', fontsize=fontsize)
ax.get_legend().remove()
# plt.legend(['Without safety layer', 'With safety layer'], bbox_to_anchor=(1, 1.0),fontsize=fontsize, borderaxespad=0)

# plt.suptitle(titlename, fontsize=fontsize)
plt.grid(color = 'green', linestyle = '--', linewidth = 1)
plt.savefig("er_edgepush.pdf", bbox_inches="tight")
plt.show()


