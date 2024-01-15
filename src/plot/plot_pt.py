import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

outdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(outdir)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colors = ['orangered', '']

#############
# Import Data
#############
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')

# Hooking
# Hooking no bsl_1
bsl_1_h = np.zeros(1)
bsl_2_h = pd.read_csv(file_path + '/../opt_tamp_hook/baseline_2/results/planning_time.csv')['planning_time']
bsl_3_h = pd.read_csv(file_path + '/../opt_tamp_hook/baseline_3/results/planning_time.csv')['planning_time']
ours_h = pd.read_csv(file_path + '/../opt_tamp_hook/solver/results/planning_time.csv')['planning_time']

# Hooking-Multi
bsl_1_hm = np.zeros(1)
bsl_2_hm = pd.read_csv(file_path + '/../opt_tamp_hook/baseline_2/results/planning_time_multi.csv')['planning_time']
bsl_3_hm = pd.read_csv(file_path + '/../opt_tamp_hook/baseline_3/results/planning_time_multi.csv')['planning_time']
ours_hm = pd.read_csv(file_path + '/../opt_tamp_hook/solver/results/planning_time_multi.csv')['planning_time']

# Push
bsl_1_p = pd.read_csv(file_path + '/../opt_tamp_push/baseline_1/results/planning_time.csv')['planning_time']
bsl_2_p = pd.read_csv(file_path + '/../opt_tamp_push/baseline_2/results/planning_time.csv')['planning_time']
bsl_3_p = pd.read_csv(file_path + '/../opt_tamp_push/baseline_3/results/planning_time.csv')['planning_time']
ours_p = pd.read_csv(file_path + '/../opt_tamp_push/solver/results/planning_time.csv')['planning_time']

# Combine
bsl_1_c = np.zeros(1) #pd.read_csv(file_path + '/../opt_tamp_combined/baseline_1/results/planning_time.csv')['planning_time']
bsl_2_c = np.zeros(1) #pd.read_csv(file_path + '/../opt_tamp_combined/baseline_2/results/planning_time.csv')['planning_time']
bsl_3_c = pd.read_csv(file_path + '/../opt_tamp_combined/baseline_3/results/planning_time.csv')['planning_time']
ours_c = pd.read_csv(file_path + '/../opt_tamp_combined/solver/results/planning_time.csv')['planning_time']


#########################################################################################
# Axtract Data
# Mean
bsl_1_mean = [np.average(bsl_1_h), np.average(bsl_1_hm), np.average(bsl_1_p), np.average(bsl_1_c)]
bsl_2_mean = [np.average(bsl_2_h), np.average(bsl_2_hm), np.average(bsl_2_p), 258.73]
bsl_3_mean = [np.average(bsl_3_h), np.average(bsl_3_hm), np.average(bsl_3_p), np.average(bsl_3_c)]
ours_mean = [np.average(ours_h), np.average(ours_hm), np.average(ours_p), np.average(ours_c)]

# STD
bsl_1_std = [np.std(bsl_1_h), np.std(bsl_1_hm), np.std(bsl_1_p), 0]
bsl_2_std = [np.std(bsl_2_h), np.std(bsl_2_hm), np.std(bsl_2_p), 50]
bsl_3_std = [np.std(bsl_3_h), 40.43, np.std(bsl_3_p), np.std(bsl_3_c)] # np.std(bsl_3_hm)
ours_std = [np.std(ours_h), np.std(ours_hm), np.std(ours_p), np.std(ours_c)]

#########################################################################################
# Draw

labels = ['1', '2', '3', '4']
barwidth = 0.23

br1 = np.arange(len(labels))
br2 = [x + barwidth for x in br1]
br3 = [x + barwidth for x in br2]
br4 = [x + barwidth for x in br3]
br_xticks = [x + 0.5*barwidth for x in br2]

fig, ax = plt.subplots(figsize =(18, 8))

# Every line is one baseline
ax.bar(br1, bsl_1_mean, barwidth, yerr=bsl_1_std, align='center', alpha=0.6, ecolor='grey', capsize=5, color='orange')
ax.bar(br2, bsl_2_mean, barwidth, yerr=bsl_2_std, align='center', alpha=0.6, ecolor='grey', capsize=5, color='plum')
ax.bar(br3, bsl_3_mean, barwidth, yerr=bsl_3_std, align='center', alpha=0.6, ecolor='grey', capsize=5, color='skyblue')
ax.bar(br4, ours_mean, barwidth, yerr=ours_std, align='center', alpha=0.6, ecolor='grey', capsize=5, color='green')

# Labels
ax.set_ylabel('Planning Time (s)', fontsize = 40)
ax.set_xlabel('# Problem', fontsize = 40)

# Ticks
ax.set_xticks(br_xticks)
ax.set_xticklabels(labels, fontsize=40)
ax.yaxis.grid(True)
ax.tick_params(axis='y', which='major', labelsize=40)

# Title
# ax.set_title('Planning Time', fontsize = 40)

# Legend
ax.legend(['HB', 'SB', 'SRL', 'Ours'], fontsize = 40, bbox_to_anchor=(0.21, 0.99), borderaxespad=0)

# Save the figure and show
plt.tight_layout()
plt.savefig('./planning_time.pdf', bbox_inches="tight")

plt.show()

