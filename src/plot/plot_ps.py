import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


outdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(outdir)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

####################################################################
# Import Data
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')

N_exp = 50

# Hooking
bsl_1_h_p = 0
bsl_2_h_p = pd.read_csv(file_path + '/../opt_tamp_hook/baseline_2/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_3_h_p = pd.read_csv(file_path + '/../opt_tamp_hook/baseline_3/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
ours_h_p = pd.read_csv(file_path + '/../opt_tamp_hook/solver/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
# bsl_1_h_e = 0
# bsl_2_h_e = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_2/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
# bsl_3_h_e = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_3/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
# ours_h_e = pd.read_csv(file_path + '/../opt-tamp-hook/solver/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp

# Hooking-Multi
bsl_1_hm_p = 0
bsl_2_hm_p = pd.read_csv(file_path + '/../opt_tamp_hook/baseline_2/results/success_multi.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_3_hm_p = pd.read_csv(file_path + '/../opt_tamp_hook/baseline_3/results/success_multi.csv')['plan_success'].to_numpy()[-1]/N_exp
ours_hm_p = pd.read_csv(file_path + '/../opt_tamp_hook/solver/results/success_multi.csv')['plan_success'].to_numpy()[-1]/N_exp
# bsl_1_hm_e = 0
# bsl_2_hm_e = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_2/results/success_multi.csv')['execute_success'].to_numpy()[-1]/N_exp
# bsl_3_hm_e = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_3/results/success_multi.csv')['execute_success'].to_numpy()[-1]/N_exp
# ours_hm_e = pd.read_csv(file_path + '/../opt-tamp-hook/solver/results/success_multi.csv')['execute_success'].to_numpy()[-1]/N_exp

# Push
bsl_1_p_p = pd.read_csv(file_path + '/../opt_tamp_push/baseline_1/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_2_p_p = pd.read_csv(file_path + '/../opt_tamp_push/baseline_2/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_3_p_p = pd.read_csv(file_path + '/../opt_tamp_push/baseline_3/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
ours_p_p = pd.read_csv(file_path + '/../opt_tamp_push/solver/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
# bsl_1_p_e = pd.read_csv(file_path + '/../opt-tamp-push/baseline_1/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
# bsl_2_p_e = pd.read_csv(file_path + '/../opt-tamp-push/baseline_2/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
# bsl_3_p_e = pd.read_csv(file_path + '/../opt-tamp-push/baseline_3/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
# ours_p_e = pd.read_csv(file_path + '/../opt-tamp-push/solver/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp

# Combine
bsl_1_c_p = 0 #pd.read_csv(file_path + '/../opt_tamp_combine/baseline_1/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_2_c_p = 5/N_exp #pd.read_csv(file_path + '/../opt_tamp_combine/baseline_2/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_3_c_p = 25/N_exp #pd.read_csv(file_path + '/../opt_tamp_combine/baseline_3/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
ours_c_p = 47/N_exp #pd.read_csv(file_path + '/../opt_tamp_combine/solver/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
# bsl_1_c_e = pd.read_csv(file_path + '/../opt-tamp-combine/baseline_1/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
# bsl_2_c_e = pd.read_csv(file_path + '/../opt-tamp-combine/baseline_2/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
# bsl_3_c_e = pd.read_csv(file_path + '/../opt-tamp-combine/baseline_3/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
# ours_c_e = pd.read_csv(file_path + '/../opt-tamp-combine/solver/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp



#########################################################################################
# Axtract Data

bsl_1_p = np.array([bsl_1_h_p, bsl_1_hm_p, bsl_1_p_p, bsl_1_c_p]) * 100
bsl_2_p = np.array([bsl_2_h_p, bsl_2_hm_p, bsl_2_p_p, bsl_2_c_p]) * 100
bsl_3_p = np.array([bsl_3_h_p, bsl_3_hm_p, bsl_3_p_p, bsl_3_c_p]) * 100
ours_p = np.array([ours_h_p, ours_hm_p, ours_p_p, ours_c_p]) * 100

# bsl_1_e = [bsl_1_h_e, bsl_1_hm_e, bsl_1_p_e]
# bsl_2_e = [bsl_2_h_e, bsl_2_hm_e, bsl_2_p_e]
# bsl_3_e = [bsl_3_h_e, bsl_3_hm_e, bsl_3_p_e]
# ours_e = [ours_h_e, ours_hm_e, ours_p_e]

#########################################################################################


# Draw
labels = ['1', '2', '3', '4']
barwidth = 0.18

br1 = np.arange(len(labels))
br2 = [x + barwidth for x in br1]
br3 = [x + barwidth for x in br2]
br4 = [x + barwidth for x in br3]
br_xticks = [x + 0.5*barwidth for x in br2]

fig, ax = plt.subplots(figsize =(18, 8))

ax.bar(br1, bsl_1_p, align='center', width = barwidth, alpha=0.6, capsize=5, color='orange')
ax.bar(br2, bsl_2_p, align='center', width = barwidth, alpha=0.6, capsize=5, color='plum')
ax.bar(br3, bsl_3_p, align='center', width = barwidth, alpha=0.6, capsize=5, color='skyblue')
ax.bar(br4, ours_p, align='center', width = barwidth, alpha=0.6, capsize=5, color='green')


# Labels
ax.set_ylabel('Success Ratio (%)', fontsize = 40)
ax.set_xlabel('# Problem', fontsize = 40)

# Ticks
ax.set_xticks(br_xticks)
ax.set_xticklabels(labels, fontsize=40)
ax.yaxis.grid(True)
ax.tick_params(axis='y', which='major', labelsize=40)

# Title
# ax.set_title('Planning Success Ratio', fontsize = 40)

# Legend
ax.legend(['HB', 'SB', 'SRL', 'Ours'], fontsize = 40, loc=1, framealpha=0.5, bbox_to_anchor=[0.93, 1])

# Save the figure and show
plt.tight_layout()
plt.savefig('./planning_success.pdf', bbox_inches="tight")

plt.show()


