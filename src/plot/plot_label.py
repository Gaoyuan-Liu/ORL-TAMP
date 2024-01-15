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
bsl_2_h_p = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_2/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_3_h_p = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_3/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
ours_h_p = pd.read_csv(file_path + '/../opt-tamp-hook/solver/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_1_h_e = 0
bsl_2_h_e = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_2/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
bsl_3_h_e = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_3/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
ours_h_e = pd.read_csv(file_path + '/../opt-tamp-hook/solver/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp

# Hooking-Multi
bsl_1_hm_p = 0
bsl_2_hm_p = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_2/results/success_multi.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_3_hm_p = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_3/results/success_multi.csv')['plan_success'].to_numpy()[-1]/N_exp
ours_hm_p = pd.read_csv(file_path + '/../opt-tamp-hook/solver/results/success_multi.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_1_hm_e = 0
bsl_2_hm_e = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_2/results/success_multi.csv')['execute_success'].to_numpy()[-1]/N_exp
bsl_3_hm_e = pd.read_csv(file_path + '/../opt-tamp-hook/baseline_3/results/success_multi.csv')['execute_success'].to_numpy()[-1]/N_exp
ours_hm_e = pd.read_csv(file_path + '/../opt-tamp-hook/solver/results/success_multi.csv')['execute_success'].to_numpy()[-1]/N_exp

# Push
bsl_1_p_p = pd.read_csv(file_path + '/../opt-tamp-push/baseline_1/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_2_p_p = pd.read_csv(file_path + '/../opt-tamp-push/baseline_2/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_3_p_p = pd.read_csv(file_path + '/../opt-tamp-push/baseline_3/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
ours_p_p = pd.read_csv(file_path + '/../opt-tamp-push/solver/results/success.csv')['plan_success'].to_numpy()[-1]/N_exp
bsl_1_p_e = pd.read_csv(file_path + '/../opt-tamp-push/baseline_1/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
bsl_2_p_e = pd.read_csv(file_path + '/../opt-tamp-push/baseline_2/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
bsl_3_p_e = pd.read_csv(file_path + '/../opt-tamp-push/baseline_3/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp
ours_p_e = pd.read_csv(file_path + '/../opt-tamp-push/solver/results/success.csv')['execute_success'].to_numpy()[-1]/N_exp

# Combine



#########################################################################################
# Axtract Data

bsl_1_p = [bsl_1_h_p, bsl_1_hm_p, bsl_1_p_p]
bsl_2_p = [bsl_2_h_p, bsl_2_hm_p, bsl_2_p_p]
bsl_3_p = [bsl_3_h_p, bsl_3_hm_p, bsl_3_p_p]
ours_p = [ours_h_p, ours_hm_p, ours_p_p]

bsl_1_e = [bsl_1_h_e, bsl_1_hm_e, bsl_1_p_e]
bsl_2_e = [bsl_2_h_e, bsl_2_hm_e, bsl_2_p_e]
bsl_3_e = [bsl_3_h_e, bsl_3_hm_e, bsl_3_p_e]
ours_e = [ours_h_e, ours_hm_e, ours_p_e]

#########################################################################################


# Draw
labels = ['domain-1', 'domain-2', 'domain-3']
barwidth = 0.15

br1 = np.arange(len(labels))
br2 = [x + barwidth for x in br1]
br3 = [x + barwidth for x in br2]
br4 = [x + barwidth for x in br3]
br_xticks = [x + 0.5*barwidth for x in br2]



fig, ax = plt.subplots(figsize =(14, 8))

# Title
# ax.set_title('Executing Success Ratio', fontsize = 40)

# Legend
ax.legend(['Baseline-1', 'Baseline-2', 'Baseline-3', 'Ours'], fontsize = 40, bbox_to_anchor=(1, 1.0), borderaxespad=0)

# Save the figure and show
plt.tight_layout()
plt.savefig('./execution_success.pdf', bbox_inches="tight")

plt.show()


