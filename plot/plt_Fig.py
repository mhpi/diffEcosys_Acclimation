import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
matplotlib.use('Qt5Agg')

df_out = pd.read_csv(f"../results/Global/csv/global_comb_models.csv")

#=======================================================================================================================

color_list     = ["#005F73","#0A9396","#EE9B00","#CA6702","#BB3E03"] #"#001219",
label_list     = [r'δ$_{\rm psn1}$', r'δ$_{\rm psn2}$', r'δ$_{\rm psn3}$', r'δ$_{\rm psn4}$']
output_path    = "../results/Global/plots"
df_out = df_out.set_index('Cat').transpose()
df_out = df_out.reset_index(drop = True)
df_out = df_out[['PFT','PFT+LNC', 'PFT+LNC+ENV']]

# Number of groups (one for each row in the DataFrame)
n_rows = df_out.shape[0]
# Number of categories (one for each column in the DataFrame)
n_categories = df_out.shape[1]

# Set the positions of the bars and the width
positions = np.arange(n_categories)  # position of groups
width = 0.1 # reduced width of each bar to increase space between groups

# Offset calculation to increase the gap between groups
group_width = width * n_rows
offsets = np.linspace(-group_width / 2, group_width / 2, n_rows)

fig, ax = plt.subplots(figsize=(8, 6))

# Plotting bars for each category in each group
for i in range(n_rows):
    bar_positions = positions + offsets[i]
    ax.bar(bar_positions, df_out.iloc[i], width, label=label_list[i], color = color_list[i])

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel(r"A$_{\rm N}$ ($\mu$mol m$^{-2}$ s$^{-1})$", fontsize = 14)
ax.set_xticks(positions)
ax.set_xticklabels(df_out.columns,fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(ncol=6, fontsize=14, loc='upper center', bbox_to_anchor=(0.48, -0.05))
ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

ax.set_ylim(8.0,10.0)
ax.set_yticks(np.linspace(8, 10, 9))
plt.tight_layout()
plt.subplots_adjust(left=0.13, right=0.95, bottom=0.14, top=0.98)
plt.savefig(os.path.join(output_path, f"AN_bar_global_2011_2020.png"))

