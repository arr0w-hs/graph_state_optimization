#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:35:16 2024

@author: hsharma4
plotting er graphs from data stored in a folder in form of pickles
"""


import sys
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

"""the location of folder with pickle files"""
data_dir = os.path.join(dir_name, "er_results_data/data")
plot_dir = os.path.join(dir_name, "er_results_data/plots")
df_list = []

def roundoff(ele):
    """function to round off float"""
    return int(ele*100)/100

for root, _, files in os.walk(data_dir):
    for file in files:
        if file == '.DS_Store':
            continue
        with open(os.path.join(data_dir, file), 'rb') as f:
            data_dict_loaded = pickle.load(f)
            f.close()

        out_df = pd.DataFrame(data_dict_loaded["out_dict"])
        df_list.append(out_df)

"""the dataframe with runtimes and number of edges"""
out_df = pd.concat(df_list, axis=0, join='outer', ignore_index=True)
out_df["prob_val"] = out_df["prob_val"].apply(roundoff)
#print(out_df)

wide = out_df.groupby(['prob_val'],as_index=False).mean()
std = out_df.groupby(['prob_val'],as_index=False).std()
max_df = out_df.groupby(['prob_val'],as_index=False).max()
min_df = out_df.groupby(['prob_val'],as_index=False).min()

print(out_df.groupby(['prob_val'],as_index=False).count())


"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
plt.plot(wide['prob_val'], wide['edge'], label='Initial edges')#, marker = 'o')
plt.fill_between(wide['prob_val'],
                 max_df['edge'],
                 y2=min_df['edge'], alpha=0.2)

plt.plot(wide['prob_val'], wide['sa_edge'], label = "SA")#, marker = 'o')
plt.fill_between(wide['prob_val'],
                 max_df['sa_edge'],
                 y2=min_df['sa_edge'], alpha=0.2)

plt.plot(wide['prob_val'], wide['sa_ilp_edge'], label = "SA+ILP")#, marker = 'o')
plt.fill_between(wide['prob_val'],
                 max_df['sa_ilp_edge'],
                 y2=min_df['sa_ilp_edge'], alpha=0.2)
plt.xlabel("Probability")
plt.ylabel("Edges")
plt.xticks(np.arange(0, 1, step=0.1))
plt.tick_params(direction='out', length=6, width=200,
                grid_alpha=0.5 , pad = 1)
plt.legend()
plt.savefig(plot_dir + "/edge_with_band" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
plt.savefig(plot_dir + "/edge_with_band" + ".png", dpi=800, format="png", bbox_inches = 'tight')



"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
out_df = out_df.rename(columns={"rt_sailp": "SA+ILP", "rt_ilp": "ILP"})
out_df = out_df.drop(columns = ("ILP"), axis = 1)
rt_df = pd.melt(out_df, id_vars=['prob_val'], value_vars=['SA+ILP'],
                var_name='Method', value_name='runtime')
#rt_df = rt_df.drop(["ILP"])
plt.figure()
sns.set_theme(style = "whitegrid")
ax = sns.catplot(data = rt_df, x="prob_val", y="runtime",
            hue = "Method", kind = "strip", alpha = 0.75,
            s = 10, legend=False, height=5, aspect=1.5)
sns.pointplot(
    data=rt_df, x="prob_val", y="runtime", hue="Method",
    dodge=0, errorbar=None)
plt.yscale("log")
plt.xlabel("Probability")
plt.ylabel("Runtime (seconds)")
plt.tick_params(direction='out', length=6, width=200,
                grid_alpha=0.5 , pad = 1)#grid_color='r'
plt.tight_layout()
plt.savefig(plot_dir  + "/catplot_withline" + ".svg",dpi=800, format="svg", bbox_inches = 'tight')
plt.savefig(plot_dir  + "/catplot_withline" + ".png",dpi=800, format="png", bbox_inches = 'tight')

rt_array = out_df[["SA+ILP"]].to_numpy().flatten()
rt_array = out_df[["sa_edge"]].to_numpy().flatten()
#rt_array = np.log(rt_array)
sa_array = out_df[["sa_ilp_edge"]].to_numpy().flatten()

#print(np.corrcoef(rt_array,sa_array))

