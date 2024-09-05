#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:08:53 2024

@author: hsharma4
plotting bd graph results from data stored in a folder in form of pickles
"""

import sys
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

data_dir = os.path.join(dir_name, "bd_results_data/data")
plot_dir = os.path.join(dir_name, "bd_results_data/plots")

df_list = []
for root, _, files in os.walk(data_dir):
    for file in files:
        if file == '.DS_Store':
            continue
        with open(os.path.join(data_dir, file), 'rb') as f:
            data_dict_loaded = pickle.load(f)
            f.close()

        out_df = pd.DataFrame(data_dict_loaded["out_dict"])
        df_list.append(out_df)

out_df = pd.concat(df_list, axis=0, join='outer', ignore_index=True)
#print(out_df)
wide = out_df.groupby(['vertex_num'],as_index=False).mean()
std = out_df.groupby(['vertex_num'],as_index=False).std()
max_df = out_df.groupby(['vertex_num'],as_index=False).max()
min_df = out_df.groupby(['vertex_num'],as_index=False).min()
#print(wide)

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""

plt.figure()
plt.plot(wide['vertex_num'], wide['edge'], label='Initial edges')
plt.fill_between(wide['vertex_num'],
                 max_df['edge'],
                 y2=min_df['edge'], alpha=0.2)

plt.plot(wide['vertex_num'], wide['sa_edge'], label = "SA")
plt.fill_between(wide['vertex_num'],
                 max_df['sa_edge'],
                 y2=min_df['sa_edge'], alpha=0.2)

plt.plot(wide['vertex_num'], wide['sa_ilp_edge'], label = "SA+ILP")
plt.fill_between(wide['vertex_num'],
                 max_df['sa_ilp_edge'] ,
                 y2=min_df['sa_ilp_edge'], alpha=0.2)
plt.xlabel("Number of vertices")
plt.ylabel("Edges")
plt.legend()
#plt.savefig(plot_dir + "/edge_with_band" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
#plt.savefig(plot_dir + "/edge_with_band" + ".png", dpi=800, format="png", bbox_inches = 'tight')

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
out_df = out_df.rename(columns={"rt_sailp": "SA+ILP", "rt_ilp": "ILP"})
rt_df = pd.melt(out_df, id_vars=['vertex_num'], value_vars=['SA+ILP', 'ILP'],
                var_name='Method', value_name='runtime')
plt.figure()
sns.set_theme(style = "whitegrid")
ax = sns.catplot(data = rt_df, x="vertex_num", y="runtime",
            hue = "Method", kind = "strip", alpha = 0.75,
            s = 10, legend=False, height=4, aspect=1.5)
sns.pointplot(
    data=rt_df, x="vertex_num", y="runtime", hue="Method",
    dodge=0, errorbar=None)
plt.yscale("log")
plt.xlabel("Number of vertices")
plt.ylabel("Runtime (seconds)")
plt.tight_layout()
#plt.savefig(plot_dir + "/catplot_withline" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
#plt.savefig(plot_dir + "/catplot_withline" + ".png", dpi=800, format="png", bbox_inches = 'tight')


rt_array = out_df[["ILP"]].to_numpy().flatten()
#rt_array = out_df[["sa_edge"]].to_numpy().flatten()
#rt_array = np.log(rt_array)
sa_array = out_df[["edge"]].to_numpy().flatten()
#print(rt_array)
#print(sa_array)
print(np.corrcoef(rt_array,sa_array))
