#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:35:16 2024

@author: hsharma4
plotting er graphs
"""


import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

data_dir = os.path.join(dir_name, "er_results_data/data")
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
        #print(data_dict_loaded["out_dict"].keys())

out_df = pd.concat(df_list, axis=0, join='outer', ignore_index=True)


#data_location = '/2024-07-13_sa_ilp/225850'
#data_location = '/2024-07-13_sa_ilp/225049'
#data_location = '/2024-07-13_sa_ilp/175625'

#data_location = '/2024-07-13_sa_ilp/154120'
#data_location = '/2024-07-12_sa_ilp/161720'
#data_location = '/2024-07-16_sa_ilp_er/172921'
#data_location = '/2024-07-16_sa_ilp_er/172921'
#data_location = '/2024-07-17_sa_ilp_er/c4300'
#data_location = '/2024-07-17_sa_ilp_er/15122__0'

#with open(dir_name+'/data'+data_location+'.pkl', 'rb') as f:
#    data_dict_loaded = pickle.load(f)
#    f.close()

#print(data_dict_loaded.keys())



#out_df = pd.DataFrame(data_dict_loaded["out_dict"])
#print(out_df)
#print(data_dict_loaded["out_dict"].keys())


def roundoff(ele):
    return int(ele*100)/100

out_df["prob_val"] = out_df["prob_val"].apply(roundoff)


wide = out_df.groupby(['prob_val'],as_index=False).mean()
std = out_df.groupby(['prob_val'],as_index=False).std()
max_df = out_df.groupby(['prob_val'],as_index=False).max()
min_df = out_df.groupby(['prob_val'],as_index=False).min()
#print(wide)
#print(std)

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
sns.lineplot(data = wide, x="prob_val", y="rt_ilp", label = "ILP")
sns.lineplot(data = wide, x="prob_val", y="rt_sailp", label = "SA+ILP")#, hue = "prob_val")
plt.yscale("log")
plt.legend()
plt.xlabel("Probability")
plt.ylabel("Average runtime (s)")
#plt.savefig(dir_name +'/plots' + data_location + "runtime" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
plt.plot(wide['prob_val'],
                 wide['rt_ilp'], label='ILP')
plt.fill_between(wide['prob_val'],
                 max_df['rt_ilp'],
                 y2=min_df['rt_ilp'], alpha=0.3)
plt.plot(wide['prob_val'],
                 wide['rt_sailp'], label="SA+ILP")
plt.fill_between(wide['prob_val'],
                 max_df['rt_sailp'],
                 y2=min_df['rt_sailp'], alpha=0.3)
plt.yscale("log")
plt.xlabel("Probability")
plt.ylabel("Runtime (s)")
plt.legend(loc = "upper left")
#plt.savefig(dir_name +'/plots' + data_location + "rt_with_band" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
wide_edge = out_df.groupby(['edge'],as_index=False).mean()
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
sns.lineplot(data = wide_edge, x="edge", y="rt_sailp", label = "SA+ILP")#, hue = "prob_val")
sns.lineplot(data = wide_edge, x="edge", y="rt_ilp", label = "ILP")
plt.yscale("log")
plt.legend()
plt.xlabel("Number of edges")
plt.ylabel("Average runtime (s)")
#plt.savefig(dir_name +'/plots' + data_location + "runtime_edges" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
#sns.color_palette("husl", 8)
sns.lineplot(data = wide, x="prob_val", y="edge", label='Initial edges')
sns.lineplot(data = wide, x="prob_val", y="sa_edge", label='SA')
sns.lineplot(data = wide, x="prob_val", y="sa_ilp_edge", label='SA+ILP')
plt.legend()
#plt.yscale("log")
plt.xlabel("Probability")
plt.ylabel("Average edges")
#plt.savefig(dir_name +'/plots' + data_location + "edges" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')


"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
plt.plot(wide['prob_val'], wide['edge'], label='Initial edges')#, marker = 'o')
plt.fill_between(wide['prob_val'],
                 wide['edge']+std['edge'],
                 y2=wide['edge']-std['edge'], alpha=0.3)

plt.plot(wide['prob_val'], wide['sa_edge'], label = "SA")#, marker = 'o')
plt.fill_between(wide['prob_val'],
                 wide['sa_edge']+std['sa_edge'] ,
                 y2=wide['sa_edge']-std['sa_edge'], alpha=0.3)

plt.plot(wide['prob_val'], wide['sa_ilp_edge'], label = "SA+ILP")#, marker = 'o')
plt.fill_between(wide['prob_val'],
                 wide['sa_ilp_edge']+std['sa_ilp_edge'] ,
                 y2=wide['sa_ilp_edge']-std['sa_ilp_edge'], alpha=0.3)
plt.xlabel("Probability")
plt.ylabel("Edges")
plt.legend()
#plt.savefig(dir_name +'/plots' + data_location + "edge_with_band" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')

plt.figure()
sns.scatterplot(data=out_df, x="edge", y="sa_ilp_edge")

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
out_df = out_df.rename(columns={"rt_sailp": "SA+ILP", "rt_ilp": "ILP"})
#print(out_df)
rt_df = pd.melt(out_df, id_vars=['prob_val'], value_vars=['SA+ILP', 'ILP'],
                var_name='Method', value_name='runtime')
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
ax = sns.catplot(data = rt_df, x="prob_val", y="runtime",
            hue = "Method", kind = "strip", alpha = 0.75,
            s = 10, legend=False, height=5, aspect=1.5)# dodge=True
sns.pointplot(
    data=rt_df, x="prob_val", y="runtime", hue="Method",
    dodge=0, errorbar=None)
plt.yscale("log")
plt.xlabel("Number of vertices")
plt.ylabel("Runtime")
#plt.legend(loc='upper left')
#plt.legend(["SA+ILP", "ILP","SA+ILP", "ILP"])
#sns.move_legend(ax, "upper left", bbox_to_anchor=(.10, .85))
plt.tick_params(direction='out', length=6, width=200,
                grid_alpha=0.5 , pad = 1)#grid_color='r'
plt.tight_layout()
#plt.savefig(dir_name +'/plots' + data_location + "catplot_withline" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')


print(0)
