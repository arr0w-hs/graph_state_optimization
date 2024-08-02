#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:08:53 2024

@author: hsharma4
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



#data_location = '/2cd024-07-13_sa_ilp/225850'
#data_location = '/2024-07-13_sa_ilp/225049'
#data_location = '/2024-07-13_sa_ilp/175625'
#data_location = '/2024-07-13_sa_ilp/154120'

#data_location = '/2024-07-12_sa_ilp/161720'
#data_location = '/2024-07-15_sa_ilp/141516'

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
        #print(data_dict_loaded["out_dict"].keys())

out_df = pd.concat(df_list, axis=0, join='outer', ignore_index=True)
#print(out_df)

wide = out_df.groupby(['vertex_num'],as_index=False).mean()
std = out_df.groupby(['vertex_num'],as_index=False).std()
max_df = out_df.groupby(['vertex_num'],as_index=False).max()
min_df = out_df.groupby(['vertex_num'],as_index=False).min()
#print(wide)
#print(std)

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
sns.lineplot(data = wide, x="vertex_num", y="rt_ilp", label = "ILP")
sns.lineplot(data = wide, x="vertex_num", y="rt_sailp", label = "SA+ILP")#, hue = "vertex_num")
plt.yscale("log")
plt.legend()
plt.xlabel("Number of vertices")
plt.ylabel("Average runtime (s)")
#plt.savefig(dir_name +'/plots' + data_location + "runtime" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
wide_edge = out_df.groupby(['edge'],as_index=False).mean()
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
sns.lineplot(data = wide_edge, x="edge", y="rt_sailp", label = "SA+ILP")#, hue = "vertex_num")
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
sns.lineplot(data = wide, x="vertex_num", y="edge", label='Initial edges')
sns.lineplot(data = wide, x="vertex_num", y="sa_edge", label='SA')
sns.lineplot(data = wide, x="vertex_num", y="sa_ilp_edge", label='SA+ILP')
plt.legend()
#plt.yscale("log")
plt.xlabel("Number of vertices")
plt.ylabel("Average edges")
#plt.savefig(dir_name +'/plots' + data_location + "edges" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')


"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
plt.plot(wide['vertex_num'], wide['edge'], label='Initial edges')
plt.fill_between(wide['vertex_num'],
                 wide['edge']+2*std['edge'],
                 y2=wide['edge']-2*std['edge'], alpha=0.2)

plt.plot(wide['vertex_num'], wide['sa_edge'], label = "SA")
plt.fill_between(wide['vertex_num'],
                 wide['sa_edge']+2*std['sa_edge'] ,
                 y2=wide['sa_edge']-2*std['sa_edge'], alpha=0.2)

plt.plot(wide['vertex_num'], wide['sa_ilp_edge'], label = "SA+ILP")
plt.fill_between(wide['vertex_num'],
                 wide['sa_ilp_edge']+2*std['sa_ilp_edge'] ,
                 y2=wide['sa_ilp_edge']-2*std['sa_ilp_edge'], alpha=0.2)
plt.xlabel("Number of vertices")
plt.ylabel("Edges")
plt.legend()
#plt.savefig(dir_name +'/plots' + data_location + "edge_with_band" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
#plt.savefig(dir_name +'/plots' + data_location + "edge_with_band" + ".png", dpi=800, format="png", bbox_inches = 'tight')
"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
plt.plot(wide['vertex_num'],
                 wide['edge'], label='ILP')
plt.plot(wide['vertex_num'],
                 max_df['edge'], label='max ILP')#, marker = 'o')

plt.plot(wide['vertex_num'],
                 wide['sa_edge'], label="SA")
plt.plot(wide['vertex_num'],
                 max_df['sa_edge'], label="max SA")


plt.plot(wide['vertex_num'],
                 wide['sa_ilp_edge'], label="ILP")
plt.plot(wide['vertex_num'],
                 max_df['sa_ilp_edge'], label="max ILP")


#plt.yscale("log")
plt.xlabel("Number of vertices")
plt.ylabel("Edges")
plt.legend(loc = "upper left")

#print(out_df)
plt.figure()
sns.scatterplot(data=out_df, x="edge", y="sa_ilp_edge")
"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
plt.figure()
plt.plot(wide['vertex_num'],
                 wide['rt_ilp'], label='ILP')
plt.fill_between(wide['vertex_num'],
                 max_df['rt_ilp'],
                 y2=min_df['rt_ilp'], alpha=0.2)
plt.plot(wide['vertex_num'],
                 wide['rt_sailp'], label="SA+ILP")
plt.fill_between(wide['vertex_num'],
                 max_df['rt_sailp'],
                 y2=min_df['rt_sailp'], alpha=0.2)
plt.yscale("log")
plt.xlabel("Number of vertices")
plt.ylabel("Runtime (s)")
plt.legend(loc = "upper left")
#plt.savefig(dir_name +'/plots' + data_location + "rt_with_band" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')

"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
out_df = out_df.rename(columns={"rt_sailp": "SA+ILP", "rt_ilp": "ILP"})
#print(out_df)
rt_df = pd.melt(out_df, id_vars=['vertex_num'], value_vars=['SA+ILP', 'ILP'],
                var_name='Method', value_name='runtime')
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
ax = sns.catplot(data = rt_df, x="vertex_num", y="runtime",
            hue = "Method", kind = "strip", alpha = 0.95,
            s = 10, legend=True, height=5, aspect=1.5)# dodge=True
plt.yscale("log")
plt.xlabel("Number of vertices")
plt.ylabel("Runtime")
#plt.legend(loc='upper left')
#plt.legend(["SA+ILP", "ILP","SA+ILP", "ILP"])
sns.move_legend(ax, "upper left", bbox_to_anchor=(.10, .85))
plt.tick_params(direction='out', length=6, width=200,
                grid_alpha=0.5 , pad = 1)#grid_color='r'
plt.tight_layout()
plt.rcParams.update({'font.size': 12})
#plt.xticks(vertex_num, [str(i) for i in y], rotation=90)
#print(rt_df)
#plt.savefig(dir_name +'/plots' + data_location + "catplot1" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')



"""
    %%%%%%%%%%%%%%%%%%% fig %%%%%%%%%%%%%%%%%%%
"""
out_df = out_df.rename(columns={"rt_sailp": "SA+ILP", "rt_ilp": "ILP"})
#print(out_df)
rt_df = pd.melt(out_df, id_vars=['vertex_num'], value_vars=['SA+ILP', 'ILP'],
                var_name='Method', value_name='runtime')
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
ax = sns.catplot(data = rt_df, x="vertex_num", y="runtime",
            hue = "Method", kind = "strip", alpha = 0.75,
            s = 10, legend=False, height=4, aspect=1.5)# dodge=True
sns.pointplot(
    data=rt_df, x="vertex_num", y="runtime", hue="Method",
    dodge=0, errorbar=None)
plt.yscale("log")
plt.xlabel("Number of vertices")
plt.ylabel("Runtime")
#plt.legend(loc='upper left')
#plt.legend(["SA+ILP", "ILP","SA+ILP", "ILP"])
#sns.move_legend(ax, "upper left", bbox_to_anchor=(.10, .85))
#plt.tick_params(direction='out', length=6, width=200,
#                grid_alpha=0.5 , pad = 1)#grid_color='r'
plt.tight_layout()
#plt.savefig(plot_dir + "/catplot_withline" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
#plt.savefig(plot_dir + "/catplot_withline" + ".png", dpi=800, format="png", bbox_inches = 'tight')

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

out_df = out_df.rename(columns={"rt_sailp": "SA+ILP", "rt_ilp": "ILP"})
#print(out_df)
rt_df = pd.melt(out_df, id_vars=['vertex_num'], value_vars=['edge', 'sa_edge', 'sa_ilp_edge'],
                var_name='Method', value_name='edges')
plt.figure()
sns.set_theme(style = "whitegrid")#, palette = None)
ax = sns.catplot(data = rt_df, x="vertex_num", y="edges",
            hue = "Method", kind = "strip", alpha = 0.15,
            s = 50, legend=False, height=5, aspect=1.5)# dodge=True
sns.pointplot(
    data=rt_df, x="vertex_num", y="edges", hue="Method",
    dodge=0, errorbar=None)
#plt.yscale("log")
plt.xlabel("Number of vertices")
plt.ylabel("Runtime")
#plt.legend(loc='upper left')
#plt.legend(["SA+ILP", "ILP","SA+ILP", "ILP"])
#sns.move_legend(ax, "upper left", bbox_to_anchor=(.10, .85))
plt.tick_params(direction='out', length=6, width=200,
                grid_alpha=0.5 , pad = 1)#grid_color='r'
plt.tight_layout()
#plt.savefig(dir_name +'/plots' + data_location + "catplot_withline" + ".svg", dpi=800, format="svg", bbox_inches = 'tight')
"""
print(0)
