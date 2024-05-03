#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:42:36 2024

@author: hsharma4
"""


import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

#data_location = '/2024-04-25_cat_disti_comparison/20240'
data_location = '/2024-05-02_local_complementation/175120'
with open(dir_name+'/data'+data_location+'.pkl', 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

print(data_dict_loaded.keys())


x_axis = data_dict_loaded["prob_list"]
edge_data = data_dict_loaded["edge_data"]
cl_data = data_dict_loaded["cl_data"]
nm_data = data_dict_loaded["nm_data"]
tr_data = data_dict_loaded["tr_data"]

sns.lineplot(data = edge_data, x = x_axis)


"""
plt.figure()
plt.grid()
plt.scatter(prob_in_state_list, data_dict_loaded["prob_cat_list"], s = 5, c = "red")#, "ob", alpha = 0.3)
plt.scatter(prob_in_state_list, data_dict_loaded["prob_nocat_list"], s = 5, c = "blue")
plt.scatter(prob_in_state_list, data_dict_loaded["prob_dist_list"], s = 5, c = "limegreen")
plt.scatter(prob_in_state_list, data_dict_loaded["prob_cat_reuse_list"], s = 5, c = "orange")
#plt.gca().invert_xaxis()
#plt.scatter(fid_raw_list, dames_prob_list, s = 4, c = "orange")
#plt.title('Probability')
#plt.yscale("log")
#plt.xscale("log")
plt.ylabel('Probability of success')
#plt.xlabel('Raw state infidelity')
#plt.xlabel('Error in state ("coherent error")')
plt.xlabel('Error probability ("mixed-ness")')
plt.xticks(rotation=45)
plt.legend(["Catalytic", "Non-catalytic", "Distillation", "Catalyst reuse"])
#plt.savefig(dir_name +'/plots' + data_location + "_probability" + ".png", dpi=800, format="png", bbox_inches = 'tight')
"""