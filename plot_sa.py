#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:42:36 2024

@author: hsharma4
file for plotting graphs from saved data
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

data_location = '/2024-05-14_simulated_annealing/15157'
with open(dir_name+'/data'+data_location+'.pkl', 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

print(data_dict_loaded.keys())

n = data_dict_loaded["n"]
#n=10
x_axis = data_dict_loaded["prob_list"]
edge_data = data_dict_loaded["edge_data"]
#cl_data = data_dict_loaded["cl_data"]
nm_data = data_dict_loaded["nm_data"]
sa_data = data_dict_loaded["sa_data"]

#print(())
#print((np.std(edge_data, axis = 1)))

mean_edge_data = np.mean(edge_data, axis = 1)
y1_edge_data = np.mean(edge_data, axis = 1) - np.std(edge_data, axis = 1)
y2_edge_data = np.mean(edge_data, axis = 1) + np.std(edge_data, axis = 1)

mean_nm_data = np.mean(nm_data, axis = 1)
y1_nm_data = np.mean(nm_data, axis = 1) - np.std(nm_data, axis = 1)
y2_nm_data = np.mean(nm_data, axis = 1) + np.std(nm_data, axis = 1)

mean_sa_data = np.mean(sa_data, axis = 1)
y1_sa_data = np.mean(sa_data, axis = 1) - np.std(sa_data, axis = 1)
y2_sa_data = np.mean(sa_data, axis = 1) + np.std(sa_data, axis = 1)


plt.figure()
plt.grid()
plt.plot(x_axis, mean_edge_data,label = "Initial edges")
plt.fill_between(x_axis, y1_edge_data, 
                 y2= y2_edge_data, alpha = 0.2, linewidth=0, label="_a")

plt.plot(x_axis, mean_nm_data, label = "New metric")
plt.fill_between(x_axis, y1_nm_data, 
                 y2= y2_nm_data, alpha = 0.2,linewidth=0, label="_a")

plt.plot(x_axis, mean_sa_data, label = "Simulated annealing")
plt.fill_between(x_axis, y1_sa_data, 
                 y2= y2_sa_data, alpha = 0.2,linewidth=0, label="_a")

plt.title("n="+str(n))
plt.ylabel('Number of edges')
plt.xlabel('Graph probability')
#plt.legend(["Initial edges", "", "New metric", "",  "Simulated annealing"])
plt.legend()



#plt.savefig(dir_name +'/plots' + data_location + "_probability" + ".png", dpi=800, format="png", bbox_inches = 'tight')
"""
sns.lineplot(data = nm_data[0])#, x = x_axis, y = "edge_data")
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