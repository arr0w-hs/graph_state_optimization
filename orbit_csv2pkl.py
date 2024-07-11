#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:47:25 2024

@author: hsharma4

code for converting adcock's orbit data into a nested list'
"""

from pathlib import Path

import sys
import os


import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

import csv
from ast import literal_eval




def csv2list_convertor(csv_location):
    orbit_list_temp = []
    with open(csv_location) as in_file:
        spamreader = csv.reader(in_file, delimiter='"', quotechar='|')
        for row in spamreader:
            orbit_list_temp.append(row[5])
        in_file.close()
    
    
    """list with elements as strings"""
    orbit_string_list = []
    for ol_ele in orbit_list_temp:
        temp_list = []
        ol_ele = ol_ele.split("(")
        for ele in ol_ele:
            ele = ele.split(")")
            temp_list.append(ele[0])
        temp_list = temp_list[2:]
        orbit_string_list.append(temp_list)
    
    orbit_list = []
    
    """orbit_list first has a list with all the possible orbits for the given number of vertices
    
       each of these lists has all the graphs which are LC equivalent of each other"""
       
    
    for orbit_element in orbit_string_list:
        orbit_edge_list = []
        for ol in orbit_element:
            #print(ol)
            edge_list = []
            edge_element = []
            flag = 0
            for ele in ol:
                if ele.isnumeric():
                    edge_element.append(int(ele))
                    flag += 1
                    #print(edge_element)
                    if flag ==2:
                            edge_list.append(tuple(edge_element))  
                            edge_element = []
                            flag = 0
                #print(' ')
        
            orbit_edge_list.append(edge_list)
        
        orbit_list.append(orbit_edge_list)
    
    return orbit_list


if __name__ == "__main__":
    
    orbit_dict = {}
    for i in range(4, 9):
        #print(i)
        orbit= 0
        csv_location = dir_name+"/orbits-data/"+str(i)+"qubitorbitsCi.csv"
        orbit = csv2list_convertor(csv_location)
        
        orbit_dict[i] = orbit
    
    
    print(dir_name)
    with open(dir_name
              + "/"+ "orbit" +'.pkl', 'wb') as f:  # open a text file
        pickle.dump(orbit_dict, f)    
            
            
            
            
            
