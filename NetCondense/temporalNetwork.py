# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:21:25 2016

@author: Bijaya Adhikari
"""

import networkx as nx

import collections

class TemporalNetwork(object):

    def __init__(self, edgelist):
        f = open(edgelist, 'r')
        lines = f.readlines()
        f.close()
    
        graphlist = {}
        n = 1
        t = 1
        for line in lines:
            tempList = line.split()
            if tempList[0].isdigit():
                index = int(tempList[0])
                node1  = int(tempList[1])
                node2 = int(tempList[2])
                if t < index:
                    t = index
                if node1 > n:
                    n = node1
                if node2 > n:
                    n = node2
                if index not in graphlist:
                    g = nx.DiGraph()
                    if(float(tempList[3]) > 0):
                        g.add_edge(node1,node2, weight=float(tempList[3]))
                    if(float(tempList[4]) > 0):
                        g.add_edge(node2,node1, weight=float(tempList[3]))
                    graphlist[index] = g
                else:
                    if(float(tempList[3]) > 0):
                        graphlist[index].add_edge(node1,node2, weight=float(tempList[3]))
                    if(float(tempList[4]) > 0):
                        graphlist[index].add_edge(node2,node1, weight=float(tempList[4]))
        
        for t,graph in graphlist.items():
            graph.add_nodes_from(range(1,n+1))
            
        self.tempGraph =  collections.OrderedDict(sorted(graphlist.items()))
        self.n = n
        self.t = t
        