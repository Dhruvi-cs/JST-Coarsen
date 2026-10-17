# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:21:25 2016

@author: Bijaya Adhikari
"""

class UnionFind(object):


    def __init__(self, size):
        parent = [0]*(size+1)
        rank = [0]*(size+1)
        for i in range(size+1):
            parent[i] = i
     
        for  i in range(size+1):
            rank[i] = 0
        self.parent = parent
        self.rank = rank

    def union( self , x, y):
         x_root = self.find(x)
         y_root = self.find(y)
         if x_root == y_root:
             return
     
         if self.rank[x_root] > self.rank[y_root]:
             self.parent[y_root] = x_root
         else:
             self.parent[x_root] = y_root
             if self.rank[x_root] == self.rank[y_root]:
                 self.rank[y_root] = self.rank[y_root] + 1
    def find(self,x):
         if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
         return self.parent[x]
     


