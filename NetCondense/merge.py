# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:21:25 2016

@author: Bijaya Adhikari
"""
import sys
from temporalNetwork import TemporalNetwork
from unionFind import UnionFind
import networkx as nx
from timeit import default_timer as timer


def readScores(scoreFile):
    score = []
    f = open(scoreFile, 'r')
    for line in f:
        scoreDatum = line.split()
        score.append([int(scoreDatum[0]), int(scoreDatum[1]), float(scoreDatum[2])])
    return score
        
        
        
        
def netCondenseRoutine(graphfile, alpha1, alpha2, edgeScore, timeScore):
    tempNetwork = TemporalNetwork(graphfile)
    print("netowrk size is : #node = "+ str(tempNetwork.n)+ " #time-stamp = "+ str(tempNetwork.t))
    superNodes = UnionFind(tempNetwork.n)
    isTimeMerge = True
    isNodeMerge = True
    
    if alpha1 == 0:
        isNodeMerge = False;
    
    if alpha2  == 0:
        isTimeMerge  = False;
    timeIndex = 0;
    nodeIndex = 0;  
    nodeScores = readScores(edgeScore)
    timeScores = readScores(timeScore)
    
    nodeMergeCount = 0
    timeMergeCount  = 0
    isTimeActive = [1]*(tempNetwork.t +1)
    print("start merging")
    while isNodeMerge or isTimeMerge:
        
        
        if (isTimeMerge and isNodeMerge and (nodeScores[nodeIndex][2] <= timeScores[timeIndex][2])) or (isNodeMerge and  not(isTimeMerge)):
            
            i = superNodes.find(nodeScores[nodeIndex][0])
            j = superNodes.find(nodeScores[nodeIndex][1])
            nodeIndex += 1
            if(i == j):
                #print("Same Nodes : "+ str(i)+" and "+ str(j))
                continue
            else:
                start = timer()
                superNodes.union(i,j)
                nodeRemain = superNodes.find(i)
                nodeOut = j  if i == nodeRemain else i
                #print("remain is "+ str(nodeRemain))
                #print("out is "+ str(nodeOut))
                for timeStamp in range(1, tempNetwork.t+1):
                    if(isTimeActive[timeStamp] == 0):
                        continue
                    else:
                        curGraph = tempNetwork.tempGraph[timeStamp]
                        neighborsI = curGraph.neighbors(nodeRemain)
                        neighborsJ = curGraph.neighbors(nodeOut)
                        dic ={}
                        for node in neighborsI:
                            dic[node]=True
                            curGraph[nodeRemain][node]['weight'] = (curGraph[nodeRemain][node]['weight'])/(2.0)
                        for node in neighborsJ:
                            if node in dic:
                                curGraph[nodeRemain][node]['weight'] = (curGraph[nodeRemain][node]['weight']*2 + curGraph[nodeOut][node]['weight'])/(4.0)
                            else:
                                curGraph.add_edge(nodeRemain, node,  weight= (curGraph[nodeOut][node]['weight'])/(2.0))
                        
                        neighborsI = curGraph.predecessors(nodeRemain)
                        neighborsJ = curGraph.predecessors(nodeOut)
                        dic ={}
                        for node in neighborsI:
                            dic[node]=True
                            curGraph[node][nodeRemain]['weight'] = (curGraph[node][nodeRemain]['weight'])/(2.0)
                        for node in neighborsJ:
                            if node in dic:
                                curGraph[node][nodeRemain]['weight'] = (curGraph[node][nodeRemain]['weight']*2 + curGraph[node][nodeOut]['weight'])/(4.0)
                            else:
                                curGraph.add_edge(node, nodeRemain,  weight= (curGraph[node][nodeOut]['weight'])/(2.0))
                        curGraph.remove_node(nodeOut)
                nodeMergeCount += 1
                end = timer()
                print(str(nodeMergeCount) + "nodes merged  : time elapsed "+ str(end - start))
                
                        
                    
                if(nodeMergeCount >= alpha1 * tempNetwork.n):
                    isNodeMerge = False
                    
                    
        elif (isTimeMerge and isNodeMerge  and (nodeScores[nodeIndex][2] > timeScores[timeIndex][2])) or (not(isNodeMerge) and isTimeMerge):
            startT = timer()            
            time1 = timeScores[timeIndex][0]
            time2 = timeScores[timeIndex][1]
            
            while  (isTimeActive[time1] != 1):
                time1 -= 1
            
            Graph1 = tempNetwork.tempGraph[time1]
            Graph2 = tempNetwork.tempGraph[time2]
            for edge in Graph1.edges(data=True):
                if Graph2.has_edge(edge[0],edge[1]):
                    Graph1[edge[0]][edge[1]]['weight'] = (Graph1[edge[0]][edge[1]]['weight'] + Graph2[edge[0]][edge[1]]['weight'])/2.0
                    Graph2.remove_edge(edge[0],edge[1])
                else:
                    Graph1[edge[0]][edge[1]]['weight'] /= 2.0
            for edge in Graph2.edges(data=True):
                Graph1.add_edge(edge[0],edge[1], weight = edge[2]['weight']/2.0)
            endT = timer()
            print('Mering time '+ str(time1)+'  and  '+str(time2)+'  '+ str(timeMergeCount) + "times merged  : time elapsed "+ str(endT - startT))
            isTimeActive[time2] = 0
            
            timeIndex += 1
            timeMergeCount += 1
            if(timeMergeCount >= alpha2 * tempNetwork.t):
                    isTimeMerge = False
    
    superNodesFile = open(graphfile+"_superNodes", 'w')
    superNodesList = {}
    for i in range(1,(tempNetwork.n)+1):
        sup = superNodes.find(i)
        if sup not in superNodesList:
            superNodesList[sup] = [i]
        else:
            superNodesList[sup].append(i)
    for sup in superNodesList:
        superNodesFile.write(str(sup)+':\t')
        for i in superNodesList[sup]:
            superNodesFile.write(str(i)+',\t')
        superNodesFile.write('\n')
    superNodesFile.close()
    
    outFile = open(graphfile+"_edgeList_final", 'w')
    for i in range(1, tempNetwork.t+1):
        if(isTimeActive[i]==1):
            curGraph = tempNetwork.tempGraph[i]
            for line in nx.generate_edgelist(curGraph, data=['weight']):
                outFile.write(str(i) + '\t' +line + "\n")
    outFile.close()
    
    return tempNetwork

def main():
    graphfile = sys.argv[1]
    alpha1 = float( sys.argv[2])
    alpha2 = float(sys.argv[3]) 
    edgeScore = sys.argv[4]
    timeScore = sys.argv[5]
    te =  netCondenseRoutine(graphfile, alpha1, alpha2, edgeScore, timeScore)


if __name__ == "__main__":
    main()
    
#cProfile.run('main()')                      
                        
        
            
    