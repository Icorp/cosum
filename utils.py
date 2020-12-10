import math
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import re
import sys
import random
import logging as log
import numpy as np
import time
from nltk import sent_tokenize
from findIt import findNumWordInSentence
from findIt import final_token
from findIt import findTokenAndLower


# this method is find how many cluster is need ...
def findK(text):
    n = len(sent_tokenize(text))
    num_of_terms = len(final_token(text))
    num_of_words = len(findTokenAndLower(text))
    k=n*(num_of_terms/num_of_words)
    return round(k,0)

# This method is select random numbers for init centroid of cluster ...
def selectCenterOfCluster(k,text):
    # Sentences...
    sentences = sent_tokenize(text)

    # Select random
    a = random.sample(sentences,int(k))
    
    # Get index 
    indexs = []
    for i in range(len(a)):
        indexs.append(sentences.index(a[i]))
    return indexs

# TODO:
def computeRank():
    M = 15
    summ = 0
    for r in range(M):
        summ += ((M-r+1)*R[r])/M

#  This method is convert kmeans.labels to matrix(CQ)
def labelInMatrix(labels):
    maxQ = max(labels)
    results = []
    for q in range(maxQ+1):
        result = [0] * (len(labels) - 1)
        indexs = [i for i,x in enumerate(labels) if x==q]
        for o in indexs:
            result.insert(o,1)
        if len(result)>len(labels):
            a = result[:-(len(result)-len(labels))]
            result = a
        results.append(result)
    return results

# This method is return cluster with index of sentence ...
def clusteringSentence(label):
    result = []
    cluster = max(label)
    for q in range(cluster+1):
        cash = []
        for i, j in enumerate(label):
            if j == q:
                cash.append(i)
        result.append(cash) 
    return result

# This method is compute similiraty between 2 sentences
def computeSimilarity(Wi,Wj):
    f1 = funcSum(Wi,Wj)
    f2 = funcSum2(Wi,Wj)
    f3 = funcSum3(Wi)
    f4 = funcSum3(Wj)
    return 1-((2*f1*f2)/((f4*f1)+(f3*f2)))

# This method is concatenate list elements to strings
def concatenate_list_data(s):
    # initialize an empty string 
    str1 = " "  
    
    # return string   
    return (str1.join(str(x) for x in s)) 

"""
# This method is calculate euclidian distance
def euclidian(X,Y):
    point1 = np.array(X)
    point2 = np.array(Y)
    # subtracting vector 
    temp = point1 - point2 
    
    # doing dot product 
    # for finding 
    # sum of the squares 
    sum_sq = np.dot(temp.T, temp) 
    
    # Doing squareroot and 
    # printing Euclidean distance 
    return np.sqrt(sum_sq)
"""

#   loop for similarity function (equantion 3) 
def funcSum(Wi,Wj):
    seqSum=0.0
    m = len(Wi)
    for k in range(m):
        cash=Wi[k]-(Wi[k]*Wj[k])
        seqSum+=cash
    return seqSum    

#   loop for similarity function (equantion 3)
def funcSum2(Wi,Wj):
    seqSum=0.0
    m = len(Wi)
    for k in range(m):
        cash = Wj[k]-Wi[k]*Wj[k]
        seqSum+=cash
    return seqSum

#   loop for similarity function (equantion 3)
def funcSum3(W):
    seqSum=0.0
    m = len(W)
    for k in range(m):
        seqSum+=W[k]
    return seqSum

#   loop for diver function (equantion 14 )
def funcSum4(nq,S,q,clusterSentence,genome):
    summ = 0
    n = nq - 1
    for i in range(n):
        for j in range(nq):
            j = i+1
            similarity = computeSimilarity(S[clusterSentence[q][i]],S[clusterSentence[q][j]]) 
            summ += (1-similarity)*genome[clusterSentence[q][i]]*genome[clusterSentence[q][j]]
    return summ

