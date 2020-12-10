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

def findK(text):
    n = len(sent_tokenize(text))
    num_of_terms = len(final_token(text))
    num_of_words = len(findTokenAndLower(text))
    k=n*(num_of_terms/num_of_words)
    return round(k,0)


def computeSimClustering(S1,S2):
    f1 = funcSum(S1,S2)
    f2 = funcSum2(S2,S2)
    f3 = funcSum3(S1)
    f4 = funcSum3(S2)
    return round(1-((2*f1*f2)/((f4*f1)+(f3*f2))),3)

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

# M = number of compared methods =15
# R = number of times the method appears in the rth rank
def computeRank():
    M = 15
    summ = 0
    for r in range(M):
        summ += ((M-r+1)*R[r])/M

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

def randomizer(arr):
    result = []
    for i in range(len(arr)):
        k = random.randint(1, len(arr[i]))
        sampling = random.sample(arr[i], k)
        result.append(sampling)
    return result
def randomizer_3(arr):
    result = []
    for i in range(len(arr)):
        sampling = random.sample(arr[i], 1)
        result.append(sampling)
    return result
    
def randomizer_6(arr):
    result = []
    for i in range(len(arr)):
        sampling = random.sample(arr[i], 2)
        result.append(sampling)
    return result

#  [[0,1],[0,1]]  ==>  [0,1,0,1]
def mix(arr):
    result = []
    for i in range(len(arr)):
        for k in range(len(arr[i])):
            result.append(arr[i][k])
    return result

def selectSentences(arr,S):
    result = []
    for k,v in enumerate(arr):
        result.append(S[v])
    return result

def computeSimilarity(Wi,Wj):
    f1 = funcSum(Wi,Wj)
    f2 = funcSum2(Wi,Wj)
    f3 = funcSum3(Wi)
    f4 = funcSum3(Wj)
    return 1-((2*f1*f2)/((f4*f1)+(f3*f2)))

def get_summary(random_s,sentences):
    result = []
    for i in range(len(random_s)):
        for k in range(len(random_s[i])):
            result.append(sentences[random_s[i][k]])
    return result
    
def listToString(s):
    # initialize an empty string 
    str1 = " "  
    
    # return string   
    return (str1.join(str(x) for x in s)) 

def toVector(array):
    cash = []
    for i in range(len(array)):
        cash.append(len(array[i]))
    max_len = max(cash)
    index = cash.index(max_len)
    for i in range(len(array)):
        a = len(array[i])
        b = len(array[index])
        for k in range(b-a):
            array[i].append(0)
    for q in range(len(array)):
        print(len(array[q]))
    return array

def vectorize(text):
    sent_tokens = sent_tokenize(text)
    result = []
    word_tokens = final_token(text)
    for i in range(len(sent_tokens)):
        cash = []
        for w in range(len(word_tokens)):
            cash.append(findNumWordInSentence(sent_tokens[i],word_tokens[w]))
        result.append(cash)
    print(sent_tokens)
    print(word_tokens)
    print(result)
    return result
"""    
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
def funcSum(Wi,Wj):
    seqSum=0.0
    m = len(Wi)
    for k in range(m):
        cash=Wi[k]-(Wi[k]*Wj[k])
        seqSum+=cash
    return seqSum    

def funcSum2(Wi,Wj):
    seqSum=0.0
    m = len(Wi)
    for k in range(m):
        cash = Wj[k]-Wi[k]*Wj[k]
        seqSum+=cash
    return seqSum

def funcSum3(W):
    seqSum=0.0
    m = len(W)
    for k in range(m):
        seqSum+=W[k]
    return seqSum

def funcSum4(nq,S,q,clusterSentence,genome):
    summ = 0
    n = nq - 1
    for i in range(n):
        for j in range(nq):
            j = i+1
            similarity = computeSimilarity(S[clusterSentence[q][i]],S[clusterSentence[q][j]]) 
            summ += (1-similarity)*genome[clusterSentence[q][i]]*genome[clusterSentence[q][j]]
    return summ

def funcSumCenter(tokens,c,l,q):
    seqSum=0.0
    for i in range(len(tokens)):
        if i in c[q]:
            uiq = 1
        else:
            uiq = 0
        seqSum += tokens[i][l]*uiq
    return seqSum
