import itertools
import cosum
import sys

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

def funcSum4(nq,S,q,X):
    summ = 0
    n = nq - 1
    for i in range(n):
        for j in range(nq):
            j = i+1
            similarity = cosum.computeSimClustering(S[i],S[j]) 
            summ +=(1-similarity*X[q][i]*X[q][j])
    return round(summ,3)

def funcSumExp(w1,w2):
    seqSum=0.0
    for k in range(len(w1)):
        cash=(w1[k]-w1[k]*w2[k])
        seqSum+=cash
    return round(seqSum,3)

def funcSumExp2(w1,w2):
    seqSum=0.0
    for k in range(len(w1)):
        cash=(w2[k]-w1[k]*w2[k])
        seqSum+=cash
    return round(seqSum,3)

def funcSumExp3(w):
    seqSum=0.0
    for i in range(len(w)):
        seqSum+=w[i]
    return round(seqSum,3)

def funcSumCenter(tokens,c,l,q):
    seqSum=0.0
    for i in range(len(tokens)):
        if i in c[q]:
            uiq = 1
        else:
            uiq = 0
        seqSum += tokens[i][l]*uiq
    return seqSum


# for numpy
def funcSumNP1(w1,w2):
    seqSum=0.0
    for k in range(len(w1)):
        cash=(w1[k]-w1[k]*w2[k])
        seqSum+=cash
    return round(seqSum,3)

def funcSumNP2(w1,w2):
    seqSum=0.0
    for k in range(len(w1)):
        cash=(w2[k]-w1[k]*w2[k])
        seqSum+=cash
    return round(seqSum,3)

def funcSumNP3(w):
    seqSum=0.0
    for i in range(len(w)):
        seqSum+=w[i]
    return round(seqSum,3)