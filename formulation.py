import itertools
import cosum

def funcSum(w1,w2):
    seqSum=0.0
    for d,f in itertools.zip_longest(w1,w2,fillvalue=0):
        cash=(d-d*f)
        seqSum+=cash
    return round(seqSum,3)    

def funcSum2(w1,w2):
    seqSum=0.0
    for d,f in itertools.zip_longest(w1,w2,fillvalue=0):
        cash=(f-d*f)
        seqSum+=cash
    return round(seqSum,3)

def funcSum3(w):
    seqSum=0.0
    for i in range(len(w)):
        seqSum+=w[i]
    return round(seqSum,3)

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
    return round(seqSum,3)

