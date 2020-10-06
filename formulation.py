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
    print(S)
    summ = 0
    n = nq - 1
    for i in range(n):
        for j in range(nq):
            j = i+1
            summ +=(1-cosum.computeSimClustering(S[i],S[j])*X[q][i]*X[q][j])
    return round(summ,3)