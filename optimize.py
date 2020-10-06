import numpy as np
import sys
from cosum import computeSimClustering
from cosum import computeAllWeightOfDocument
from cosum import labelInMatrix
from cosum import selectSentences
from cosum import computeMatrixSimRRN
from cosum import mix
from formulation import funcSum4
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans

l_max = 100
#The evaluation function is an operation to evaluate how good the solution (sentence selection,i.e., summary) of each individual is, making comparison between different solutions possible. 
#The evaluation function consists of calculating the value of the objective function (9) of the summary
#represented by each individual.
def F(X,O,sentences,random):
    index_sentences = mix(random)
    final_sentences = selectSentences(index_sentences,sentences)
    text = ' '.join(final_sentences)
    count = 0
    for listElem in random:
        count += len(listElem)   
    print("Количество предложении =",count)
    w = computeAllWeightOfDocument(text)
    f_cov = f_cover(O,random,w)
    #print("F_COVER =",f_cov)
    f_div = f_diver(random,w)
    #print("F_DIVER =",f_div)
    return round((2*f_cov*f_div)/(f_cov+f_div),3)

# O - it is cluster centers. O [[],[],[]]
# index_sentences - 
def f_cover(O,x,s):
    summ = 0
    k = len(x)
    for q in range(k):
        n = len(x[q])
        for i in range(n):
            summ +=computeSimClustering(s[i],O[q])*x[q][i]
    return round(summ,3)

def f_diver(X,w):
    k = len(X)
    summ = 0
    result = 0   
    for q in range(k):
        nq = len((X[q]))
        summ += funcSum4(nq,w,q,X)
        result += (2/nq*(nq-1))*summ
    return round(result,3)

def stageOne(x,document):
    sentences = sent_tokenize(document)
    l_avg = len(word_tokenize(document))/len(sentences)
    num_of_word = 0
    for k in range(len(sentences)):
        num_of_word += len(word_tokenize(sentences[k]))
    result = []
    for q in range(len(x)):
        for i in range(len(x[q])):
            l_i = len(word_tokenize(sentences[i]))
            if (l_i*x[q][i])<=l_avg:
                result.append(True)
            else:
                result.append(False)
        
        for i in range(len(x[q])):
            if num_of_word <= l_max:
                result[i] == True
            else:
                result[i] == False
    return result
def stageTwo(status):
    if all(status) == True:
        stageThree()
    else:
        return "Не соответствует требованиям"

def stageThree():
    result = F()
    print("F(x) = ",result)