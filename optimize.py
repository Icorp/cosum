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
def F(X,O,sentences,random_s):
    index_sentences = mix(random_s)
    final_sentences = selectSentences(index_sentences,sentences)
    text = ' '.join(final_sentences)
    count = 0
    for listElem in random_s:
        count += len(listElem)   
    print("Number of sentences =",count)
    s = computeAllWeightOfDocument(text)
    f_cov = f_cover(O,X,s,text)
    print("F_COVER =",f_cov)
    f_div = f_diver(X,s)
    print("F_DIVER =",f_div)
    return round((2*f_cov*f_div)/(f_cov+f_div),3)

def f_cover(O,X,s,text):
    cash2 = computeMatrixSimRRN(text)
    summ = 0
    k = len(X)
    for q in range(k):
        n = X[q].count(1)
        for i in range(n):
            summ +=computeSimClustering(s[i],O[q])*X[q][i]
    return summ

def f_diver(X,S):
    k = len(X)
    summ = 0
    result = 0   
    for q in range(k):
        n = X[q].count(1)
        summ += funcSum4(n,S,q,X)
        result += (2/n*(n-1))*summ
    return result

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
        print("Не соответствует требованиям")
def stageThree():
    result = F()
    print("F(x) = ",result)