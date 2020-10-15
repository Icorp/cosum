import math
import numpy as np
import formulation
import random
from time import perf_counter
from findIt import findNumOfWord
from findIt import tokenizeAndRemoveStopWord
from findIt import tokenizeRemoveAndStemm
from findIt import findNumWordInSentence
from findIt import findNumSentencesSearchingWord
from findIt import findSentences
from findIt import finalTokenSentence
from findIt import final_token
from findIt import findTokenAndLower
from formulation import funcSum
from formulation import funcSum2
from formulation import funcSum3
from formulation import funcSum4


# Find sentence


#This method finds the average sentence length. 
#Input: string and array sentences.Example:"text" , "["s1","s2"...,"sn"]" 
#Returns: int
def computeAverageSentenceLength(text,sentences):
    return len(tokenizeAndRemoveStopWord(text))/len(sentences)
                
#This method calculates the weight of one word in the proposed sentence.
#Input: string,string,string.Example:"s1. s2. s3. ..." ,"word","document"  
#Returns: [float]
def computeWeightOfSentence(sentence,sentences,text):
    result = []
    words = final_token(sentence)
    for i in range(len(words)):
        TF = findNumWordInSentence(sentence,words[i].lower())
        Li = len(tokenizeAndRemoveStopWord(sentence))
        Lavg = computeAverageSentenceLength(text,sentences)
        Isf = computeIDF(words[i].lower(),sentence,sentences)
        result.append(round(((TF)/(TF+0.5+(1.5*(Li/Lavg))))*Isf,3))
    return result

def computeAllWeightOfDocument(document):
    sentences = findSentences(document)
    result = []
    for i in range(len(sentences)):
        result.append(computeWeightOfSentence(sentences[i],sentences,document))
    return result
    
def findWeightOfWord(sentence,word,text):
    TF = findNumWordInSentence(sentence,word.lower())
    return round(((TF)/(TF+0.5+(1.5*(len(tokenizeAndRemoveStopWord(sentence))/computeAverageSentenceLength(text,findSentences(text))))))*computeIDF(word.lower(),sentence,findSentences(text)),3)

def findWeightOfWordOpt(sentence,word,Lavg,tokens,sentences):
    TF = findNumWordInSentence(sentence,word.lower())
    IDF = computeIDF(word.lower(),sentence,sentences)
    return round((TF/(TF+0.5+(1.5*(len(tokens)/Lavg))))*IDF,3)

def computeIDF(word,sentence,sentences):
    if findNumSentencesSearchingWord(sentences,word.lower()) == 0:
        print("Данного слова в этом документ нет, введите словo из документа.")
        return 0
    else:
        return math.log10(len(sentences)/findNumSentencesSearchingWord(sentences,word.lower()))

def computeSimRRN(s1,s2,document,sentencesIndex):
    sentences = findSentences(document)
    w1 = computeWeightOfSentence(s1,sentences,document)
    w2 = computeWeightOfSentence(s2,sentences,document)
    f1 = funcSum(w1,w2)
    f2 = funcSum2(w1,w2)
    f3 = funcSum3(w1)
    f4 = funcSum3(w2)
    print("S"+str(sentencesIndex[0])," == ","S"+str(sentencesIndex[1]),"OK","time:"+str(perf_counter()))
    return round(1-((2*(f1)*f2)/((f3*f1)+(f4*f2))),3)

def computeAllSimRRN(document):
    print("Start compute weight ...")
    w = computeAllWeightOfDocument(document)
    print("finish = 100%")
    
    sentences = findSentences(document)
    result = []
    for i,x in enumerate(sentences):
        for k,y in enumerate(sentences):
            print("S"+str(i)," == ","S"+str(k),"OK","time:"+str(perf_counter()))
            result.append(compute_sim_opt(w[i],w[k]))
    return result

def computeMatrixSimRRN(document):
    print("Start compute weight ...")
    w = computeAllWeightOfDocument(document)
    print("finish = 100%")
    
    sentences = findSentences(document)
    result = []
    for i,x in enumerate(sentences):
        cash = []
        for k,y in enumerate(sentences):
            #print("S"+str(i)," == ","S"+str(k),"OK","time:"+str(perf_counter()))
            cash.append(compute_sim_opt(w[i],w[k]))
        result.append(cash)
    return result

def computeMatrixSimTwoRRN(s1,s2,document):
    d = []
    sentences = findSentences(document)
    d.append(computeMoreSimRRNOpt(s1,document,0,sentences))
    d.append(computeMoreSimRRNOpt(s2,document,1,sentences))
    return d

def findK(n,text):
    num_of_terms = len(final_token(text))
    num_of_words = len(findTokenAndLower(text))
    k=n*(num_of_terms/num_of_words)
    return k


def computeClustering(S,O,k,n,document):
    summ = 0
    for q in range(k):
        for i in range(n):
            summ += round(computeSimClustering(S[i],O[q]))
    return round(summ,3)

def computeSimClustering(S1,O):
    cash = []
    if isinstance(O[0],np.float64):
        for i in range(O.size):
            k = float(O[i])
            cash.append(k)
        S2 = cash
    else:
        S2 = O
    f1 = funcSum(S1,S2)
    f2 = funcSum2(S2,S2)
    f3 = funcSum3(S1)
    f4 = funcSum3(S2)
    return 1-((2*f1*f2)/((f4*f1)+(f3*f2)))

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

def compute_sim_opt(w1,w2):
    f1 = funcSum(w1,w2)
    f2 = funcSum2(w1,w2)
    f3 = funcSum3(w1)
    f4 = funcSum3(w2)
    return round(1-((2*(f1)*f2)/((f3*f1)+(f4*f2))),3)

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