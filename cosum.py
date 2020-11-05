import math
import numpy as np
import random
from nltk import sent_tokenize
from nltk import word_tokenize
from time import perf_counter
from cosum.findIt import findNumOfWord
from cosum.findIt import tokenizeAndRemoveStopWord
from cosum.findIt import tokenizeRemoveAndStemm
from cosum.findIt import findNumWordInSentence
from cosum.findIt import findNumSentencesSearchingWord
from cosum.findIt import finalTokenSentence
from cosum.findIt import final_token
from cosum.findIt import findTokenAndLower
from cosum.formulation import funcSum
from cosum.formulation import funcSum2
from cosum.formulation import funcSum3
from cosum.formulation import funcSum4
from cosum.formulation import funcSumExp
from cosum.formulation import funcSumExp2
from cosum.formulation import funcSumExp3
from cosum.formulation import funcSumCenter


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
    sentences = sent_tokenize(document)
    result = []
    for i in range(len(sentences)):
        result.append(computeWeightOfSentence(sentences[i],sentences,document))
    return result

def computeFullWeight(text):
    sent_tokens = sent_tokenize(text)
    word_tokens = final_token(text)
    result = []
    for i in range(len(sent_tokens)):
        cash = []
        for m in range(len(word_tokens)):
            cash.append(computeTDIDF(word_tokens[m],sent_tokens[i],sent_tokens,text))
        result.append(cash)
    return result
        

def computeTDIDF(word,sentence,sent_tokens,text):
    TF = findNumWordInSentence(sentence,word.lower())
    idf = computeIDF(word,sentence,sent_tokens)
    l_avg = computeAverageSentenceLength(text,sent_tokens)
    l_i = len(tokenizeAndRemoveStopWord(sentence))
    return round(((TF)/(TF+0.5+(1.5*(l_i/l_avg))))*idf,3)

def findWeightOfWord(sentence,word,text):
    TF = findNumWordInSentence(sentence,word.lower())
    return round(((TF)/(TF+0.5+(1.5*(len(tokenizeAndRemoveStopWord(sentence))/computeAverageSentenceLength(text,sent_tokenize(text))))))*computeIDF(word.lower(),sentence,sent_tokenize(text)),3)

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
    sentences = sent_tokenize(document)
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
    
    sentences = sent_tokenize(document)
    result = []
    for i,x in enumerate(sentences):
        for k,y in enumerate(sentences):
            print("S"+str(i)," == ","S"+str(k),"OK","time:"+str(perf_counter()))
            result.append(compute_sim_opt(w[i],w[k]))
    return result

def computeMatrixSimRRN(document):
    print("Start compute weight ...")
    w = computeFullWeight(document)
    print("finish = 100%")
    
    sentences = sent_tokenize(document)
    result = []
    for i,x in enumerate(sentences):
        cash = []
        for k,y in enumerate(sentences):
            #print("S"+str(i)," == ","S"+str(k),"OK","time:"+str(perf_counter()))
            cash.append(compute_sim_opt(w[i],w[k]))
        result.append(cash)
    return result


def findK(text):
    n = len(sent_tokenize(text))
    num_of_terms = len(final_token(text))
    num_of_words = len(findTokenAndLower(text))
    k=n*(num_of_terms/num_of_words)
    return round(k,0)


def computeClustering(S,q):
    sentences = S
    centroid = [] 
    result = []
    for k in range(q):
        index = random.randint(0, len(sentences))
        centroid.append(index)
        sentences[k] = 0
    for i in range(len(sentences)):
        cash = []
        for k in range(q):
            if sentences[i] == 0:
                cash.append(0)
            else:
                print("Start clustering...")
                print("Centroid => ",centroid)
                print("Loop #",k)
                print("Sentence",sentences[i])
                print("Sentence",sentences[centroid[k]])
                cash.append(computeSimClustering(sentences[i],sentences[centroid[k]]))
                    
                index = cash.index(max(cash))
                print("cash:",cash)
                result.append(index)        
        print("Centroids:",centroid)    
    return result

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
    return round(1-((2*f1*f2)/((f4*f1)+(f3*f2))),3)

def computeSimilirity(tokens,centroids):
    similarities = []
    for i in range(len(tokens)):
        cash = []
        for k in range(len(centroids)):
            cash.append(compute_sim_opt(tokens[i],centroids[k]))
        similarities.append(cash)
    return similarities

def findLabels(similarities):
    labels = []
    for i in range(len(similarities)):
        label = similarities[i].index(max(similarities[i]))
        labels.append(label)
    return labels

def getCentroidValue(centroidsIndexs,tokens):
    result = []
    for k,v in enumerate(centroidsIndexs):
        result.append(tokens[v])
    return result

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
    
def calculateCenter(tokens,matrix,c):
    result = []
    for q in range(len(c)):
        cash = []
        for l in range(len(tokens[0])):
            summ = funcSumCenter(tokens,c,l,q)
            w = summ/matrix[q].count(1)
            cash.append(round(w,3))
        result.append(cash)
    return result

def computeOptimalCentroid(tokens,centroids,exit):
    if exit == 0:
        return centroids
    # compute similarity
    similarities = computeSimilirity(tokens,centroids)
    #print("Similiraties:",similarities,"\n")
    
    # clustering result
    labels = findLabels(similarities)
    print("Labels:",labels,"\n")

    # toMatrix
    matrix = labelInMatrix(labels)
    #print("Matrix:",matrix,"\n")
        
    # get index
    c = getIndexCentroid(centroids,matrix)
    print("C",c)

    # calculate center
    result = calculateCenter(tokens,matrix,c)
    print(result)
    cash = []
    for s in range(len(result)):
        a = set(centroids[s]) & set(result[s])
        if len(a)==0:
            cash.append(True)
        else:
            cash.append(False)
    if False in cash:
        computeOptimalCentroid(tokens,centroids,exit-1)
    else:
        return result


def getIndexCentroid(centroids,matrix):
    c = []
    for q in range(len(centroids)):
        cash = []
        for i in range(len(matrix[q])):
            if 1 == matrix[q][i]:
                cash.append(i)
        c.append(cash)
    return c

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
    f1 = funcSumExp(w1,w2)
    f2 = funcSumExp2(w1,w2)
    f3 = funcSumExp3(w1)
    f4 = funcSumExp3(w2)
    return round(1-((2*f1*f2)/((f4*f1)+(f3*f2))),3)

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
    


            