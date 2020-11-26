import math
import numpy as np
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import re
import sys
import random
import logging as log
import time
from nltk import sent_tokenize
from nltk import word_tokenize
from time import perf_counter
from findIt import findNumOfWord
from findIt import tokenizeAndRemoveStopWord
from findIt import tokenizeRemoveAndStemm
from findIt import findNumWordInSentence
from findIt import findNumSentencesSearchingWord
from findIt import finalTokenSentence
from findIt import final_token
from findIt import findTokenAndLower
from formulation import funcSum
from formulation import funcSum2
from formulation import funcSum3
from formulation import funcSum4
from formulation import funcSumExp
from formulation import funcSumExp2
from formulation import funcSumExp3
from formulation import funcSumCenter

# Test
from nltk.corpus import stopwords

# This is vectorizer. Todo: write documentation ...
class CosumTfidfVectorizer:
    def unionTokens(self):
        self.vocabulary = sorted(set(self.vocabulary), key=self.vocabulary.index)    
    
    def Stemm(self):
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in self.tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        self.notUnionTokens = [stemmer.stem(t) for t in filtered_tokens]

        # delete duplicate words
        self.vocabulary = sorted(set(self.notUnionTokens), key=self.notUnionTokens.index)
        
    
    def computeTF(self, word, i):
        indices = [i for i, x in enumerate(self.sentencesInWords[i]) if x == word]
        self.TF = len(indices)        
    
            
    def findNumSentencesSearchingWord(self, word):
        self.nj = 0
        for i in range(len(self.sentencesInWords)):
            if word in self.sentencesInWords[i]:
                self.nj += 1

    # Считает количество слов в предложении. На хвод принимает предложение, удаляет стоп слова. Sentence = "word word word"
    # Example:
    #   compute(self," I will be better tommorrow")
    #   Output: 2

    def computeLi(self, i):
        self.l_i = len(self.sentencesInWords[i])
    
    

    def computeISF(self, word):
        if self.nj == 0:
            log.error("Данного слова в этом документ нет, введите словo из документа.")
            sys.exit()
        else:
            self.isf =  math.log10(len(self.sentences)/self.nj)

    # Cреднее количество слов в каждом предложении.
    def computeAverageSentenceLength(self):
        self.l_avg = len(self.vocabulary)/len(self.sentences)

    # Данный метод находит вес одного слова.
    def computeTDIDF(self,word,i):
        self.computeLi(i)
        self.computeTF(word, i)
        self.findNumSentencesSearchingWord(word)
        self.computeISF(word)
        #print("TF\t i = ", i, "\t w = ", word, self.TF)
        #print("IDF\t w = ", word, self.nj)
        #print("L_I",self.l_i)
        #print("L_avg",self.l_avg)
        #print("L_i / L_avg = ",self.l_i/self.l_avg)
        #print()
        self.weight = (self.TF/(self.TF+0.5+(1.5*(self.l_i/self.l_avg))))*self.isf
    
    def convertSentenceToWords(self):
        for i in range(len(self.sentences)):
            word_tokens = word_tokenize(self.sentences[i])
            stop_words = set(stopwords.words('english'))
            self.cash_sentence_words = [w for w in word_tokens if not w in stop_words]
            self.sentencesInWords.append(self.cash_sentence_words)
            # Частота появленяи слова в предложении.

    # Метод считает вес всех слов.
    def computeFullWeight(self):
        self.weight_matrix = []
        self.sentencesInWords = []
        self.convertSentenceToWords()
        self.computeAverageSentenceLength()
        for i in range(len(self.sentences)):
            cash = []
            for m in range(len(self.vocabulary)):
                self.computeTDIDF(self.vocabulary[m],i)
                cash.append(self.weight)
            #print("Time:",round(time.perf_counter(),3)," sec")
            self.weight_matrix.append(cash)


    def fit(self,document):
        stop_words = set(stopwords.words('english'))
        self.document = document

        # 0 step: get sentences
        self.sentences = sent_tokenize(document)

        # 1 step: get words
        self.vocabulary = []
        for i in range(len(self.sentences)):
            word_tokens = word_tokenize(self.sentences[i])
            tokens = [w for w in word_tokens if not w in stop_words]
            for j in range(len(tokens)):
                self.vocabulary.append(tokens[j])
        
        # 2 step: delete duplicate words
        self.unionTokens()
        
        # 3 step: Stemming
        #self.Stemm()
        self.new_vocabulary = self.vocabulary
        # 3 step: calculate weight
        self.computeFullWeight()
        

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

class k_means:
    def __init__(self, k=3, max_iterations=52, metric="similarity"):
        self.metric = metric
        self.k = k
        self.max_iterations = max_iterations
    
    # Select random center
    def selectCenterOfCluster(self):
        self.centroidsIndexs = []
        
        # Select random
        a = random.sample(self.data,self.k)
            
        for i in range(len(a)):
            self.centroidsIndexs.append(self.data.index(a[i]))
    
    # convert centroid indexs to centroid value
    def getCentroidValue(self):
        self.centroids = []
        for k,v in enumerate(self.centroidsIndexs):
            self.centroids.append(self.data[v])
    
        
    # compute similirity between centroid and sentences 
    def computeSimilarity(self):
        if self.metric == "euclidean":
            self.similarities = []
            for i in range(len(self.data)):
                cash = []
                for k in range(len(self.centroids)):
                    cash.append(euclidian(self.data[i],self.centroids[k]))
                self.similarities.append(cash)
        else:
            self.similarities = []
            for i in range(len(self.data)):
                cash = []
                for k in range(len(self.centroids)):
                    cash.append(computeSimilarity(self.data[i],self.centroids[k])*self.matrix[k][i])
                self.similarities.append(cash)
        
    # compute similirity between centroid and sentences 
    def computeSimilarityBetweenSentences(self):
        self.similarities = []
        for i in range(len(self.data)):
            cash = []
            for k in range(len(self.centroidsIndexs)):
                cash.append(euclidian(self.data[i],self.data[self.centroidsIndexs[k]]))
            self.similarities.append(cash)
    
    # classifies sentences 
    def findLabels(self):
        self.labels = []
        for i in range(len(self.similarities)):
            if self.metric == "euclidean":
                label = self.similarities[i].index(min(self.similarities[i]))
                self.labels.append(label)
            else:
                label = self.similarities[i].index(max(self.similarities[i]))
                self.labels.append(label)
    # convert labels to matrix
    def labelInMatrix(self):
        self.matrix = []
        maxQ = max(self.labels)
        for q in range(maxQ+1):
            cash = [0] * (len(self.labels) - 1)
            indexs = [i for i,x in enumerate(self.labels) if x==q]
            for o in indexs:
                cash.insert(o,1)
            if len(cash)>len(self.labels):
                a = cash[:-(len(cash)-len(self.labels))]
                cash = a
            self.matrix.append(cash)

    # get Centroid indexs
    def getIndexCentroid(self):
        self.c = []
        for q in range(len(self.centroids)):
            cash = []
            for i in range(len(self.matrix[q])):
                if 1 == self.matrix[q][i]:
                    cash.append(i)
            self.c.append(cash)
    
    # calculate center
    def calculateCenter(self):
        self.centroids = []
        for q in range(len(self.c)):
            cash = []
            for l in range(len(self.data[0])):
                self.funcSumCenter(l,q)
                w = self.summ/self.matrix[q].count(1)
                cash.append(w)
            self.centroids.append(cash)

    def funcSumCenter(self,l,q):
        self.summ=0.0
        for i in range(len(self.data)):
            if i in self.c[q]:
                uiq = 1
            else:
                uiq = 0
            self.summ += self.data[i][l]*uiq

    # [0,1,0,1,0,1] ==> [[0,2,4],[1,3,5]]
    def clusteringSentence(self):
        self.cq = []
        cluster = max(self.labels)
        for q in range(cluster+1):
            cash = []
            for i, j in enumerate(self.labels):
                if j == q:
                    cash.append(i)
            self.cq.append(cash)

    # compute kMeans
    def fit(self, data, metric):
        self.metric = metric
        self.data = data

        # select center from centences
        self.selectCenterOfCluster()
        
        # covert centroid index to centroid value
        self.getCentroidValue()
        print("centroidsIndexs",self.centroidsIndexs)
        
        # Computing similirity between sentences
        self.computeSimilarityBetweenSentences()
        
        # find labels X = [x1,x2,x3,...,xn]
        self.findLabels()

        # toMatrix  Cq = [q1 -[1,0..,uiq], q2 -[], q3 -[]]
        # if s1 from q1 uiq = 1 else 0
        self.labelInMatrix()

        # get index  self.centroids
        # [[O1],[O2],[Oq]]
        self.getIndexCentroid()
        
        
        for i in range(self.max_iterations):
            previous_centroid = self.centroids
        
            # compute similarity between S1,S(random center)
            self.computeSimilarity()
            
            # clustering result
            self.findLabels()
            
            # toMatrix
            self.labelInMatrix()
                
            # get index
            self.getIndexCentroid()
            
            
            # average the cluster datapoints to re-calculate the centroids
            self.calculateCenter()
            # convert labels to Cq 
            #self.clusteringSentence()
            
            # check centroid value on equals
            comparison = np.array(previous_centroid) == np.array(self.centroids)
            equal_arrays = comparison.all()
            if equal_arrays == True:
                break
        self.y_means = self.labels