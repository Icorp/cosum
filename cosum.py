import math
import re
import sys
import random
import logging as log
import numpy as np
import time
from nltk import sent_tokenize
from nltk import word_tokenize
from time import perf_counter
from findIt import findNumSentencesSearchingWord
from findIt import finalTokenSentence
from findIt import final_token
from utils import funcSumCenter
from utils import computeSimilarity
#from utils import euclidian

# Test
from nltk.corpus import stopwords

# This is vectorizer. Todo: write documentation ...
class CosumTfidfVectorizer:
    def unionTokens(self):
        self.vocabulary = sorted(set(self.vocabulary), key=self.vocabulary.index)
        
    
    def computeTF(self, word, i):
        indices = [i for i, x in enumerate(self.sentences_in_words[i]) if x == word]
        self.TF = len(indices)        
    
            
    def findNumSentencesSearchingWord(self, word):
        self.nj = 0
        for i in range(len(self.sentences_in_words)):
            if word in self.sentences_in_words[i]:
                self.nj += 1

    # Считает количество слов в предложении. На хвод принимает предложение, удаляет стоп слова. Sentence = "word word word"
    # Example:
    #   compute(self," I will be better tommorrow")
    #   Output: 2

    def computeLi(self, i):
        self.l_i = len(self.sentences_in_words[i])
    
    

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
        self.sentences_in_words = []
        for i in range(len(self.sentences)):
            self.sentences_in_words.append(finalTokenSentence(self.sentences[i]))
    

    # Метод считает вес всех слов.
    def computeFullWeight(self):
        self.weight_matrix = []
        self.computeAverageSentenceLength()
        for i in range(len(self.sentences)):
            cash = []
            for m in range(len(self.filter_words)):
                self.computeTDIDF(self.filter_words[m],i)
                cash.append(self.weight)
            #print("Time:",round(time.perf_counter(),3)," sec")
            self.weight_matrix.append(cash)


    def fit(self,document):
        self.document = document

        # 0 step: get sentences
        self.sentences = sent_tokenize(document)

        # 1 step: get filter_words
        self.filter_words = final_token(document)

        # 2 step: get unfilter_words
        self.vocabulary = word_tokenize(document)

        # 3 step: Stemming
        self.convertSentenceToWords()

        # 3 step: calculate weight
        self.computeFullWeight()

class K_means:
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
        """
        if self.metric == "euclidean":
            self.similarities = []
            for i in range(len(self.data)):
                cash = []
                for k in range(self.k):
                    cash.append(euclidian(self.data[i],self.centroids[k]))
                self.similarities.append(cash)
        else:
        """
        self.similarities = []
        for i in range(len(self.data)):
            cash = []
            for k in range(self.k):
                cash.append(computeSimilarity(self.data[i],self.centroids[k])*self.matrix[k][i])
            self.similarities.append(cash)
        
    # compute similirity between centroid and sentences 
    def computeSimilarityBetweenSentences(self):
        self.similarities = []
        for i in range(len(self.data)):
            cash = []
            for k in range(len(self.centroidsIndexs)):
                cash.append(computeSimilarity(self.data[i],self.data[self.centroidsIndexs[k]]))
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
    
    def convertLabelInMatrix(self):
        self.matrix = []
        for q in range(self.k):
            cash = []
            for i in range(len(self.labels)):
                if self.labels[i] == q:
                    cash.append(1)
                else:
                    cash.append(0)
            self.matrix.append(cash)

    # calculate center
    def calculateCenter(self):
        self.centroids = []
        for q in range(len(self.cq)):
            cash = []
            for l in range(len(self.data[0])):
                self.funcSumCenter(l,q)
                w = self.summ/len(self.cq[q])
                cash.append(w)
            self.centroids.append(cash)

    def funcSumCenter(self,l,q):
        self.summ=0.0
        for i in range(len(self.data)):
            if i in self.cq[q]:
                uiq = 1
            else:
                uiq = 0
            self.summ += self.data[i][l]*uiq

    # [0,1,0,1,0,1] ==> [[0,2,4],[1,3,5]]
    def clusteringSentence(self):
        self.cq = []
        for q in range(self.k):
            cash = []
            for i, j in enumerate(self.labels):
                if j == q:
                    cash.append(i)
            self.cq.append(cash)

    # compute kMeans
    def fit(self, data, metric):
        # If NullCluster is true. All clusters have some elements, else NullCluster is False on of cluster doesnt have value.
        self.NullCluster = False
        self.metric = metric
        self.data = data
        while self.NullCluster != True:
            # select center from centences
            self.selectCenterOfCluster()
            
            # covert centroid index to centroid value
            self.getCentroidValue()
            
            # Computing similirity between sentences
            self.computeSimilarityBetweenSentences()
            
            # find labels X = [x1,x2,x3,...,xn]
            self.findLabels()

            # toMatrix  Cq = [q1 -[1,0..,uiq], q2 -[], q3 -[]]
            # if s1 from q1 uiq = 1 else 0
            self.convertLabelInMatrix()

            # get index  self.centroids
            # [[O1],[O2],[Oq]]
            self.clusteringSentence()

            isNull = []
            for i in range(self.k):
                if len(self.cq[i])>1:
                    isNull.append(True)
                else:
                    isNull.append(False)
            if all(isNull):
                self.NullCluster = True

        
        
        print("Центроиды:",self.centroidsIndexs)
        for i in range(self.max_iterations):
            previous_centroid = self.centroids
        
            # compute similarity between S1,S(random center)
            self.computeSimilarity()
            
            # clustering result
            self.findLabels()
            
            # toMatrix
            self.convertLabelInMatrix()
                
            # get index
            self.clusteringSentence()
            
            # average the cluster datapoints to re-calculate the centroids
            
            self.calculateCenter()
            # convert labels to Cq 
            
            # check centroid value on equals
            try:
                comparison = np.array(previous_centroid) == np.array(self.centroids)
                equal_arrays = comparison.all()
                print(comparison)
            except AttributeError:
                print(comparison)
                sys.exit()
            if equal_arrays == True:
                break
        self.y_means = np.array(self.labels)