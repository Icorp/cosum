import logging as log
import random
from cosum import K_means
from utils import computeSimilarity
from utils import funcSum4
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import sys

l_max = 100
class Objective(K_means):
    
    # O - it is cluster centers. O [[],[],[]]
    # x - [[2,3,4,5],[5,23,46,5],[...]]
    # w - weight of sentences. [[w11..w1n],[w21..w2n],[...]]
    def compute_f_cover(self):
        self.f_cover = 0
        for q in range(self.K):
            for i in self.clusterSentence[q]:
                self.f_cover += computeSimilarity(self.data[i],self.O[q])

    #   The second term f diver (X) minimizes the sum of intersentence similarities among sentences chosen from each cluster.
    #   Equation (14)
    #   X - index_sentences [[2,3,4,5],[5,23,46,5],[...]]
    #   w - weight of sentences. [[w11..w1n],[w21..w2n],[...]]
    def compute_f_diver(self):
        self.f_diver = 0
        summ = 0
        for q in range(self.K):
            nq = len(self.clusterSentence[q])
            summ += funcSum4(nq,self.data,q,self.clusterSentence, self.genomes)
            self.f_diver += (2/nq*(nq-1))*summ
    
    def computeObjectiveFunction(self):
        self.F = 0
        self.compute_f_cover()
        self.compute_f_diver()
        self.F = (2*self.f_cover*self.f_diver)/(self.f_cover+self.f_diver)
    
    def Fx(self, data, genomes, cq, centroids, clusterSentence, K):
        self.K = K
        self.clusterSentence = clusterSentence
        self.O = centroids
        self.cq = cq
        self.data = data
        self.genomes = genomes  # sentences
        self.computeObjectiveFunction()
        self.printRandomSentences()
    # This stage check all statements. 
    # li*x[i][q] <= Lavg
    # li*x[i][q] <= Lmax
    # Lmax = 100
    # lavg = number of words / number of sentences

    def printRandomSentences(self):
        self.selected_sentences = []
        for i in range(len(self.genomes)):
            if self.genomes[i] == 1:
                self.selected_sentences.append(i)
        #print("Количество предложении =",self.selected_sentences,"\n")