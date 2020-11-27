# -*- coding: utf-8 -*-
import cosum
import numpy as np
import logging as log
import time
import findIt
import rouge
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords 
from sklearn.pipeline import Pipeline
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.cluster import KMeans
from file import writeToFile
from file import readFile
from file import saveStats
from file import readText
from rouge import Rouge 
import matplotlib.pyplot as plt
import sys
from cosum import k_means
from cosum import CosumTfidfVectorizer
from optimize import objective

print("Start ...")
print("Reading document ...")
text = readText("training/AP880310-0257")

# sentences = sent_tokenize(text)
K = int(cosum.findK(text))





vectorizer = CosumTfidfVectorizer()
vectorizer.fit(text)
vector = vectorizer.weight_matrix



#vector = cosum.computeFullWeight(text)
writeToFile(vector)
# Reading data of weight from file
#vector = np.array(readFile())

# Computing centroids
print("Computing centroids ...")
kmeans = k_means(3,max_iterations=100000)
kmeans.fit(vector,metric="similarity")
U = kmeans.similarities
sys.exit()
# X = [1,2,4,1,2,3,5,7,2]  This is number of cluster 
X = kmeans.labels

# O = [[centroid value],[centroid value],[centroid value]] this it centroids of cluster
O = kmeans.centroids

# U = U_iq    i - is index of sentences, q - is index of cluster
# U = [[1,0,0,0,1], [0,1,1,1,0], [...] ]
U = kmeans.matrix


Cq = kmeans.c
objectives = objective()
objectives.start(vector, Cq, X, text, O, U)
print(objectives.F)
hypothesis = objectives.summary
fx = objectives.F
indexs = objectives.random_sentences
summary = objectives.summary


#print(X)
#for i in range(1000):
t_max = 5
population = 2
random_all_fx = []
random_best_fx = []
random_all_sent = []


reference = "Senators McClure (R) and Metzenbaum (D) have sponsored bills to prevent plastic guns from slipping through airport security.  The gun, not yet manufactured, is intended for military and police use. Metzenbaum's bill would require some detectable metal content; McClure's would require more sensitive detection equipment at airports, almost certainly causing passenger delays. The NRA opposes the first federal gun ban bill in America, and warns members their guns will soon be inspected and weighed by government agents. However, on this issue they will compromise, not to ban the gun, but to increase airport security.  Handgun control advocates and law enforcement officials back Metzenbaum's proposal."

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)
print(hypothesis)
saveStats(hypothesis,fx,indexs,summary,scores)
print("Finish!")