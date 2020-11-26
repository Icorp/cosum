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
text2 = "My best friends name's Misha. We made friends a few years ago. We are of the same age. We live in the same block of flats, so we see each other almost every day. Misha is a tall slender boy. He has got dark hair, large dark eyes, a straight nose, and thin lips. He wears spectacles. Misha is a nice guy. He is very honest and just, understanding and kind. I trust him a lot, and I'm sure that I can rely on him in any situation. He never lets people down. Misha is only 16 but he is very responsible — he finishes whatever he starts. He's got only one shortcoming. He is a bit stubborn; nevertheless he is pleasant to deal with. Misha's an only child and his parents love him very much. His father is a lawyer. He is the most brilliant man I've ever met. He knows everything there is to know about the law. His mother is a music teacher. No wonder Michael is so talented. He's got a very good ear for music. He likes jazz and plays the piano very well. We spend a lot of time together. We often watch video or listen to music. Sometimes we go to theatre, or walk around the centre of the city, visiting small cafes, museums, art galleries, and shops. We talk for hours about all sorts of things (politics, love, teachers, and girls). We discuss films, television programmes, books. I never quarrel with Misha. But if there is some misunderstanding between us we try to make peace as soon as possible.We are of the same age. We live in the same block of flats, so we see each other almost every day. My best friends name's Petrov. What I like best about him is that he is always willing to help and share his knowledge, thoughts, and feelings. I respect him for his fairness, strong will, intellect, and modesty. I miss Misha when we don't see each other for a long time. Without him I would feel lonely and uncomfortable. Our friendship helps me feel strong and sure of myself."

# sentences = sent_tokenize(text)
K = int(cosum.findK(text2))





vectorizer = CosumTfidfVectorizer()
vectorizer.fit(text2)
vector = vectorizer.weight_matrix



#vector = cosum.computeFullWeight(text)
writeToFile(vector)
# Reading data of weight from file
#vector = np.array(readFile())

# Computing centroids
print("Computing centroids ...")
kmeans = k_means(3,max_iterations=100000)
kmeans.fit(vector)

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