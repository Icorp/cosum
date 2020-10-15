import cosum
import numpy as np
import findIt
import rouge
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.cluster import KMeans
from file import writeToFile
from file import readFile
from file import saveStats
from file import readText
from optimize import stageOne
from optimize import stageTwo
from optimize import stageThree
from optimize import F
from optimize import startTest
from rouge import Rouge 
import sys

document = readText("training/AP880310-0257")
Sentences = sent_tokenize(document)
#data = cosum.computeMatrixSimRRN(document2)
#writeToFile(data,"Sim")
data = np.array(readFile())
kmeans = KMeans(n_clusters=3,random_state=42).fit(data)

X = cosum.labelInMatrix(kmeans.labels_)
O = kmeans.cluster_centers_
#S = cosum.computeAllWeightOfDocument(document)

#  S - is weight of all words in the document
# S = [[w1,w2,...,wn],
#      [w1,w2,...,wn]          
#      ]
#S = cosum.computeAllWeightOfDocument(document)
arr = kmeans.labels_.tolist()
clusters = cosum.clusteringSentence(arr)
print("Cq = ",clusters,"\n")

hypothesis,fx,indexs,summary = startTest(clusters,document,X,O,clusters,Sentences)


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
