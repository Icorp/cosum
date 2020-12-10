# -*- coding: utf-8 -*-
import logging as log
import time
import findIt
import re
from ga import Ga
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from file import saveStats
from file import readText
from rouge_score import rouge_scorer
import sys
from utils import clusteringSentence
from utils import findK
from cosum import K_means
from cosum import CosumTfidfVectorizer

print("Start ...")
print("Reading document ...")
text = readText("training/AP880310-0257")
sentences = sent_tokenize(text)

# l_avg 
l_avg = len(word_tokenize(text))/len(sentences)

# sentences = sent_tokenize(text)
K = int(findK(text))

vectorizer = CosumTfidfVectorizer()
vectorizer.fit(text)
vector = vectorizer.weight_matrix

# Computing centroids
print("Computing centroids ...")
kmeans = K_means(K,max_iterations=100000)
for i in range(1000):
    try:
        kmeans.fit(vector,metric="similarity")
        break
    except ZeroDivisionError:
        print("Попытка #",i)

print("Класстеризация завершена...")

# X = [1,2,4,1,2,3,5,7,2]  This is number of cluster 
X = kmeans.labels

clusterSentence = clusteringSentence(X)

# O = [[centroid value],[centroid value],[centroid value]] this it centroids of cluster
O = kmeans.centroids

# U = U_iq    i - is index of sentences, q - is index of cluster
# U = [[1,0,0,0,1], [0,1,1,1,0], [...] ]
U = kmeans.matrix

Cq = kmeans.cq
print("Старт генетического алгоритма")
genetic = Ga()
genetic.fit(vector, Cq, O, l_avg, clusterSentence, K)

indexs = genetic.best_genome
hypothesis = ""
fx = genetic.F
for i in indexs:
    hypothesis+=(sentences[i])

reference = "Text summarization is a process of extracting salient information from a source text \
and presenting that information to the user in a condensed form while preserving \
its main content. In the text summarization, most of the difficult problems are providing wide topic coverage and diversity in a summary. Research based on clustering, \
optimization, and evolutionary algorithm for text summarization has recently shown \
good results, making this a promising area. In this paper, for a text summarization, a\
two‐stage sentences selection model based on clustering and optimization techniques, called COSUM, is proposed. At the first stage, to discover all topics in a text,\
the sentences set is clustered by using k‐means method. At the second stage, for \
selection of salient sentences from clusters, an optimization model is proposed. This \
model optimizes an objective function that expressed as a harmonic mean of the \
objective functions enforcing the coverage and diversity of the selected sentences \
in the summary. To provide readability of a summary, this model also controls length  \
of sentences selected in the candidate summary. For solving the optimization problem, an adaptive differential evolution algorithm with novel mutation strategy is \
developed. The method COSUM was compared with the 14 state‐of‐the‐art methods: \
DPSO‐EDASum; LexRank; CollabSum; UnifiedRank; 0–1 non‐linear; query, cluster, \
summarize; support vector machine; fuzzy evolutionary optimization model; conditional random fields; MA‐SingleDocSum; NetSum; manifold ranking; ESDS‐GHS‐ \
GLO; and differential evolution, using ROUGE tool kit on the DUC2001 and \
DUC2002 data sets. Experimental results demonstrated that COSUM outperforms \
the state‐of‐the‐art methods in terms of ROUGE‐1 and ROUGE‐2 measures." 

reference2 = "Senators McClure (R) and Metzenbaum (D) have sponsored bills to prevent plastic guns from slipping through airport security.  The gun, not yet manufactured, is intended for military and police use. Metzenbaum's bill would require some detectable metal content; McClure's would require more sensitive detection equipment at airports, almost certainly causing passenger delays. The NRA opposes the first federal gun ban bill in America, and warns members their guns will soon be inspected and weighed by government agents. However, on this issue they will compromise, not to ban the gun, but to increase airport security.  Handgun control advocates and law enforcement officials back Metzenbaum's proposal."


scores = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scores.score(hypothesis,reference2)
print(scores)
saveStats(fx,indexs,scores)
print("Finish!")