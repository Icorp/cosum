# -*- coding: utf-8 -*-
import logging as log
import time
import findIt
import re
from ga import Ga
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from file import saveStats
from file import readText
from rouge_score import rouge_scorer
from utils import clusteringSentence
from utils import findK
from cosum import K_means
from cosum import CosumTfidfVectorizer


# Constants
reference = "Senators McClure (R) and Metzenbaum (D) have sponsored bills to\
prevent plastic guns from slipping through airport security.  The gun,\
not yet manufactured, is intended for military and police use.\
Metzenbaum's bill would require some detectable metal content;\
McClure's would require more sensitive detection equipment at\
airports, almost certainly causing passenger delays.\
The NRA opposes \"the first federal gun ban bill in America\", and warns\
members their guns will soon be inspected and weighed by government\
agents.\
However, on this issue they will compromise, not to ban the gun, but\
to increase airport security.  Handgun control advocates and law\
enforcement officials back Metzenbaum's proposal."


print("Start ...")
print("Reading document ...")
text = readText("training/AP880310-0257")
sentences = sent_tokenize(text)

# l_avg
l_avg = len(word_tokenize(text))/len(sentences)

# calculate how many clusters we need
K = int(findK(text))

# convert text to vector
vectorizer = CosumTfidfVectorizer()
vectorizer.fit(text)
vector = vectorizer.weight_matrix

# Computing centroids
print("Стартуем кластеризацию ...")
kmeans = K_means(K, max_iterations=100000)
for i in range(1000):
    try:
        kmeans.fit(vector, metric="similarity")
        break
    except ZeroDivisionError:
        print("Попытка #", i)
print("Класстеризация завершена ...")

# X = [1,2,4,1,2,3,5,7,2]  This is number of cluster
X = kmeans.labels
clusterSentence = clusteringSentence(X)


# Start compute genetic algorithm
print("Старт генетического алгоритма ...")
genetic = Ga(t_max=1000, pop_size=200)
genetic.fit(vector, kmeans.cq, kmeans.centroids, l_avg, clusterSentence, K)
print("Рассчет генетического алгоритма завершена ...")

# Convert genomes to normal summary text
indexs = genetic.best_summary

summary = ""
fx = genetic.F
for i in indexs:
    summary += (sentences[i])

print("\nЛучший набор предложении")
print(indexs)

# calculate Rouge
scores = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
scores = scores.score(summary, reference)

# save results to file
print(scores)
saveStats(fx, indexs, scores)
print("Finish!")
