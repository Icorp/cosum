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
reference = "On Monday President Bush nominated Clarence Thomas, a conservative\
black Republican, to replace black liberal Justice Thurgood Marshall\
on the Supreme Court.  Conservatives quickly embraced the nomination\
while liberals responded either with caution or hostility. Aside from\
his strong opposition to affirmative action, Thomas's views on other\
major issues before the court are unknown. Although Thomas attended\
Roman Catholic schools and studied for the priesthood and the Catholic\
Church vigorously opposes abortion, Thomas has apparently never taken\
a public stand on abortion or the constitutional right to privacy. If\
Thomas remains silent on these issues his nomination is likely to be\
confirmed."


print("Start ...")
print("Reading document ...")
text = readText("dataset/data/training/d01a/docs/SJMN91-06184003")
sentences = sent_tokenize(text)

# l_avg
l_avg = len(word_tokenize(text))/len(sentences)

# sentences = sent_tokenize(text)
K = int(findK(text))

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
