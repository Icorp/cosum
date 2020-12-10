# -*- coding: utf-8 -*-
import logging as log
import time
import findIt
import re
from ga import Ga
import random
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
from file import saveRandomSummary


t_max = 100000
# Случайно число для количество выбора предложении
def randomNumberSentence(s):
    return 10
    #return random.randint(1,s/2)

# Cлучайные предложения In: Нужное количество предложении, Максимальные номер предложения
def randomSentence(number_of_sentence,max_n):
    result = []
    while len(result) != number_of_sentence:
        rand = random.randint(0,max_n-1)
        if rand in result:
            continue
        else:
            result.append(rand)
    return result

# Вышитывает rouge
def computeRouge(summary, reference):
    scores = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scores.score(summary,reference)
    return scores.get('rouge1').recall

# Массив с количествами слов
def calculate_words(data):
        result = []
        for i in range(len(data)):
            counter = 0
            for m in range(len(data[i])):
                if data[i][m]>0:
                    counter+=1
            result.append(counter)
        return result

# Находит лучший rouge
def computeBestRouge(reference,data, t_max, max_n, num_sentence, sentences,l_avg):
    prev_rouge = 0
    li = calculate_words(data)
    for t in range(t_max):
        indexs = randomSentence(num_sentence,max_n)
        
        # Конвертация индексов в предложения
        summary = ""
        new_li = 0
        isLavg = []
        for i in indexs:
            summary+=(sentences[i])
            new_li+=li[i]
            if li[i]<l_avg:
                isLavg.append(True)
            else:
                isLavg.append(False)
        
        

        # Вычисляем Rouge
        rouge = computeRouge(summary, reference)
        calculate_words(data)

        # Проверка лучше чем предыдущий rouge?
        if rouge > prev_rouge and new_li < 100 and all(isLavg) == True:
            prev_rouge = rouge
            my_string = "Попытка = {}\t Rouge = {}\t Количество слов = {}\t"
            print(my_string.format(t,rouge,new_li))

print("Start ...")
print("Reading document ...")
text = readText("training/AP880310-0257")
reference = "Senators McClure (R) and Metzenbaum (D) have sponsored bills to prevent plastic guns from slipping through airport security.  The gun, not yet manufactured, is intended for military and police use. Metzenbaum's bill would require some detectable metal content; McClure's would require more sensitive detection equipment at airports, almost certainly causing passenger delays. The NRA opposes the first federal gun ban bill in America, and warns members their guns will soon be inspected and weighed by government agents. However, on this issue they will compromise, not to ban the gun, but to increase airport security.  Handgun control advocates and law enforcement officials back Metzenbaum's proposal."
sentences = sent_tokenize(text)
max_n = len(sentences)

l_avg = len(word_tokenize(text))/len(sentences)

# STAGE 1: Случайно количество предложении
num_sentence = randomNumberSentence(max_n)

print("Нужно {} количество предложении..".format(num_sentence))

vectorizer = CosumTfidfVectorizer()
vectorizer.fit(text)
vector = vectorizer.weight_matrix

print("Вычисляем rouge")
# STAGE 2: Вычисливаем rouge
computeBestRouge(reference,vector,t_max,max_n,num_sentence,sentences,l_avg)

print("Finish!")