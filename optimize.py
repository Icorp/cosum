import numpy as np
from cosum import computeSimClustering
from cosum import computeAllWeightOfDocument
from cosum import labelInMatrix
from cosum import selectSentences
from cosum import computeMatrixSimRRN
from cosum import mix
from formulation import funcSum4
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans

l_max = 100

#Our objective function combines two objectives of coverage f cover (X) (relevance of a summary is the amount of relevant information the summary
#contains) and diversity f diver (X) (summary should not contain multiple sentences that convey the same information):
def F(X,O,sentences,random):
    index_sentences = mix(random)
    final_sentences = selectSentences(index_sentences,sentences)
    text = ' '.join(final_sentences)
    count = 0
    for listElem in random:
        count += len(listElem)   
    print("Количество предложении =",count)
    w = computeAllWeightOfDocument(text)
    f_cov = f_cover(O,random,w)
    #print("F_COVER =",f_cov)
    f_div = f_diver(random,w)
    #print("F_DIVER =",f_div)
    return round((2*f_cov*f_div)/(f_cov+f_div),3)

# O - it is cluster centers. O [[],[],[]]
# x - [[2,3,4,5],[5,23,46,5],[...]]
# w - weight of sentences. [[w11..w1n],[w21..w2n],[...]]
def f_cover(O,x,w):
    summ = 0
    k = len(x)
    for q in range(k):
        n = len(x[q])
        for i in range(n):
            summ +=computeSimClustering(w[i],O[q])*x[q][i]
    return round(summ,3)

# X - index_sentences [[2,3,4,5],[5,23,46,5],[...]]
# w - weight of sentences. [[w11..w1n],[w21..w2n],[...]]
def f_diver(X,w):
    k = len(X)
    summ = 0
    result = 0   
    for q in range(k):
        nq = len((X[q]))
        summ += funcSum4(nq,w,q,X)
        result += (2/nq*(nq-1))*summ
    return round(result,3)

# This stage check all statements. 
# li*x[i][q] <= Lavg
# li*x[i][q] <= Lmax
# Lmax = 100
# lavg = number of words / number of sentences
def stageOne(x,document):
    sentences = sent_tokenize(document)
    l_avg = len(word_tokenize(document))/len(sentences)
    num_of_word = 0
    for k in range(len(sentences)):
        num_of_word += len(word_tokenize(sentences[k]))
    result = []
    for q in range(len(x)):
        for i in range(len(x[q])):
            l_i = len(word_tokenize(sentences[i]))
            if (l_i*x[q][i])<=l_avg:
                result.append(True)
            else:
                result.append(False)
        
        for i in range(len(x[q])):
            if num_of_word <= l_max:
                result[i] == True
            else:
                result[i] == False
    return result

# If all element in array are True. Go to next method stageThree()
def stageTwo(status,X,O,sentences,random):
    if all(status) == True:
        stageThree(X,O,sentences,random)
    else:
        return "Не соответствует требованиям"

# Compute F(X).
def stageThree(X,O,sentences,random):
    result = F(X,O,sentences,random)
    print("F(x) = ",result)

# Main function for check all stage
def optimize(X,O,sentences,random,document):
    result_stage_one = stageOne(random,document)
    if all(result_stage_one) == True:
        stageThree(X,O,sentences,random)
    else:
        return "Не соответствует требованиям"
