import numpy as np
import sys
import cosum
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
def F(X,O,sentences,random,Cq):
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
    f_div = f_diver(random,w,Cq)
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
def f_diver(X,w,Cq):
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
def stageOne(x,summary,document):
    # I Находим l cредний = количество слов, деленное на количество предложений
    sentences = sent_tokenize(document)
    result = []
    l_avg = round(len(word_tokenize(document))/len(sentences),3)
    print("\nПроверка I. Меньше ли длина предложения чем средняя длина ???")
    print("Средняя длина (l_avg) = >",l_avg)
    summ = 0
    print("Summary = ",summary)
    for k in range(len(summary)):
        tokens = word_tokenize(summary[k])
        num_of_word = len(tokens)
        summ +=num_of_word
        if(num_of_word)<=l_avg:
            result.append(True)
        else:
            result.append(False)
        print("Количество слов в {} предложении".format(k+1),num_of_word)
        print("Tokens:",tokens,"\n\n    ")
    
    return result,summ
def stageTwo(li):
    print("\nПроверка II. Количество слов в вашем summary должно быть меньше l_max = 100  ???")
    print("Всего слов => ",li)
    if li <=l_max:
        print("Результат: Успешный!")
        return True
    else:
        print("Результат: Не проходит!")
        return False
    
    


# Compute F(X).
def stageThree(X,O,sentences,random,Cq,summary):
    result = F(X,O,sentences,random,Cq)
    print("F(x) = ",result)
    if result > 0:
        return summary,result
    else:
        return "F(x) not working"

def startTest(clusters,document,X,O,Cq,Sentences):
    sentences = sent_tokenize(document)
    
    while True:
        
        random_s = cosum.randomizer_6(clusters)
        summary_index = cosum.get_summary(random_s,Sentences)
        print(summary_index)
        summary = ''.join(summary_index)
        print("random = >",random_s)
        #random_s = [[43,41],[14,36],[29,25]]
        print("Были выбраны предложения с индексами : ",random_s)
        st_1,summ = stageOne(random_s,summary_index,document)
        print("Результат(I) :",st_1,"\n")
        if all(st_1) == True:
            st_2 = stageTwo(summ)
            if st_2 == True:
                a,b = stageThree(X,O,sentences,random_s,Cq,summary) 
                return a,b,random_s,summary
                break
            else:
                print("Stage II: No")
        else:
            print("Stage I: NO")
    """
    if all(st_1) == True:
        st_2 = stageTwo(summ)
        if st_2 == True:
            a = stageThree(X,O,sentences,random_s,Cq,summary)
            return a
    
    else:
        print("Не прошел первый этап !!!")
        sys.exit()
    """    
    