import sys
import cosum
import logging as log
import random
from cosum import computeSimClustering
from cosum import computeAllWeightOfDocument
from cosum import labelInMatrix
from cosum import selectSentences
from cosum import computeMatrixSimRRN
from cosum import mix
from formulation import funcSum4
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

l_max = 100
class objective:
    def randomizer_cluster(self):
        self.random_sentences = []
        for i in range(len(self.Cq)):
            sampling = random.sample(self.Cq[i], 1)
            self.random_sentences.append(sampling)

    # O - it is cluster centers. O [[],[],[]]
    # x - [[2,3,4,5],[5,23,46,5],[...]]
    # w - weight of sentences. [[w11..w1n],[w21..w2n],[...]]
    def compute_f_cover(self):
        self.f_cover = 0
        k = len(self.matrix)
        for q in range(k):
            n = len(self.matrix[q])
            for i in range(n):
                self.f_cover += round(computeSimClustering(self.data[i],self.O[q])*self.matrix[q][i],3)

    #   The second term f diver (X) minimizes the sum of intersentence similarities among sentences chosen from each cluster.
    #   Equation (14)
    #   X - index_sentences [[2,3,4,5],[5,23,46,5],[...]]
    #   w - weight of sentences. [[w11..w1n],[w21..w2n],[...]]
    def compute_f_diver(self):
        self.f_diver = 0
        k = len(self.matrix)
        summ = 0
        for q in range(k):
            nq = len((self.matrix[q]))
            summ += funcSum4(nq,self.data,q,self.matrix)
            self.f_diver += round((2/nq*(nq-1))*summ,3)
    
    def computeObjectiveFunction(self):
        self.compute_f_cover()
        self.compute_f_diver()
        self.F = round((2*self.f_cover*self.f_diver)/(self.f_cover+self.f_diver),3)
        count = 0
        for listElem in self.random_sentences:
            count += len(listElem)
        print("Количество предложении =",count)
    
    def get_summary(self):
        self.summary_index = []
        for i in range(len(self.random_sentences)):
            for k in range(len(self.random_sentences[i])):
                self.summary_index.append(self.sentences[self.random_sentences[i][k]])
    
    # This stage check all statements. 
    # li*x[i][q] <= Lavg
    # li*x[i][q] <= Lmax
    # Lmax = 100
    # lavg = number of words / number of sentences
    def stageOne(self):
        # I Находим l cредний = количество слов, деленное на количество предложений
        sentences = sent_tokenize(self.document)
        self.stage_one_result = []
        self.summ = 0
        l_avg = round(len(word_tokenize(self.document))/len(sentences),3)
        log.info("\nПроверка I. Меньше ли длина предложения чем средняя длина ???")
        log.info('Средняя длина (l_avg) = >',l_avg)
        log.info("Summary = ",self.summary_index)
        for k in range(len(self.summary_index)):
            tokens = word_tokenize(self.summary_index[k])
            num_of_word = len(tokens)
            self.summ +=num_of_word
            if(num_of_word)<=l_avg:
                self.stage_one_result.append(True)
            else:
                self.stage_one_result.append(False)
            log.info("Количество слов в {} предложении".format(k+1),num_of_word)

    def stageTwo(self):
        self.stage_two_result = False
        log.info("\nПроверка II. Количество слов в вашем summary должно быть меньше l_max = 100  ???")
        log.info("Всего слов => ",)
        if self.summ <=l_max:
            log.info("Результат: Успешный!")
            self.stage_two_result = True
        else:
            log.warning("Результат: Не проходит!")
            self.stage_two_result = False
    
    # Compute F(X).
    def stageThree(self):
        self.computeObjectiveFunction()
        print("F(x) = ",self.F)
        if self.F > 0:
            print("All Good")
            self.summary
            self.F
        else:
            print("F(x) not working")
    
    def start(self, data, Cq, labels, document, O, matrix):
        self.O = O
        self.document = document
        self.sentences = sent_tokenize(document)
        self.data = data
        self.Cq = Cq
        self.matrix = matrix
        while True:
            self.randomizer_cluster()
            self.get_summary()
            print(self.summary_index)
            self.summary = ''.join(self.summary_index)
            print("Были выбраны предложения с индексами : ",self.random_sentences)
            self.stageOne()
            print("Результат(I) :",self.stage_one_result,"\n")
            if all(self.stage_one_result) == True:
                self.stageTwo()
                if self.stage_two_result == True:
                    self.stageThree()
                    break
                else:
                    print("Stage II: No")
            else:
                print("Stage I: NO")

