import cosum
import numpy as np
import findIt
import rouge
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from evolution import Initialization
from evolution import Initialization_exp
from evolution import Vp
from evolution import inertia_weight
from evolution import sigmoid
from evolution import crossover
from sklearn.cluster import KMeans
from file import writeToFile
from file import readFile
from file import writeToJson
from optimize import stageOne
from optimize import stageTwo
from optimize import stageThree
from optimize import F
from optimize import startTest
from rouge import Rouge 
import sys

f = open('SJMN91-06184003.txt',"r",encoding="utf-8",)
f =f.read()
a = re.findall(r'(<TEXT>.+?</TEXT>)', f,flags=re.DOTALL)
a = a[0].replace("\n"," ")
a = a.replace(".;",".")
a = a.replace('<TEXT>','')
document2 = a.replace('</TEXT>','')
document = "The school system of Canada is very much like the one in the USA, but there are certain differences. Education in Canada is general and compulsory for children from 6 to 16 years old, and in some provinces — to 14. It is within the competence of the local authorities, and therefore it may differ from province to province. For example, Newfoundland has an 11-grade system.Some other provinces have 12-grade systems, and Ontario has even a 13-grade system. Grades 1—6 are usually elementary schools, and grades 7—12 are secondary schools. In some provinces there is a kindergarten year before the first grade. Elementary education is general and basic, but in the junior high school years the students can select some courses themselves. Most secondary schools provide programmes for all types of students. Some of them prepare students for continuing their studies at the university. Vocational schools are separate institutions for those who will not continue their education after secondary schools. There also exist some commercial high schools. Some provinces have private kindergartens and nursery schools for children of pre-elementary age. There also exist Roman Catholic schools and private schools in some provinces. In most provinces private schools receive some form of public support. Admission to the university in Canada is after high school with specific courses. Getting a degree in law, medicine, dentistry or engineering usually takes 3—4 years of studying. University tuition fees vary among different provinces. All provinces also have public non-university institutions. They are regional colleges, institutes of technology, institutes of applied arts, colleges of agricultural technology and others. Criteria for admission to these institutions are less strict.The educational system in Kazakhstan is conducted in two languages - Kazakh and Russian and consists of several levels of state and private educational establishments: infant schools, elementary (or primary) schools, comprehensive schools, colleges and academies. The constitution of the Republic of Kazakhstan fixes the right of citizens of the republic on free-of-charge secondary education which is obligatory. The constitution prohibits any discrimination on the basis of language or ethnicity and guarantees equal rights in education regardless of nationality. Children start school at the age of 7 and finish at 17. As a rule a child attends the school, located in the neighborhood. However, in big cities there are so-called special schools, offering more in depth studies of the major European languages (English, French, German) or the advanced courses in physics and mathematics and children, attending one of this may have to commute from home."
Sentences = sent_tokenize(document2)
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
random_s = cosum.randomizer_6(clusters)
#random_s = [[43,41],[14,36],[29,25]]
print("Были выбраны предложения с индексами : ",random_s)

summary_index = cosum.get_summary(random_s,Sentences)
print(summary_index)
summary = ''.join(summary_index)
print("random = >",random_s)
hypothesis = startTest(random_s,summary_index,document2,X,O,clusters,summary)


#print(X)
#for i in range(1000):
t_max = 5
population = 2
random_all_fx = []
random_best_fx = []
random_all_sent = []
def start():
    result = []
    
    # Создаем популяции
    for p in range(population):   
        optimize = []          # Лист для хранения fx()
        sent = []              # Лист для хранения индексов предложения        
        random_s = 0.0          # Лист для хранения вектора
        vector = list
        for m in range(t_max):
            random_s,vector = Initialization_exp(X)
            fx = F(X,O,Sentences,random_s)
            optimize.append(fx)
            sent.append(random_s)
            random_all_sent.append(vector)
        random_all_fx.append(optimize)
        # Создаем особь для популяции. Количество задано в t_max
        for t in range(t_max):
            print("Популяция #",p+1)
            print("Особь #",t+1,"\n")
            best_local = max(optimize)          # Находим максимальное значение f(x), среди t генерации.Это будет best_local  
            print("Best_local",best_local)
            all_fx_mixed = []
            # Проверяем на пустату популяции если пусто global = local
            if len(random_all_fx)==0:
                best_global = max(optimize)
            else:
                print("Search GLOBAL BEST ....")
                all_fx_mixed = cosum.mix(random_all_fx)
                print("ALL fx => ",all_fx_mixed)
                best_global = max(all_fx_mixed)
                print("Best global fx(X)",best_global)
                index_g = all_fx_mixed.index(best_global)
                print("Index of the best global",index_g)


            # Вытаскиваем индексы лучших предложении ,из предложении sent.
            index_l = optimize.index(best_local)
            #print("GLOBAL Index",random_all_sent)
            #print("LOCAL Index",random_all_sent)
            
            # Пытаемся мутировать лучший вариант
            result.append(Vp(random_all_sent[index_g],random_all_sent[index_l],random_all_sent[t],t))
            #crossover(resukt)
    return result


reference = "Clarence Thomas, black nominee to the Supreme Court, comes from a childhood in segregated Georgia where he was taught the ethic of hard work, self discipline, and independence. He went north to college at Holy Cross. By the time he entered Yale Law School he was described as a freewheeling liberal. Brilliant conservative law professors and the writings of a conservative black economist had a profound influence on Thomas.  In his service in various state and federal positions and on the U.S. Court of Appeals he has been a proponent of personal strength over dependence and individualism over government activism."

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)

