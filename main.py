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
from file import saveStats
from file import readText
from optimize import stageOne
from optimize import stageTwo
from optimize import stageThree
from optimize import F
from optimize import startTest
from rouge import Rouge 
import sys

document2 = readText("training/AP880310-0257")
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



hypothesis,fx,indexs,summary = startTest(clusters,document2,X,O,clusters,Sentences)


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


reference = "Senators McClure (R) and Metzenbaum (D) have sponsored bills to prevent plastic guns from slipping through airport security.  The gun, not yet manufactured, is intended for military and police use. Metzenbaum's bill would require some detectable metal content; McClure's would require more sensitive detection equipment at airports, almost certainly causing passenger delays. The NRA opposes the first federal gun ban bill in America, and warns members their guns will soon be inspected and weighed by government agents. However, on this issue they will compromise, not to ban the gun, but to increase airport security.  Handgun control advocates and law enforcement officials back Metzenbaum's proposal."

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)
print(hypothesis)
print("Saving file ...")
saveStats(hypothesis,fx,indexs,summary,scores)
print("Finish!")
