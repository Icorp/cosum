import cosum
import numpy as np
import findIt
import rouge
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from evolution import Initialization
from evolution import Initialization_exp
from evolution import Vp
from evolution import inertia_weight
from evolution import sigmoid
from sklearn.cluster import KMeans
from file import writeToFile
from file import readFile
from file import writeToJson
from optimize import stageOne
from optimize import stageTwo
from optimize import stageThree
from optimize import F

document = "The school system of Canada is very much like the one in the USA, but there are certain differences. Education in Canada is general and compulsory for children from 6 to 16 years old, and in some provinces — to 14. It is within the competence of the local authorities, and therefore it may differ from province to province. For example, Newfoundland has an 11-grade system.Some other provinces have 12-grade systems, and Ontario has even a 13-grade system. Grades 1—6 are usually elementary schools, and grades 7—12 are secondary schools. In some provinces there is a kindergarten year before the first grade. Elementary education is general and basic, but in the junior high school years the students can select some courses themselves. Most secondary schools provide programmes for all types of students. Some of them prepare students for continuing their studies at the university. Vocational schools are separate institutions for those who will not continue their education after secondary schools. There also exist some commercial high schools. Some provinces have private kindergartens and nursery schools for children of pre-elementary age. There also exist Roman Catholic schools and private schools in some provinces. In most provinces private schools receive some form of public support. Admission to the university in Canada is after high school with specific courses. Getting a degree in law, medicine, dentistry or engineering usually takes 3—4 years of studying. University tuition fees vary among different provinces. All provinces also have public non-university institutions. They are regional colleges, institutes of technology, institutes of applied arts, colleges of agricultural technology and others. Criteria for admission to these institutions are less strict.The educational system in Kazakhstan is conducted in two languages - Kazakh and Russian and consists of several levels of state and private educational establishments: infant schools, elementary (or primary) schools, comprehensive schools, colleges and academies. The constitution of the Republic of Kazakhstan fixes the right of citizens of the republic on free-of-charge secondary education which is obligatory. The constitution prohibits any discrimination on the basis of language or ethnicity and guarantees equal rights in education regardless of nationality. Children start school at the age of 7 and finish at 17. As a rule a child attends the school, located in the neighborhood. However, in big cities there are so-called special schools, offering more in depth studies of the major European languages (English, French, German) or the advanced courses in physics and mathematics and children, attending one of this may have to commute from home."
document2 = "Natural-language processing (NLP) is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages Natural-language."

Sentences = sent_tokenize(document)
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
#print("Cq = ",clusters,"\n")
#random_s = cosum.randomizer(clusters)
#print("Были выбраны предложения с индексами : ",random_s)
#checkItData = stageOne(random_s,document)
#print("Проверка на условия: ",checkItData)
#print("Результат :",stageTwo(checkItData))
#print("F(X) =",F(X,O,Sentences,random_s))
print("Initialization...")
#print(X)
#for i in range(1000):
t_max = 5
random_all_fx = []
random_all_sent = []
def start():
    # Создаем популяции
    for i in range(2):   
        optimize = []          # Лист для хранения fx()
        sent = []               # Лист для хранения индексов предложения

        
        # Создаем особь для популяции. Количество задано в t_max
        for t in range(t_max):
            # I
            random_s = Initialization_exp(X)    # Выбираем случайные предложения
            print("U_p,s(",t,")",random_s)
            
            # II
            fx = F(X,O,Sentences,random_s)      # Считаем f(x)
            print("F(X) =",fx)
            
            # III
            optimize.append(fx)                 # Записываем значения f(x)) в список optimize[]
            sent.append(random_s)               # Записываем индексы предложения в список sent[]
            
            # IV
            best_local = max(optimize)          # Находим максимальное значение f(x), среди t генерации.Это будет best_local  

            # Проверяем на пустату популяции если пусто global = local
            if not random_all_sent:
                print("Population is empty")
                best_global = best_local
            else:
                print("BEST_GLOBAL_ARR",random_all_fx)
                best_global = max(random_all_fx)
                print("MAX from global",best_global)
            print("F[] = ",random_all_fx,"\n")

            # Вытаскиваем индексы лучших предложении ,из предложении sent.
            index = optimize.index(best_local)

            # Пытаемся мутировать лучший вариант
            a = Vp(best_global,best_local,sent[index],t)
        random_all_fx.append(optimize)
        random_all_sent.append(sent)
        print("RANDOM_ALL_FX",random_all_fx)
print("Finish")
start()
#print("Sigmoid =",sigmoid())

