import nltk
import cosum
from nltk import sent_tokenize
import pickle
import findIt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
from sklearn.cluster import KMeans
from formulation import funcSum4
from evolution import Initialization
DOCUMENT = "The school system of Canada is very much like the one in the USA, but there are certain differences. Education in Canada is general and compulsory for children from 6 to 16 years old, and in some provinces — to 14. It is within the competence of the local authorities, and therefore it may differ from province to province. For example, Newfoundland has an 11-grade system.Some other provinces have 12-grade systems, and Ontario has even a 13-grade system. Grades 1—6 are usually elementary schools, and grades 7—12 are secondary schools. In some provinces there is a kindergarten year before the first grade. Elementary education is general and basic, but in the junior high school years the students can select some courses themselves. Most secondary schools provide programmes for all types of students. Some of them prepare students for continuing their studies at the university. Vocational schools are separate institutions for those who will not continue their education after secondary schools. There also exist some commercial high schools. Some provinces have private kindergartens and nursery schools for children of pre-elementary age. There also exist Roman Catholic schools and private schools in some provinces. In most provinces private schools receive some form of public support. Admission to the university in Canada is after high school with specific courses. Getting a degree in law, medicine, dentistry or engineering usually takes 3—4 years of studying. University tuition fees vary among different provinces. All provinces also have public non-university institutions. They are regional colleges, institutes of technology, institutes of applied arts, colleges of agricultural technology and others. Criteria for admission to these institutions are less strict.The educational system in Kazakhstan is conducted in two languages - Kazakh and Russian and consists of several levels of state and private educational establishments: infant schools, elementary (or primary) schools, comprehensive schools, colleges and academies. The constitution of the Republic of Kazakhstan fixes the right of citizens of the republic on free-of-charge secondary education which is obligatory. The constitution prohibits any discrimination on the basis of language or ethnicity and guarantees equal rights in education regardless of nationality. Children start school at the age of 7 and finish at 17. As a rule a child attends the school, located in the neighborhood. However, in big cities there are so-called special schools, offering more in depth studies of the major European languages (English, French, German) or the advanced courses in physics and mathematics and children, attending one of this may have to commute from home."


def writeToFile(data):
    with open('listfile.data', 'wb') as filehandle:
        pickle.dump(data, filehandle)

def readFile():
    with open('listfile.data', 'rb') as filehandle:  
        # сохраняем данные как двоичный поток
        placesList = pickle.load(filehandle)
    return placesList

sentences = sent_tokenize(DOCUMENT)
s = cosum.computeMatrixSimRRN(DOCUMENT)
print(s)
data = np.array(readFile())
kmeans = KMeans(n_clusters=3, random_state=42).fit(data)
#data = cosum.computeClustering(S,kmeans.cluster_centers_.tolist(),3,25,DOCUMENT)
#X = cosum.labelInMatrix(kmeans.labels_)
O = kmeans.cluster_centers_
#print(n)
#cosum.computeAllSimRRN(DOCUMENT)
