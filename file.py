import pickle
import json
import numpy as np
from cosum import listToString

def writeToFile(data,fileName):
    with open('listfile.data', 'wb') as filehandle:
        pickle.dump(data, filehandle)

def readFile():
    with open('listfile.data', 'rb') as filehandle:  
        # сохраняем данные как двоичный поток
        placesList = pickle.load(filehandle)
    return placesList
def writeToJson(data,fileName):
    a_file = open(fileName, "w")
    for row in data:
        np.savetxt(a_file, row)
    a_file.close()
    print("Writing file")
    print("Status:Ok")

def saveStats(hypothesis,fx,indexs,summary,scores):
    fx = str(fx)
    indexs = listToString(indexs)
    rouge = str(scores[0].get('rouge-1').get('r'))
    result = "Fx = "+fx+"\t"+"Indexs = "+indexs+"\t"+"Rouge = "+rouge+"\n"
    f = open('stats.txt', 'a') 
    f.write(result)
    f.close()
