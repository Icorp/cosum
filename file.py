import pickle
import json
import numpy as np
from cosum import listToString
import re

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
    f = open('results/stats.txt', 'a') 
    f.write(result)
    f.close()

def readText(name):
    f = open(name,"r",encoding="utf-8",)
    f =f.read()
    text = re.findall(r'(<TEXT>.+?</TEXT>)', f,flags=re.DOTALL)
    text = text[0].replace("\n"," ")
    text = text.replace(".;",".")
    text = text.replace('<TEXT>','')
    text = text.replace('</TEXT>','')
    return text