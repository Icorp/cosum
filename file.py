import pickle
import json
from cosum import listToString
import re

def writeToFile(data):
    with open('files/listfile.data', 'wb') as filehandle:
        pickle.dump(data, filehandle)

def readFile():
    with open('files/listfile.data', 'rb') as filehandle:  
        # сохраняем данные как двоичный поток
        placesList = pickle.load(filehandle)
    return placesList

def saveStats(hypothesis,fx,indexs,summary,scores):
    fx = str(fx)
    indexs = listToString(indexs)
    rouge = str(round(scores[0].get('rouge-1').get('r'),3))
    result = "Fx = "+fx+"\t"+"Indexs = "+indexs+"\t"+"Rouge = "+rouge+"\n"
    f = open('results/stats.txt', 'a') 
    f.write(result)
    f.close()
    print("Writing file")
    print("Status:Ok")

def readText(name):
    f = open(name,"r",encoding="utf-8",)
    f =f.read()
    text = re.findall(r'(<TEXT>.+?</TEXT>)', f,flags=re.DOTALL)
    text = text[0].replace("\n"," ")
    text = text.replace(".;",".")
    text = text.replace('<TEXT>','')
    text = text.replace('</TEXT>','')
    return text