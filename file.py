import pickle
import json
from utils import concatenate_list_data
import re

def writeToFile(data):
    with open('files/listfile.data', 'wb') as filehandle:
        pickle.dump(data, filehandle)

def readFile():
    with open('files/listfile.data', 'rb') as filehandle:  
        # сохраняем данные как двоичный поток
        placesList = pickle.load(filehandle)
    return placesList

def saveJson(nameFile,data):
    with open('results/data.json', 'w') as outfile:
        json.dump(data, outfile)

def saveStats(fx,indexs,scores):
    fx = str(fx)
    indexs = concatenate_list_data(indexs)
    rouge = str(round(scores.get('rouge1').recall,3))
    result = "Fx = "+fx+"\t"+"Indexs = "+indexs+"\t"+"Rouge = "+rouge+"\n"
    f = open('results/stats.txt', 'a') 
    f.write(result)
    f.close()
    print("Writing file")
    print("Status:Ok")

def saveGenomes(genomes):
    indexs = concatenate_list_data(genomes)
    result = "Sentences = "+indexs+"\n"
    f = open('results/genomes.txt', 'a') 
    f.write(result)
    f.close()

def saveZpt(zpt):
    zpt = concatenate_list_data(zpt)
    result = "Sentences = "+zpt+"\n"
    f = open('results/zpt.txt', 'a') 
    f.write(result)
    f.close()


def saveBestGenomes(genomes,t,p,best_global):
    indexs = concatenate_list_data(genomes)
    t = str(t)
    p = str(p)
    best_global = str(best_global)
    result = "P ="+p+"\t t="+t+"\t genomes= "+indexs+"\t best_global="+best_global+"\n"
    f = open('results/best.txt', 'a') 
    f.write(result)
    f.close()

# read text from DUC documents
def readText(name):
    f = open(name,"r",encoding="utf-8",)
    f =f.read()
    text = re.findall(r'(<TEXT>.+?</TEXT>)', f,flags=re.DOTALL)
    text = text[0].replace("\n"," ")
    text = text.replace(".;",".")
    text = text.replace('<TEXT>','')
    text = text.replace('</TEXT>','')
    return text

def saveRandomSummary(index):
    indexs = concatenate_list_data(index)
    result = "Sentences = "+indexs+"\n"
    f = open('results/bestRandom.txt', 'a') 
    f.write(result)
    f.close()