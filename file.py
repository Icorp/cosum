import pickle
import json
import numpy as np

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
