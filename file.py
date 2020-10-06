import pickle

def writeToFile(data):
    with open('listfile.data', 'wb') as filehandle:
        pickle.dump(data, filehandle)

def readFile():
    with open('listfile.data', 'rb') as filehandle:  
        # сохраняем данные как двоичный поток
        placesList = pickle.load(filehandle)
    return placesList