import random
class ga:
    def initializeData(self):
        self.po
        self.us = []
        for i in range(len(self.data)):
            rand_p = random.randint(0, 1)
            self.us = min(self.data[i])+(max(self.data[i])-min(self.data[i]))*rand_p
    def fit(self, data):
        self.data = data