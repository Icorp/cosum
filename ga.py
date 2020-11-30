import random
import math
import sys
import time
from formulation import funcSum
from formulation import funcSum2
from formulation import funcSum3
from optimize import Objective

class Ga(Objective):
    def __init__(self, CR = 0.7, t_max = 1000, B_minus=0.1, B_plus=0.5, l_max = 100, pop_size = 200):
        self.CR = CR
        self.t_max = t_max
        self.B_minus = B_minus
        self.B_plus = B_plus
        self.l_max = l_max
        self.pop_size = pop_size
    

    """
    Equation 25 - The parameter β can be increased during the run to penalize infeasible solutions and drive the
                search to feasible ones that means the adaptive control of the penalty costs
    """
    def computeSimilarity(self, Wi, Wj):
        f1 = funcSum(Wi,Wj)
        f2 = funcSum2(Wi,Wj)
        f3 = funcSum3(Wi)
        f4 = funcSum3(Wj)
        self.similarity.append(1-((2*f1*f2)/((f4*f1)+(f3*f2))))
    
    def init_S_value(self, s):
        self.similarity = []
        self.rand_p_s = random.random()
        
        for i in range(len(self.data)):
            self.computeSimilarity(self.data[s], self.data[i])
        
        U_min = min(self.similarity)
        U_max = max(self.similarity)

        self.start_ups_value = U_min +(U_max-U_min)*self.rand_p_s 
    
    def initialization(self):
        self.genomes = []
        for s in range(len(self.data)):
            self.init_S_value(s)
            self.calculate_sigmoid(self.start_ups_value)
            if self.rand_p_s < self.sigmoid:
                self.genomes.append(1)
            else:
                self.genomes.append(0)
        
    def calculate_B(self, t):
        self.B = self.B_minus+(self.B_plus-self.B_minus)*(t/self.t_max)
    
    # Equation 17 - F (t) is the scaling factor
    def calculate_scaling_factor(self, t):
        self.scaling_factor = 1/(1+math.exp(-t/self.t_max))
    
    def calculate_inertia_weight(self, t):
        self.inertia_weight = 0.9-0.5*(t/self.t_max)

    def calculate_sigmoid(self, z):
        self.sigmoid = 1/(1+math.exp(-z))
    
    def initializeData(self):
        self.us = []
        for i in range(len(self.data)):
            rand_p = random.randint(0, 1)
            self.us = min(self.data[i])+(max(self.data[i])-min(self.data[i]))*rand_p

    def ChooseSentences(self, t):
        self.indexs = []
    
    # is the local best solution of the pth individual during t generation
    def best_local(self, t):
        print("local")
    
    def best_global(self, t):
        print("global")
    
    def fitness(self):
        objectives = Objective()
        objectives.computeObjectiveFunction()

    def fit(self, data):
        self.data = data
        self.population = []
        for t in range(self.t_max):
            start_time = time.time()
            self.initialization()
            self.population.append(self.genomes)
            print("--- %s seconds ---" % (time.time() - start_time))
        print(self.population)