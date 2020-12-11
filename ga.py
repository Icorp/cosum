import random
import math
import sys
import time
from utils import computeSimilarity
from utils import funcSum
from utils import funcSum2
from utils import funcSum3
from optimize import Objective
from file import saveGenomes
from file import saveBestGenomes
from file import saveZpt
from file import saveJson

class Ga(Objective):
    def __init__(self, CR = 0.7, t_max = 1000, B_minus=0.1, B_plus=0.5, l_max = 100, pop_size = 200):
        self.CR = CR
        self.t_max = t_max
        self.B_minus = B_minus
        self.B_plus = B_plus
        self.l_max = l_max
        self.pop_size = pop_size
    
    def init_S_value(self, s):
        """
    
        Equation 15 - calculate
        rand_p,s_values is a random number between 0 and 1, chosen once for each s ∈ {1, 2, …, n}.
        
        """
        self.similarity = []
        self.rand_p_s_value = random.random()
        self.rand_p_s.append(self.rand_p_s_value)
        for i in range(self.len_data):
            self.similarity.append(computeSimilarity(self.data[s], self.data[i]))
        
        U_min = min(self.similarity)
        U_max = max(self.similarity)

        self.start_ups_value = U_min +(U_max-U_min)*self.rand_p_s_value 
    
    def initialization(self):
        self.genomes = []
        self.upt = []
        self.rand_p_s = []
        for s in range(self.len_data):
            self.init_S_value(s)
            self.upt.append(self.start_ups_value)
    
    def binarization(self, genomes_value):
        self.genomes = []
        for s in range(self.len_data):
            rand_p_s = random.random()
            self.calculate_sigmoid(genomes_value[s])
            if rand_p_s < self.sigmoid:
                self.genomes.append(1)
            else:
                self.genomes.append(0)

    def calculate_B(self, t):

        """
        Equation(25)

        About:
            The parameter β can be increased during the run to penalize infeasible solutions and drive the
            search to feasible ones that means the adaptive control of the penalty costs

        Default value:
            B_minus=0.1, B_plus=0.5
        """

        self.B = self.B_minus+(self.B_plus-self.B_minus)*(t/self.t_max)
    
    def calculate_scaling_factor(self, t):

        """   
        Equation(17)

        About:
            F(t) is the scaling factor
        """
        
        self.scaling_factor = 1/(1+math.exp(-t/self.t_max))
    
    def calculate_inertia_weight(self, t):

        """   
        Equation(18)

        About:
            The inertia weight ω is linearly decreased from 0.9 to 0.4 (Das & Suganthan, 2011)
        """

        self.inertia_weight = 0.9-0.5*(t/self.t_max)

    def calculate_sigmoid(self, z):

        """   
        Equation(18)

        About:
            The motivation to use the sigmoid function is to map interval umins ; umaxs
            for each s ∈ {1, 2, …, n} into the interval (0, 1), which is equivalent
            to the interval of a probability function
        """

        self.sigmoid = 1/(1+math.exp(-z))

    def mutation(self,t):

        """   
        Equation(16)

        About:
            DE is based on a mutation operator, which adds an amount obtained by the difference of two randomly
            chosen individuals of the current population, in contrast to most of the evolutionary algorithms, 
            in which the mutation operator is defined by a probability function. Mutation expands
            the search space. In each generation to change each population member, a mutant vector is created.

            self.best_global is the global best solution of population
            self.best_local is the local best solution of the pth individual during (t) generation, respectively
        """

        self.vpt = []
        self.calculate_inertia_weight(t)
        self.calculate_scaling_factor(t)
        for s in range(self.len_data):
            self.vpt.append((self.inertia_weight*self.best_local)+(self.scaling_factor*(self.best_local-self.fitness))+(1-self.scaling_factor)*(self.best_global-self.fitness))

    def crossover(self):
        """   
        Equation(19)

        About:
            In order to increase the diversity of the perturbed parameter vectors, a crossover operator is introduced. 
            The parent vector Up(t) is mixed with the mutated vector Vp(t) to produce a trial vector Zp(t)=[zp, 1(t), …, zp, n(t)]. 
            It is developed from the elements of the target vector,Up(t), and the elements of the mutant vector, Vp(t)
        """
        self.zpt = []
        for i in range(self.len_data):
            rand = random.randint(0,self.len_data)
            if self.rand_p_s[i] <= self.CR:
                self.zpt.append(self.vpt[i])
            else:
                self.zpt.append(self.upt[i])
    
    def calculate_words(self):
        """   
        Equation(19)

        About:
            Calculate number of word in sentence, and calcalute all summ
        """
        self.li = []
        for i in range(len(self.data)):
            counter = 0
            for m in range(len(self.data[i])):
                if self.data[i][m]>0:
                    counter+=1
            self.li.append(counter)

    def check_loop(self, types):
        """   
        Equation(25) Second multiplier

        About:
            . The second multiplier is defined as an additional penalty function for maximization. 
            β represents the cost of overloaded summary. Initial value of β is set by the user. If a solution is not feasible, the second term will be less than 1,
            and therefore, the search will be directed to a feasible solution. If the summary length is not exceeded, this term will be equal to 1 to ensure that the solution is not to be penalized.
        """
        self.check_l_value = 0.0
        if types == "lmax":
            for q in range(self.K):
                for i in self.clusterSentence[q]:
                    self.check_l_value += self.li[i]*self.genomes[i]
            self.check_l_value = self.check_l_value - self.l_max 
        else:
            for q in range(self.K):
                for i in self.clusterSentence[q]:
                    self.check_l_value += self.li[i]*self.genomes[i] - self.l_avg

    def calculate_fitness(self, t):
        """   
        Equation(24) Second multiplier

        About:
                The first multiplier f (X) in Equation (24) is the objective function (9)
        """
        self.objectives = Objective()
        self.objectives.Fx(self.data, self.genomes, self.cq, self.centroids, self.clusterSentence, self.K)
        self.F = self.objectives.F
        self.selected = self.objectives.selected_sentences
        self.calculate_B(t)
        self.check_loop("lmax")
        self.check_lmax_value = self.check_l_value
        self.check_loop("li")
        self.check_li_value = self.check_l_value
        self.a = math.exp(-self.B*max(0,self.check_lmax_value))
        self.b = math.exp(-self.B*max(0,self.check_li_value))
        self.fitness = self.F*self.a*self.b
    
    def fit(self, data, cq, centroids, l_avg, clusterSentence, K):
        self.K = K
        self.clusterSentence = clusterSentence
        self.l_avg = l_avg
        self.centroids = centroids
        self.data = data
        self.cq = cq
        self.populations = []
        self.all_genomes = []
        self.best_global = 0.0
        self.best_summary = []
        self.len_data = len(self.data)
        self.calculate_words()
        self.best_genome = []

        # init genome and calculate fitness
        my_string = "Population = {}\t F(X) = {}\t fitness = {}\t time = {}"
        data = {}
        for p in range(self.pop_size):
            population = []
            # start_time
            start_time = time.time()
            data['pop'+str(p)] = []
            self.pop_genomes = []
            self.best_local = 0.0
            
            # initialization start random genomes_values
            self.initialization()
            
            # # convert to genome [1,0,0,1,1]
            self.binarization(self.upt)
            
            # calculate fitness
            self.calculate_fitness(0)
            
            for t in range(self.t_max):
                

                # save prev fitness and summary values
                prev_fitness = self.fitness
                prev_genomes = self.genomes
                
                # appending
                self.pop_genomes.append(self.genomes)
                population.append(prev_fitness)
                self.best_local = max(population)
                
                # check best global
                if self.best_local > self.best_global:
                    self.best_global = self.best_local
                    self.best_summary = self.selected
                    saveBestGenomes(self.best_summary,t,p,self.best_global)
                
                # mutation
                self.mutation(t)
                self.crossover()

                # convert to genome [1,0,0,1,1]
                self.binarization(self.vpt)
                
                # calculate fitness of new genomes
                self.calculate_fitness(t)
                if self.fitness >= prev_fitness:
                    self.best_genome = []
                    self.best_genome = self.genomes
                else:
                    self.genomes = prev_genomes

                # printing and saving to json
                data['pop'+str(p)].append({
                    't':t,
                    'genomes':self.genomes,
                    'fitness':self.fitness
                })
                
                # writing to file /results/genomes.txt
                saveGenomes(self.selected)
        
            # printing
            print(my_string.format(p, self.F, self.fitness, time.time() - start_time))

        # save all loop in genetic algorithm in json file. Out: /result/data.json
        saveJson("data",data)