from math import exp
import random
# CR = 0.7
# B- = 0.1
# B+ = 0.5
# Lmax = 100
# Tmax = 1000
# P = 200
B_minus = 0.1
B_plus = 0.5
Population = 200         # Final number of Population
t_max = 1000    # 
Lavg = 100

def Initialization(S):
    result = []
    min_values = []
    max_values = []
    for s in range(len(S)):
        min_values.append(min(S[s]))
        max_values.append(max(S[s]))
        UsMin = min(min_values)
        UsMax = max(max_values)
        rand = round(random.random(),3)
        Ups = UsMin +(UsMax-UsMin)*rand
        result.append(round(Ups,3))
    return result

def finalInit(S,t):
    print("start")

# T - current generation
# T_max - 1000
def scaling_factor(t):
    return round(1/(1+exp(-t/t_max)),3)

def inertia_weight(t):
    return round(0.9-(0.5*(t/t_max)),3)

def crossover():
    return something

def Vp(t):
    result = []
    for t in range(t_max):
        w = inertia_weight(t)
        f = scaling_factor(t)
        U_global = max(P)
        U_local = max(T)
        U_pt = Initialization()
        return round(w*U_local+f*(Ul-u)+(1-f)*(Ug-u),3)
def finalVp():
    result = []
    for p in range(Population):
        for t in range(t_max):
            result.append(Vp(t))


def sigm(z):
    return 1/1+exp(-z)

def fitness(X):
    F = F(X)*exp(-B*max(0,sum))*exp(-B*max(0,L[0]))

def sum():
    for q in range(k):
        for i in range(nq):
            result = l[i]*x[q][i]-Lmax
    return result

def Beta(t):
    return round(B_minus+(B_plus-B_minus)*t/t_max,3)