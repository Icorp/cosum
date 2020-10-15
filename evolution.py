from math import exp
import random
from optimize import F
# CR = 0.7
# B- = 0.1
# B+ = 0.5
# Lmax = 100
# Tmax = 1000
# P = 200
B_minus = 0.1
B_plus = 0.5
POPULATION_SIZE = 200         # Final number of Population
MAX_GENERATION = 1000    # 
Lavg = 100
CR = 0.7

def Initialization(S):
    result = []
    for s in range(len(S)):
        UsMin = min(S[s])
        UsMax = max(S[s])
        rand = round(random.random(),3)
        Ups = UsMin +(UsMax-UsMin)*rand
        result.append(round(Ups,3))
    return result
def Initialization_exp(S):
    cash = []
    cash2 = []
    result = []
    for s in range(len(S)):
        cash = []
        for i in range(len(S[s])):
            UsMin = 0
            if S[s][i]==0:
                UsMax = 0
            else:
                UsMax = 1
            rand = random.randint(0, 1)
            Ups = UsMin +(UsMax-UsMin)*rand 
            cash.append(Ups)
        cash2.append(cash)
    for i in range(len(cash2)):
        cash3 = []
        for k in range(len(cash2[i])):
            if cash2[i][k] == 1:
                cash3.append(k)
        result.append(cash3)
    # Check on null list
    for i in range(len(result)):
        if not result[i]:
            print("Не выбрано предложение в одной из кластеров!")
            print("ReStart Initialization ...")
            return Initialization_exp(S)
    return result,cash2
 
def finalInit(S,t):
    print("start")

# T - current generation
# T_max - 1000
def scaling_factor(t):
    return round(1/(1+exp(-t/MAX_GENERATION)),3)

def inertia_weight(t):
    return round(0.9-(0.5*(t/MAX_GENERATION)),3)

def crossover():
    return something

def Vp(best_global,best_local,vector,t):
    #print("Start Vp mutation ...")
    #print("Vector",vector)
    result = []
    w = inertia_weight(t)
    f = scaling_factor(t)
    for q in range(len(vector)):
        cash = []
        for s in range(len(vector[q])):
            vps = w*best_local[q][s]+f*(best_local[q][s]-vector[q][s])+(1-f)*(best_global[q][s]-vector[q][s])
            cash.append(round(vps,3))
        result.append(cash)
    #print("V_p,s(",t,")",result)
    return result
    #return round(w*U_local+f*(Ul-u)+(1-f)*(Ug-u),3)
def finalVp():
    result = []
    for p in range(Population):
        for t in range(t_max):
            result.append(Vp(t))

def Zp(ups,vps):
    rand = round(random.random(),3)
    if rand <= CR:
        return vps
    else:
        return ups
def sigmoid(z):
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

def best_local(fx):
    result = max(fx)
    return result

def start():
    result = []
    
    # Создаем популяции
    for p in range(population):   
        optimize = []          # Лист для хранения fx()
        sent = []              # Лист для хранения индексов предложения        
        random_s = 0.0          # Лист для хранения вектора
        vector = list
        for m in range(t_max):
            random_s,vector = Initialization_exp(X)
            fx = F(X,O,Sentences,random_s)
            optimize.append(fx)
            sent.append(random_s)
            random_all_sent.append(vector)
        random_all_fx.append(optimize)
        # Создаем особь для популяции. Количество задано в t_max
        for t in range(t_max):
            print("Популяция #",p+1)
            print("Особь #",t+1,"\n")
            best_local = max(optimize)          # Находим максимальное значение f(x), среди t генерации.Это будет best_local  
            print("Best_local",best_local)
            all_fx_mixed = []
            # Проверяем на пустату популяции если пусто global = local
            if len(random_all_fx)==0:
                best_global = max(optimize)
            else:
                print("Search GLOBAL BEST ....")
                all_fx_mixed = cosum.mix(random_all_fx)
                print("ALL fx => ",all_fx_mixed)
                best_global = max(all_fx_mixed)
                print("Best global fx(X)",best_global)
                index_g = all_fx_mixed.index(best_global)
                print("Index of the best global",index_g)


            # Вытаскиваем индексы лучших предложении ,из предложении sent.
            index_l = optimize.index(best_local)
            #print("GLOBAL Index",random_all_sent)
            #print("LOCAL Index",random_all_sent)
            
            # Пытаемся мутировать лучший вариант
            result.append(Vp(random_all_sent[index_g],random_all_sent[index_l],random_all_sent[t],t))
            #crossover(resukt)
    return result
