import csv
from cosum import CosumTfidfVectorizer
import csv
import sys


def funcSum(Wi,Wj):
    seqSum=0.0
    m = len(Wi)
    for k in range(m):
        cash=Wi[k]-(Wi[k]*Wj[k])
        seqSum+=cash
    return seqSum    

def funcSum2(Wi,Wj):
    seqSum=0.0
    m = len(Wi)
    for k in range(m):
        cash = Wj[k]-Wi[k]*Wj[k]
        seqSum+=cash
    return seqSum

def funcSum3(W):
    seqSum=0.0
    m = len(W)
    for k in range(m):
        seqSum+=W[k]
    return seqSum


def computeSimClustering(Wi,Wj):
    f1 = funcSum(Wi,Wj)
    f2 = funcSum2(Wi,Wj)
    f3 = funcSum3(Wi)
    f4 = funcSum3(Wj)
    return 1-((2*f1*f2)/((f4*f1)+(f3*f2)))
 
text = "My best friends name's Misha. We made friends a few years ago. We are of the same age. We live in the same block of flats, so we see each other almost every day. Misha is a tall slender boy. He has got dark hair, large dark eyes, a straight nose, and thin lips. He wears spectacles. Misha is a nice guy. He is very honest and just, understanding and kind. I trust him a lot, and I'm sure that I can rely on him in any situation. He never lets people down. Misha is only 16 but he is very responsible — he finishes whatever he starts. He's got only one shortcoming. He is a bit stubborn; nevertheless he is pleasant to deal with. Misha's an only child and his parents love him very much. His father is a lawyer. He is the most brilliant man I've ever met. He knows everything there is to know about the law. His mother is a music teacher. No wonder Michael is so talented. He's got a very good ear for music. He likes jazz and plays the piano very well. We spend a lot of time together. We often watch video or listen to music. Sometimes we go to theatre, or walk around the centre of the city, visiting small cafes, museums, art galleries, and shops. We talk for hours about all sorts of things (politics, love, teachers, and girls). We discuss films, television programmes, books. I never quarrel with Misha. But if there is some misunderstanding between us we try to make peace as soon as possible.We are of the same age. We live in the same block of flats, so we see each other almost every day. My best friends name's Petrov. What I like best about him is that he is always willing to help and share his knowledge, thoughts, and feelings. I respect him for his fairness, strong will, intellect, and modesty. I miss Misha when we don't see each other for a long time. Without him I would feel lonely and uncomfortable. Our friendship helps me feel strong and sure of myself."

vectorizer = CosumTfidfVectorizer()
vectorizer.fit(text)
vector = vectorizer.weight_matrix
sentences = vectorizer.sentences
words = vectorizer.vocabulary
newwords = vectorizer.new_vocabulary
checksentence = 2
cash = []
similarity = []
var1 = computeSimClustering(vector[2],vector[2])
var2 = computeSimClustering(vector[30],vector[2])
var3 = computeSimClustering(vector[21],vector[2])


print(var1)
print(var2)
print(var3)
sys.exit()
for i in range(len(vector)):
    some = computeSimClustering(vector[checksentence],vector[i])
    similarity.append(some)
    cash.append(some)
cash.sort(reverse=True)

result = []
for i in range(10):
    result.append(cash[i])
index = []
for i in range(len(result)):
    indexOfSentences = similarity.index(result[i]) 
    index.append(indexOfSentences)
print(index)
print("\nВзято предложение:\t",sentences[checksentence],"\n")
print("Похожие предложения взятые из текста:\n")
for i in range(len(index)):
    print(sentences[index[i]],"\t",similarity[index[i]])
