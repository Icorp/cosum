from cosum import findK
from cosum import labelInMatrix
from cosum import computeFullWeight
from cosum import selectCenterOfCluster
from cosum import compute_sim_opt
from cosum import computeSimilirity
from cosum import getIndexCentroid
from cosum import computeOptimalCentroid
from cosum import getCentroidValue
from cosum import readText
from cosum import calculateCenter
from cosum import findLabels
from nltk import sent_tokenize
from cosum.formulation import funcSumCenter

print("Start ...")
print("Reading document ...")
text = readText("../training/AP880310-0257")
sentences = sent_tokenize(text)

K = findK(text)
print("K = ",K)
vector = computeFullWeight(text)
print("Tokens",vector)
print("Computing centroids ...")

# select centroids
centroidsIndexs = selectCenterOfCluster(K,text)
print("centroid",centroidsIndexs)

# get centroid values
centroids = getCentroidValue(centroidsIndexs,vector)

# calculate Optimal center
centroid = computeOptimalCentroid(vector,centroids,42)