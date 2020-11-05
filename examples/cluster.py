from cosum.cosum import findK
from cosum.cosum import labelInMatrix
from cosum.cosum import computeFullWeight
from cosum.cosum import selectCenterOfCluster
from cosum.cosum import compute_sim_opt
from cosum.cosum import computeSimilirity
from cosum.cosum import getIndexCentroid
from cosum.cosum import computeOptimalCentroid
from cosum.cosum import getCentroidValue
from cosum.file import readText
from cosum.cosum import calculateCenter
from cosum.cosum import findLabels
from nltk import sent_tokenize
from cosum.formulation import funcSumCenter
print("Start ...")
print("Reading document ...")
text = readText("../training/AP880310-0257")
sentences = sent_tokenize(text)

K = findK(text)
print("K = ",K)
tokens = computeFullWeight(text)
print("Tokens",tokens)
print("Computing centroids ...")

# select centroids
centroidsIndexs = selectCenterOfCluster(K,text)
print("centroid",centroidsIndexs)

# get centroid values
centroids = getCentroidValue(centroidsIndexs,tokens)

# calculate Optimal center
centroid = computeOptimalCentroid(tokens,centroids,42)