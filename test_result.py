# -*- coding: utf-8 -*-
import logging as log
import time
import findIt
import re
from ga import Ga
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from file import saveStats
from file import readText
from rouge_score import rouge_scorer

# Constants
reference = "Senators McClure (R) and Metzenbaum (D) have sponsored bills to prevent plastic guns from slipping through airport security.  The gun, not yet manufactured, is intended for military and police use. Metzenbaum's bill would require some detectable metal content; McClure's would require more sensitive detection equipment at airports, almost certainly causing passenger delays. The NRA opposes the first federal gun ban bill in America, and warns members their guns will soon be inspected and weighed by government agents. However, on this issue they will compromise, not to ban the gun, but to increase airport security.  Handgun control advocates and law enforcement officials back Metzenbaum's proposal."



print("Start ...")
print("Reading document ...")
text = readText("training/AP880310-0257")
sentences = sent_tokenize(text)


# Convert genomes to normal summary text
indexs = [9,12, 13, 18, 24, 25, 26, 34, 35, 36 ,40, 42]

hypothesis = ""
for i in indexs:
    hypothesis+=(sentences[i])

print("\nЛучший набор предложении")
print(indexs)

# calculate Rouge
scores = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scores.score(hypothesis,reference)

# save results to file
print(scores)
print("Finish!")