import re
from nltk.tokenize import sent_tokenize

f = open('SJMN91-06184003.txt',"r",encoding="utf-8",)
f =f.read()
a = re.findall(r'(<TEXT>.+?</TEXT>)', f,flags=re.DOTALL)
a = a[0].replace("\n"," ")
a = a.replace(".;",".")
a = a.replace('<TEXT>','')
a = a.replace('</TEXT>','')
s = sent_tokenize(a)
print(a)

