from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer

def findTokenAndLower(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def finalTokenSentence(sentence):
    return final_token(sentence)

def Stemm(tokens):
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def final_token(text):
    cash = tokenizeAndRemoveStopWord(text)
    tokens = Stemm(cash)
    result = unionTokens(tokens)
    return result

def tokenizeRemoveAndStemm(text):
    cash = tokenizeAndRemoveStopWord(text)
    return Stemm(cash)
    
def tokenizeAndRemoveStopWord(text):
    stop_words = set(stopwords.words('english')) 
  
    word_tokens = word_tokenize(text) 
    
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    
    filtered_sentence = []
    filtered_tokens = [] 
    for w in word_tokens: 
        if w.lower() not in stop_words: 
            filtered_sentence.append(w)
    for token in filtered_sentence:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

# Find sentence
def findSentences(text):
    return sent_tokenize(text)

# Find total number of sentence, and return result.
def findNumOfSentence(text):
    return len(sent_tokenize(text))

# Fint total number of words in text, and return result.
def findNumOfWord(text):
    return len(findTokenAndLower(text))

# This func search word in your sentences. If he find it, he adding 1 point to counter, and return result.
def findNumSentencesSearchingWord(sentences,word):
    counter = 0
    for i in range(len(sentences)):
        token = finalTokenSentence(sentences[i])
        for k in range(len(token)):
            if word.lower() == token[k].lower():
                counter += 1
                break
    return counter

def findNumWordInSentence(sentence,word):
    counter = 0
    token = tokenizeRemoveAndStemm(sentence)
    for k in range(len(token)):
        if word.lower() == token[k].lower():
            counter += 1
    return counter
        
def unionTokens(tokens):
    result = sorted(set(tokens), key=tokens.index)
    return result