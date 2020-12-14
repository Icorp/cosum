from cosum import CosumTfidfVectorizer

text = "This is sentence one. This is sentence three. This is sentence four. This is sentence five."
vectorizer = CosumTfidfVectorizer()
vectorizer.fit(text)
vector = vectorizer.weight_matrix
print(vector)
