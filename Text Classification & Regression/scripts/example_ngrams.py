from nltk import ngrams
from nltk.tokenize import word_tokenize

N = 3

sentence = "hello, how are you? It is nice to meet you. Let's go play basketball! Yes!!!"

tokenized_sentence = word_tokenize(sentence)

n_grams = list(ngrams(tokenized_sentence, N))
print(n_grams)