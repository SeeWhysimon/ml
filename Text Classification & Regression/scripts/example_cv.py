from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

corpus = ["hello, how are you?",
          "It is nice to meet you.",
          "Let's go play basketball!",
          "Yes!!!"]

# Ignore token_pattern that defines regex for tokenization, because word_tokenize is used here
ctv = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

ctv.fit(corpus)

corpus_transformed = ctv.transform(corpus)

print("original string: ")
print(corpus)
print("vocabulary: ")
print(ctv.vocabulary_)
print("transformed string: ")
print(corpus_transformed)