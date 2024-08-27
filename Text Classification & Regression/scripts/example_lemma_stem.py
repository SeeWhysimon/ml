from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

words = ["fishing", "fishes", "fished"]

for word in words:
    print(f"word={word}, stemmed word={stemmer.stem(word)}, lemma={lemmatizer.lemmatize(word)}")