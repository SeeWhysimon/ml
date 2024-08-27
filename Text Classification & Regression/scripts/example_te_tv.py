import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Stopwords list initialization
stop_words = set(stopwords.words('english'))

def clean_text(s: str) -> str:
    tokens = word_tokenize(s)
    # Remove punctuation and stopwords, and convert to lowercase
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    s = " ".join(tokens)
    return s

# Read only 10k samples from the training data
corpus = pd.read_csv("../../data/Text Classification & Regression/IMDB dataset.csv", nrows=10000)
corpus.loc[:, "review"] = corpus["review"].apply(clean_text)
corpus = corpus["review"].values

# Create a TF-IDF Vectorizer with custom tokenizer
tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(1, 1))
corpus_transformed = tfv.fit_transform(corpus)

# Apply Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=10)
corpus_svd = svd.fit_transform(corpus_transformed)

# Extract the feature scores for the first component
sample_index = 0
feature_scores = dict(zip(tfv.get_feature_names_out(), corpus_svd[:, sample_index]))

# Get the top N features with the highest scores
N = 5
top_features = sorted(feature_scores, key=feature_scores.get, reverse=True)[:N]
print(top_features)