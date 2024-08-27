import io
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

def load_word_embeddings(file_path: str) -> dict:
    """
    Load word embeddings from a file.
    
    :param file_path: Path to the embeddings file.
    :return: A dictionary with words as keys and their corresponding embeddings as values.
    """
    with io.open(file_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f_in:
        num_words, embedding_dim = map(int, f_in.readline().split())
        embeddings = {}
        for line in f_in:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vector = list(map(float, tokens[1:]))
            embeddings[word] = vector
    return embeddings

def text_to_vector(text: str, embedding_dict: dict, stop_words: list, tokenizer) -> np.ndarray:
    """
    Convert a sentence to a vector by averaging its word embeddings.
    
    :param text: Input sentence.
    :param embedding_dict: Dictionary containing word embeddings.
    :param stop_words: List of stopwords to exclude from the sentence.
    :param tokenizer: Function to tokenize the sentence.
    :return: A numpy array representing the averaged word embeddings of the sentence.
    """
    tokens = tokenizer(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    word_vectors = [embedding_dict[token] for token in tokens if token in embedding_dict]
    
    if not word_vectors:
        return np.zeros(300)
    
    word_vectors = np.array(word_vectors)
    sentence_vector = word_vectors.sum(axis=0)
    return sentence_vector / np.linalg.norm(sentence_vector)

def main():
    # Load dataset and shuffle it
    df = pd.read_csv("../../data/Text Classification & Regression/IMDB Dataset.csv")
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Load word embeddings
    print("Loading word embeddings...")
    word_embeddings = load_word_embeddings("../../data/Text Classification & Regression/crawl-300d-2M.vec")

    # Convert sentences to vectors
    print("Converting sentences to vectors...")
    sentence_vectors = np.array([
        text_to_vector(text=review, 
                       embedding_dict=word_embeddings, 
                       stop_words=[], 
                       tokenizer=word_tokenize)
        for review in df["review"].values
    ])
    
    labels = df["sentiment"].values
    kfold = model_selection.StratifiedKFold(n_splits=5)
    
    # Train and evaluate the model using cross-validation
    for fold_index, (train_indices, val_indices) in enumerate(kfold.split(X=sentence_vectors, y=labels)):
        print(f"Training fold {fold_index}...")
        x_train, x_val = sentence_vectors[train_indices], sentence_vectors[val_indices]
        y_train, y_val = labels[train_indices], labels[val_indices]

        model = linear_model.LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)

        predictions = model.predict(x_val)
        accuracy = metrics.accuracy_score(y_val, predictions)
        print(f"Fold {fold_index} Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    main()
