import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    # Lemmatizer initialization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

if __name__ == "__main__":
    df = pd.read_csv("../../data/Text Classification & Regression/IMDB Dataset.csv")
    df["kfold"] = -1
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    df = df.sample(frac=1).reset_index(drop=True)

    y = df["sentiment"].values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold_, (train_, val_) in enumerate(kf.split(df, y)):
        df.loc[val_, "kfold"] = fold_

    for fold_ in range(5):
        train_df = df[df["kfold"] != fold_].reset_index(drop=True)
        val_df = df[df["kfold"] == fold_].reset_index(drop=True)

        # TfidVectorizer initialization with N-grams
        ctv = TfidfVectorizer(tokenizer=tokenize_and_lemmatize, token_pattern=None, ngram_range=(2, 2))
        ctv.fit(train_df["review"])

        x_train = ctv.transform(train_df["review"])
        x_val = ctv.transform(val_df["review"])

        # Logistic Regression model initialization
        model = linear_model.LogisticRegression(max_iter=1000)
        model.fit(x_train, train_df["sentiment"])

        preds = model.predict(x_val)
        accuracy = metrics.accuracy_score(val_df["sentiment"], preds)
        print(f"Fold: {fold_}, Accuracy: {accuracy}")