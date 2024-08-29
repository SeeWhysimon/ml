import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    df = pd.read_csv("../data/IMDB Dataset.csv")
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

        # CounVectorizer initialization
        ctv = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
        ctv.fit(train_df["review"])

        x_train = ctv.transform(train_df["review"])
        x_val = ctv.transform(val_df["review"])

        # Naive Bayes model initialization
        model = naive_bayes.MultinomialNB()
        model.fit(x_train, train_df["sentiment"])

        preds = model.predict(x_val)
        accuracy = metrics.accuracy_score(val_df["sentiment"], preds)
        print(f"Fold: {fold_}, Accuracy: {accuracy}")