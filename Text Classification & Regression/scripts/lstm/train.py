import io
import torch

import numpy as np
import pandas as pd

from sklearn import metrics
from keras.preprocessing import sequence
from keras.preprocessing import text
import torch.utils
import torch.utils.data

import config
import dataset
from create_folds import create_folds
import engine
import lstm

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

def create_embedding_matrix(word_index, embbeding_dict):
    embbeding_matrix = np.zeros((len(word_index) + 1, 300))
    
    for word, idx in word_index.item():
        if word in embbeding_dict:
            embbeding_matrix[idx] = embbeding_dict[word]
    
    return embbeding_matrix

def run(df: pd.DataFrame, fold):
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    print("Fitting tokenizer...")
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(df["review"].values.tolist())

    x_train = tokenizer.texts_to_sequences(train_df["review"].values)
    x_valid = tokenizer.texts_to_sequences(valid_df["review"].values)

    x_train = sequence.pad_sequences(x_train, maxlen=config.MAX_LEN)
    x_valid = sequence.pad_sequences(x_valid, maxlen=config.MAX_LEN)

    train_dataset = dataset.IMDBDataset(reviews=x_train, targets=train_df["sentiment"].values)
    valid_dataset = dataset.IMDBDataset(reviews=x_valid, targets=valid_df["sentiment"].values)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=config.TRAIN_BATCH_SIZE,
                                                    num_workers=2)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=config.VALID_BATCH_SIZE,
                                                    num_workers=1)
    
    print("Loading embeddings...")
    embedding_dict = load_word_embeddings("../../../data/Text Classification & Regression/crawl-300d-2M.vec")
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = lstm.LSTM(embedding_matrix)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config.ADAM_LEARNING_RATE)

    print("Training model...")
    best_accuracy = 0
    early_stop_counter = 0
    for epoch in range(config.EPOCHS):
        engine.train(data_loader=train_data_loader, 
                     model=model, 
                     optimizer=optimizer, 
                     device=device)
        outputs, targets = engine.evaluate(data_loader=valid_data_loader,
                                           model=model, 
                                           device=device)
        outputs = np.array(outputs) >= config.THRESHOLD

        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Fold {fold}, Epoch {epoch}, Accuracy Score {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stop_counter += 1
        if early_stop_counter > 2:
            break

if __name__ == "__main__":
    df = create_folds("../../../data/Text Classification & Regression/IMDB Dataset.csv")
    for fold in range(config.FOLD_NUM):
        run(df=df, fold=fold)