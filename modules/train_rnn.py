from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
from modules.preprocess import *
from modules.utils import build_dataset, text_to_word2vec, evaluate
from modules.rnn_model import TextRNN
import spacy
import nltk
import gensim.downloader as api


parser = argparse.ArgumentParser()
parser.add_argument('num_samples', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('hidden_size', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('tag', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    
    #nltk.download('punkt')
    #nltk.download('stopwords')
    nlp = spacy.load("fr_core_news_sm")

    dataset = build_dataset(path="ml1_termproject/lapresse_crawler", num_samples=args.num_samples, rnd_state=10)

    dataset = text_edit(dataset, grp_num=False, rm_newline=True, rm_punctuation=True,
              rm_stop_words=False, lowercase=True, lemmatize=False, html_=False, convert_entities=False, expand=False)
    
    X = [x['text'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'affaires', 'arts', 'international']]
    Y = [x['section_label'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'affaires', 'arts', 'international']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

    model_name = 'fasttext-wiki-news-subwords-300'  
    word2vec_model = api.load(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    text = "Ceci est un texte exemple"
    vector = text_to_word2vec(text, word2vec_model)

    input_size = vector.shape[0]  
    hidden_size = args.hidden_size
    output_size = len(set(Y_train))  

    rnn = TextRNN().to(device)
    optimizer = optim.Adam(rnn.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()

    X_train = torch.stack([torch.tensor(text_to_word2vec(x, word2vec_model), dtype=torch.float32).view(1,-1) for x in X_train], dim=0).to_device()
    X_test = torch.stack([torch.tensor(text_to_word2vec(x, word2vec_model), dtype=torch.float32).view(1,-1) for x in X_test], dim=0).to_device()

    Y_train = torch.tensor(Y_train, dtype=torch.long).to_device()
    Y_test = torch.tensor(Y_test, dtype=torch.long).to_device()

    batch_size = args.batch_size

    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_dataset = TensorDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    epochs = args.epochs  
    best_test_loss = float('inf')

for epoch in range(epochs):
    train_losses = []
    test_losses = []
    for X, Y in dataloader:  
        rnn.train()
        optimizer.zero_grad()
        outputs = rnn(X)
        loss = criterion(outputs, Y)
        loss.backward() 
        optimizer.step()
        train_losses.append(loss.detach())
    for X, Y in test_dataloader:  
        rnn.eval()
        outputs = rnn(X)
        loss = criterion(outputs, Y)
        test_losses.append(loss.detach())

    mean_test_loss = np.mean(test_losses)
    print(f'Results for epoch {epoch}:')
    print(f'Mean train loss for epoch: {np.mean(train_losses)}')
    print(f'Mean test loss for epoch: {mean_test_loss}')

    if mean_test_loss < best_test_loss:
        best_test_loss = mean_test_loss
        torch.save(rnn.state_dict(), f'{args.tag}.pt') 
        print(f'Model saved at epoch {epoch} with test loss {mean_test_loss}')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn = TextRNN(input_size, hidden_size, output_size).to(device)  
    state_dict = torch.load('rnn_best.pt', map_location=device)  
    rnn.load_state_dict(state_dict)

    rnn.eval()
    pred_outputs = []
    for tensor_ in X_test:
        output = rnn(tensor_.view(1,1,-1))
        pred_class = np.argmax(output.detach())
        pred_outputs.append(int(pred_class))

    print(evaluate(Y_test.numpy(), np.array(pred_outputs)))