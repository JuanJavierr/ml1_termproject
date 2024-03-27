from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
from modules.autoencoder import AUTOENCODER
from modules.preprocess import *
from modules.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('num_samples', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('autoencoder_tag', type=str)
args = parser.parse_args()


if __name__ == "__main__":

    dataset = build_dataset(path="lapresse_crawler/output.json", num_samples=args.num_samples, rnd_state=10)

    dataset = text_edit(dataset, grp_num=True, rm_newline=True, rm_punctuation=True,
              rm_stop_words=True, lowercase=True, lemmatize=True, html_=True, convert_entities=True, expand=True)
    
    X = [x['text'] for x in dataset.values() if x['section_1'] in ['actualites', 'international', 'sports', 'arts', 'affaires', 'debats']]
    Y = [x['section_label'] for x in dataset.values() if x['section_1'] in ['actualites', 'international', 'sports', 'arts', 'affaires', 'debats']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

    vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, min_df=0.01, max_df=0.99)
    tfidf_train = vectorizer.fit_transform(X_train)
    tfidf_test =  vectorizer.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    auto = AUTOENCODER().to(device)
    optimizer = optim.Adam(auto.parameters(), lr = args.lr)
    loss_function = nn.MSELoss()

    tfidf_train_dense_tensor = torch.unsqueeze(torch.tensor(tfidf_train.toarray(), dtype=torch.float32), dim=1).to(device)
    tfidf_test_dense_tensor = torch.unsqueeze(torch.tensor(tfidf_test.toarray(), dtype=torch.float32), dim=1).to(device)

    batch_size = args.batch_size
    dataset = TensorDataset(tfidf_train_dense_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_dataset = TensorDataset(tfidf_test_dense_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    epochs = args.epochs  
    best_test_loss = float('inf')

    for epoch in range(epochs):
        train_losses = []
        test_losses = []
        for batch in dataloader:
            X, = batch
            X = X.to(device)
            auto.train()
            auto_out = auto(X)
            auto.zero_grad()
            loss = loss_function(auto_out, X)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_losses.append(loss_value)
        
        for batch in test_dataloader:
            X, = batch  
            X = X.to(device)
            auto.eval()
            auto_out = auto(X)
            loss = loss_function(auto_out, X)
            loss_value = loss.item()
            test_losses.append(loss_value)

        mean_test_loss = np.mean(test_losses)
        print(f'Results for epoch {epoch}:')
        print(f'Mean train loss for epoch: {np.mean(train_losses)}')
        print(f'Mean test loss for epoch: {mean_test_loss}')

        if mean_test_loss < best_test_loss:
            best_test_loss = mean_test_loss
            torch.save(auto.state_dict(), f'auto_{args.autoencoder_tag}.pt') 
            print(f'Model saved at epoch {epoch} with test loss {mean_test_loss}')