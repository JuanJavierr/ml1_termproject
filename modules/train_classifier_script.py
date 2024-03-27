from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
from modules.autoencoder import AUTOENCODER
from modules.classifier import CLASSIFIER
from modules.preprocess import *
from modules.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('num_samples', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('autoencoder_tag', type=str)
parser.add_argument('classifier_tag', type=str)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto = AUTOENCODER().to(device)  
    state_dict = torch.load(f'auto_{args.autoencoder_tag}.pt', map_location=device)  
    auto.load_state_dict(state_dict)

    tfidf_train_dense_tensor = torch.unsqueeze(torch.tensor(tfidf_train.toarray(), dtype=torch.float32), dim=1).to(device)  
    tfidf_test_dense_tensor = torch.unsqueeze(torch.tensor(tfidf_test.toarray(), dtype=torch.float32), dim=1).to(device)  

    auto.eval()
    autoencoder_train_tensor = []
    for tensor_ in tfidf_train_dense_tensor:
        encode_output = auto.encode(torch.unsqueeze(tensor_, dim=1))
        autoencoder_train_tensor.append(encode_output)

    autoencoder_train_tensor = torch.stack(autoencoder_train_tensor, dim=1)
    shape_ = autoencoder_train_tensor.shape[1:]
    autoencoder_train_tensor = autoencoder_train_tensor.view(shape_[0],shape_[1],shape_[2])

    autoencoder_test_tensor = []
    for tensor_ in tfidf_test_dense_tensor:
        encode_output = auto.encode(torch.unsqueeze(tensor_, dim=1))
        autoencoder_test_tensor.append(encode_output)

    autoencoder_test_tensor = torch.stack(autoencoder_test_tensor, dim=1)
    shape_ = autoencoder_test_tensor.shape[1:]
    autoencoder_test_tensor = autoencoder_test_tensor.view(shape_[0],shape_[1],shape_[2])


    classifier = CLASSIFIER(k=5, num_class=len(set(Y))).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr = args.lr)
    loss_function = nn.CrossEntropyLoss()

    batch_size= args.batch_size

    dataset = TensorDataset(autoencoder_train_tensor, torch.tensor(Y_train,dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(autoencoder_test_tensor, torch.tensor(Y_test,dtype=torch.long))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    best_test_loss = float('inf')
    epochs = args.epochs

    for epoch in range(epochs):
        train_losses = []
        test_losses = []
        for X, Y in dataloader:  
            X, Y = X.to(device), Y.to(device)
            classifier.train()
            pred_out = classifier(X)
            classifier.zero_grad()
            loss = loss_function(pred_out.view(len(X),-1), Y)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_losses.append(loss_value)
        
        for X, Y in test_dataloader:
            X, Y = X.to(device), Y.to(device)
            classifier.eval()
            pred_out = classifier(X)
            loss = loss_function(pred_out.view(len(X),-1), Y)
            loss_value = loss.item()
            test_losses.append(loss_value)
            
    mean_test_loss = np.mean(test_losses)
    print(f'Results for epoch {epoch}:')
    print(f'Mean train loss for epoch: {np.mean(train_losses)}')
    print(f'Mean test loss for epoch: {mean_test_loss}')

    if mean_test_loss < best_test_loss:
        best_test_loss = mean_test_loss
        torch.save(auto.state_dict(), f'classifier_{args.classifier_tag}.pt')  
        print(f'Model saved at epoch {epoch} with test loss {mean_test_loss}')

    classifier.eval()
    pred_outputs = []
    for tensor_ in autoencoder_test_tensor:
        encode_output = classifier(torch.unsqueeze(tensor_, dim=0))
        pred_class = np.argmax(encode_output.detach().numpy())
        pred_outputs.append(pred_class)

    evaluate(Y_test, pred_outputs)