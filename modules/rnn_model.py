import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_first=True, dropout=0, nonlinearity='relu', bidirectional=True, num_layers=1):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first, nonlinearity=nonlinearity, dropout=dropout, bidirectional=bidirectional, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out, _ = self.rnn(x)  
        out = self.fc(out[:, -1, :])
        return self.softmax(out)
    

class TextLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_first=True, dropout=0, bidirectional=True, num_layers=1):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out, _ = self.lstm(x)  
        out = self.fc(out[:, -1, :])  
        return self.softmax(out)