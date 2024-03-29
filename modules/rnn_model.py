import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return self.softmax(out)
