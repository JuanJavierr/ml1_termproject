import torch.nn as nn

class CLASSIFIER(nn.Module):
    def __init__(self, k=5, num_class=4):
        super().__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv1 = self.conv_block(512, 256, k=k)
        self.conv2 = self.conv_block(256, 128, k=k)
        self.conv3 = self.conv_block(128, 64, k=k)
        self.conv4 = self.conv_block(64, 32, k=k)
        self.conv5 = self.conv_block(32, 1, k=k)
        self.num_class = num_class
        self.fc = None
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        conv1_out = self.maxpool(self.conv1(x))
        conv2_out = self.maxpool(self.conv2(conv1_out))
        conv3_out = self.maxpool(self.conv3(conv2_out))
        conv4_out = self.maxpool(self.conv4(conv3_out))
        conv5_out = self.maxpool(self.conv5(conv4_out))

        if self.fc is None:
            self.fc = nn.Linear(conv5_out.shape[-1], self.num_class).to(conv5_out.device)
        out = self.softmax(self.fc(conv5_out))
        return out

    @staticmethod
    def conv_block(in_channels, out_channels, k=5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(out_channels)
        )
        return block