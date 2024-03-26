import torch
import torch.nn as nn
from torch.nn import functional as F

class AUTOENCODER(nn.Module):

    def __init__(self, k=5, p=0.5):
        super().__init__()

        #encoder pooling operation
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=p)
        #encoder convolutional blocks
        self.conv1 = self.conv_block(1, 32, k=k)
        self.conv2 = self.conv_block(32, 64, k=k)
        self.conv3 = self.conv_block(64, 128, k=k)
        self.conv4 = self.conv_block(128, 256, k=k)
        self.conv5 = self.conv_block(256, 512, k=k)
        self.convb = self.conv_block(512, 512, k=k)
        #decoder convolutional blocks
        self.dconv4 = self.conv_block(512, 256, k=k)
        self.dconv3 = self.conv_block(256, 128, k=k)
        self.dconv2 = self.conv_block(128, 64, k=k)
        self.dconv1 = self.conv_block(64, 32, k=k)
        self.dconvf = self.final_block(32, 1, k=k)
        #decoder upsampling operations
        self.upsample4 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        #output function
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # Encoder layer 1
        x = self.maxpool(self.conv1(x))
        # Encoder layer 2
        x = self.maxpool(self.conv2(x))
        # Encoder layer 3
        x = self.maxpool(self.conv3(x))
        # Encoder layer 4
        x = self.maxpool(self.conv4(x))
        # Encoder layer 5
        x = self.conv5(x)
        # Encoder bottleneck layer
        encoded_features = self.convb(x)
        return encoded_features

    def forward(self, x):
        init_shape = x.shape[2]
        #encoder layer 1
        block1 = self.conv1(x)
        x = self.maxpool(block1)
        
        #encoder layer 2
        block2 = self.conv2(x) 
        x = self.maxpool(block2)

        #encoder layer 3
        block3 = self.conv3(x) 
        x = self.maxpool(block3)

        #encoder layer 4
        block4 = self.conv4(x) 
        x = self.maxpool(block4)

        #encoder layer 5
        block5 = self.conv5(x) 
        x = self.maxpool(block5)

        #encoder layer 6
        block6 = self.convb(x) 
        x = block6

        #decoder layer 4
        upsamp4 = self.upsample4(x)
        upsamp4 = F.interpolate(upsamp4, block4.shape[2])
        cat4 = torch.cat((upsamp4, block4), 1)
        x = self.dconv4(cat4)
        
        #decoder layer 3
        upsamp3 = self.upsample3(x)
        upsamp3 = F.interpolate(upsamp3, block3.shape[2])
        cat3 = torch.cat((upsamp3, block3), 1)
        x = self.dconv3(cat3)

        #decoder layer 2
        upsamp2 = self.upsample2(x)
        upsamp2 = F.interpolate(upsamp2, block2.shape[2])
        cat2 = torch.cat((upsamp2, block2), 1)
        x = self.dconv2(cat2)

        #decoder layer 1
        upsamp1 = self.upsample1(x)
        upsamp1 = F.interpolate(upsamp1, block1.shape[2])
        cat1 = torch.cat((upsamp1, block1), 1)
        x = self.dconv1(cat1)

        #decoder layer f (final layer)
        x = self.dconvf(x)
        x = F.interpolate(x, init_shape)
        return x

    @staticmethod
    def conv_block(in_channels, out_channels, k=5):
        block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, padding='same'),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=k, groups=out_channels, padding='same'),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
        )
        return block

    @staticmethod
    def final_block(in_channels, out_channels, k=1):
        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
        )
        return block