import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ImgCapDataset import ImgCapDataset
from Models import CNN_Encoder, RNN_Decoder


class BaseModel():
    
    def __init__(self, df, image_address, sentence_length, vocab_size):
        
        self.df = df.sample(frac=1).reset_index(drop=True)
        train_df, val_df = self.train_val_split(self.df)
        train_ds = ImgCapDataset(train_df, image_address)
        val_ds = ImgCapDataset(val_df, image_address)
        
        self.train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
        self.val_dl = DataLoader(val_ds, batch_size=16, shuffle=True)
        
        self.EPOCH = 10
        self.encoder = CNN_Encoder()
        self.decoder = RNN_Decoder(256, sentence_length, vocab_size)
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), 0.0001)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), 0.0001)
        self.LOSS = torch.nn.CrossEntropyLoss()
        
    def train_val_split(self, df):
        split = 4*len(df)//5
        train = df.iloc[:split,:]
        val = df.iloc[split:,:]
        return train, val
    
    def train(self):
        self._training_loop(
            self.EPOCH, self.encoder, self.decoder, self.encoder_optim, 
            self.decoder_optim, self.LOSS, self.train_dl, self.val_dl
        )
        
    def _training_loop(self, epochs, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_f, train_loader, val_loader):
        for epoch in range(epochs):
            loss_train = 0.0
            for x1, x2, y in train_loader:
                out = encoder(x1, x2)
                out = decoder(out, y)
                loss = loss_f(torch.flatten(out, 0, 1), torch.flatten(y))

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                loss_train += loss.item()
                
            loss_val = 0.0
            for x1, x2, y in val_loader:
                out = encoder(x1, x2)
                out = decoder(out, y)
                loss = loss_f(torch.flatten(out, 0, 1), torch.flatten(y))
                loss_val += loss.item()
            print("Epoch", epoch, "Train Loss", loss_train/len(train_loader), "Validation Loss", loss_val/len(val_loader))
        