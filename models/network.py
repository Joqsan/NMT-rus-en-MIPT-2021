import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.experimental.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
#         self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):

        #src = [sentence_length, batch size]

        # embedded = [sentence_length, batch_size, emb_dim]
        embedded = self.embedding(src)

        embedded = self.dropout(embedded)

        
        # output = [sentence_length, batch_size, hid_dim]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        output, (hidden, cell) = self.rnn(embedded)
        
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        
        # output_dim = len of target vocabulary
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
         

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
      

        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
        

        self.dropout = nn.Dropout(p=dropout)# <YOUR CODE HERE>

    def forward(self, input, hidden, cell):
        # input = [batch_size]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        
        # input = [seq_len, batch size] =  [1, batch_size]
        #   (because we are predicting character by character)
        input = input.unsqueeze(0)

        # embedded = [1, batch size, emb_dim]
        embedded = self.dropout(self.embedding(input))# <YOUR CODE HERE>

        
        # output = [1, batch_size, hid_dim]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # output.squeeze(0) =  [batch_size, hid_dim]
        # prediction = [batch_size, output_dim]
        prediction = self.out(output.squeeze(0))

        
        # prediction = [batch_size, output_dim]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        #src = [sentence_length, batch size]
        #trg = [sentence_length, batch size]
        
  
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        hidden, cell = self.encoder(src)

        # input = [batch_size]. One <sos> for each batch instance
        input = trg[0,:]

        for t in range(1, max_len):
            # prediction = [batch_size, output_dim]
            # hidden = [n_layers, batch_size, hid_dim]
            # cell   = [n_layers, batch_size, hid_dim]
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1] # get index (= token) of highest value (equiv. to argmax)
            input = (trg[t] if teacher_force else top1)

        return outputs
