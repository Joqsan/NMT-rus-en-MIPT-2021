import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        output, enc_hidden = self.rnn(embedded)
        
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        return output, enc_hidden


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
            in_features=2*hid_dim,
            out_features=output_dim
        )
        

        self.dropout = nn.Dropout(p=dropout)# <YOUR CODE HERE>

    def forward(self, dec_input, dec_hidden, enc_output):
        # input = [batch_size]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        
        # input = [seq_len, batch size] =  [1, batch_size]
        #   (because we are predicting character by character)
        dec_input = dec_input.unsqueeze(0)

        # embedded = [1, batch size, emb_dim]
        dec_emb = self.dropout(self.embedding(dec_input))# <YOUR CODE HERE>

        
        # output = [1, batch_size, hid_dim]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        dec_output, dec_hidden = self.rnn(dec_emb, dec_hidden)
        
        
        ## ATTENTION PART
        enc_output_p = enc_output.permute(1, 2, 0)
        dec_output_p = dec_output.permute(1, 0, 2)
        
        # Alligment scores
        scores = dec_output_p.bmm(enc_output_p)
        
        # Attention scores
        alpha = F.softmax(scores, dim=-1)
        
        enc_output_pp = enc_output.permute(1, 0, 2)
        
        # Context vector
        context = alpha.bmm(enc_output_pp).permute(1, 0, 2)
        
        # Concatenation
        concat = torch.cat([context, dec_output], dim=-1)
        
        # Linear layer
        prediction = self.out(concat.squeeze(0))

        
        # prediction = [batch_size, output_dim]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        return prediction, dec_hidden


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
        
        # enc_output = [sequence_len_en, batch_size, hid_dim]
        # hidden = [n_layers, batch_size, hid_dim]
        # cell   = [n_layers, batch_size, hid_dim]
        enc_output, enc_hidden = self.encoder(src)
      
        # input = [batch_size]. One <sos> for each batch instance
        input = trg[0,:]
        
        dec_hidden = enc_hidden # first hidden states in decoder = last hidden states in encoder
        
        for t in range(1, max_len):
            # prediction = [batch_size, output_dim]
            # hidden = [n_layers, batch_size, hid_dim]
            # cell   = [n_layers, batch_size, hid_dim]
            output, dec_hidden = self.decoder(input, dec_hidden, enc_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1] # get index (= token) of highest value (equiv. to argmax)
            input = (trg[t] if teacher_force else top1)

        return outputs
